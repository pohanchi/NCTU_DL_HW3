import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torchvision
import numpy as np

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import weights_init_normal
from scipy.misc import imsave
import os
import argparse
import datetime
import tensorboardX
from tensorboardX import SummaryWriter

import time
start_time = time.time()

# parameters
#TODO : set up all the parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_interval', type=int, default=25, help='log_interval')
parser.add_argument(
    '--lr', default=[1e-3, 1e-4,1e-5], type=list, help='learning rate')
parser.add_argument('--batch_size', default=100, type=float, help='batch_size')
parser.add_argument('--epoch', default=600, type=int, help='epochs')
parser.add_argument('--size', default=64, type=int, help='hidden')
parser.add_argument(
    '--B_A', default=[5.0, 1.0, 10.0], type=list, help="B_A")
parser.add_argument(
    '--A_B', default=[5.0, 1.0, 10.0], type=list, help="A_B")
parser.add_argument(
    "--folder", default="./cyclegan_", type=str, help="saving_folder")
opt = parser.parse_args()

device = None
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("here")
else:
    device = torch.device('cpu')

for lr in opt.lr:
    for B_A in opt.B_A:
        for A_B in opt.A_B:
            now = datetime.datetime.now()
            now = now.strftime("%Y_%m_%d-%H%M")
            if not os.path.exists(opt.folder + str(now)):
                os.makedirs(opt.folder + str(now))
                folder = opt.folder + now
            epochs = opt.epoch  # number of epochs of training
            batchsize = opt.batch_size  # size of the batches
            animation_root = "./dataset_2_question"  # root directory of the dataset
            cartoon_root = "./dataset"  # root directory of the dataset
            size = opt.size  # size of the data crop (squared assumed)
            input_nc = 3  # number of channels of input data
            output_nc = 3  # number of channels of output data
            lamda_A_B = A_B
            lamda_B_A = B_A
            f_h = open(folder + "/hyperparam.txt", "a")
            print("learning rate: {}".format(lr), file=f_h)
            print("Current time: ", datetime.datetime.now(), file=f_h)
            print("epochs: ", epochs, file=f_h)
            print("batch size: ", opt.batch_size, file=f_h)
            print("folder: ", folder, file=f_h)
            print("lamda_B_A: ", lamda_B_A, file=f_h)
            print("lamda_A_B: ", lamda_A_B, file=f_h)
            f_h.close()
            writer = SummaryWriter(folder)

            ###### Definition of variables ######
            # Networks
            netG_A2B = Generator(input_nc, output_nc)
            netG_B2A = Generator(output_nc, input_nc)
            netD_A = Discriminator(input_nc)
            netD_B = Discriminator(output_nc)

            netG_A2B = netG_A2B.to(device)
            netG_B2A = netG_B2A.to(device)
            netD_A = netD_A.to(device)
            netD_B = netD_B.to(device)

            # Lossess
            criterion_GAN = torch.nn.MSELoss()
            criterion_cycle = torch.nn.L1Loss()
            criterion_identity = torch.nn.L1Loss()

            # Optimizers
            optimizer_G = torch.optim.Adam(
                itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                lr=lr,
                betas=(0.5, 0.999))
            optimizer_D_A = torch.optim.Adam(
                netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
            optimizer_D_B = torch.optim.Adam(
                netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

            # Inputs & targets memory allocation
            tensor_fake = torch.tensor((), dtype=torch.float64, device=device)
            tensor_real = torch.tensor((), dtype=torch.float64, device=device)

            fake_A_buffer = ReplayBuffer()
            fake_B_buffer = ReplayBuffer()

            # Dataset loader
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            animation_set = torchvision.datasets.ImageFolder(
                animation_root, transform)
            cartoon_set = torchvision.datasets.ImageFolder(
                cartoon_root, transform)
            animation_loader = torch.utils.data.DataLoader(
                dataset=animation_set, batch_size=batchsize, shuffle=True)
            cartoon_loader = torch.utils.data.DataLoader(
                dataset=cartoon_set, batch_size=batchsize, shuffle=True)
            ###################################
            G_loss = []
            DA_loss = []
            DB_loss = []
            ###### Training ######
            for epoch in range(epochs):
                i = 1
                print('epoch', epoch)
                for batch in zip(animation_loader, cartoon_loader):
                    # Set model input
                    A = batch[0][0].to(device)
                    B = batch[1][0].to(device)

                    target_real = tensor_real.new_ones((A.size(0), 1)).float()
                    target_fake = tensor_fake.new_ones((B.size(0), 1)).float()
                    real_A = A
                    real_B = B

                    ###### Generators A2B and B2A ######
                    optimizer_G.zero_grad()

                    # Identity loss
                    # G_A2B(B) should equal B if real B is fed
                    same_B = netG_A2B.forward(real_B)
                    loss_identity_B = criterion_identity(same_B,
                                                         real_B) * lamda_A_B
                    # G_B2A(A) should equal A if real A is fed
                    same_A = netG_B2A.forward(real_A)
                    loss_identity_A = criterion_identity(same_A,
                                                         real_A) * lamda_A_B

                    # GAN loss
                    fake_B = netG_A2B.forward(real_A)
                    pred_fake = netD_B.forward(fake_B)
                    loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

                    fake_A = netG_B2A.forward(real_B)
                    pred_fake = netD_A.forward(fake_A)
                    loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

                    # Cycle loss
                    recovered_A = netG_B2A.forward(fake_B)
                    loss_cycle_ABA = criterion_cycle(recovered_A,
                                                     real_A) * lamda_B_A

                    recovered_B = netG_A2B.forward(fake_A)
                    loss_cycle_BAB = criterion_cycle(recovered_B,
                                                     real_B) * lamda_B_A

                    # Total loss
                    loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                    loss_G.backward()

                    optimizer_G.step()
                    ###################################

                    ###### Discriminator A ######
                    optimizer_D_A.zero_grad()

                    # Real loss
                    pred_real = netD_A(real_A)
                    loss_D_real = criterion_GAN(pred_real, target_real)

                    # Fake loss
                    fake_A = fake_A_buffer.push_and_pop(fake_A)
                    pred_fake = netD_A.forward(fake_A.detach())
                    loss_D_fake = criterion_GAN(pred_fake, target_fake)

                    # Total loss
                    loss_D_A = (loss_D_real + loss_D_fake) * 0.5
                    loss_D_A.backward()

                    optimizer_D_A.step()
                    ###################################

                    ###### Discriminator B ######
                    optimizer_D_B.zero_grad()

                    # Real loss
                    pred_real = netD_B(real_B)
                    loss_D_real = criterion_GAN(pred_real, target_real)

                    # Fake loss
                    fake_B = fake_B_buffer.push_and_pop(fake_B)
                    pred_fake = netD_B.forward(fake_B.detach())
                    loss_D_fake = criterion_GAN(pred_fake, target_fake)

                    # Total loss
                    loss_D_B = (loss_D_real + loss_D_fake) * 0.5
                    loss_D_B.backward()

                    optimizer_D_B.step()

                    ###################################

                    G_loss.append(loss_G.data.item())
                    DA_loss.append(loss_D_A.data.item())
                    DB_loss.append(loss_D_B.data.data.item())
                    # Progress report
                    if ( i %100 == 0):
                        print("loss_G : ",
                              loss_G.data.cpu().numpy(), ",loss_D:",
                              (loss_D_A.data.cpu().numpy() +
                               loss_D_B.data.cpu().numpy()))
                        writer.add_scalars('loss_G',{'train':(loss_G.data.cpu().numpy())},epoch)
                        writer.add_scalars('loss_D',{'train':(loss_D_A.data.cpu().numpy() +
                               loss_D_B.data.cpu().numpy())},epoch)

                        
                    if (i % 100 == 0 and epoch % 25 == 0):
                        real_B_image = real_B.permute(0, 2, 3, 1).numpy()
                        real_A_image = real_A.permute(0, 2, 3, 1).numpy()
                        
                        image_B = netG_A2B(real_A).permute(0, 2, 3,
                                                           1).detach().numpy()
                        image_A = netG_B2A(real_B).permute(0, 2, 3,
                                                           1).detach().numpy()
                        map_1 = np.zeros((32 * 10, 32 * 16, 3))
                        for j in range(0, 16, 2):
                            for k in range(10):
                                map_1[k * 32:(k + 1) * 32, j * 32:(j + 1) *
                                      32, :] = image_B[j * 5 + k]
                                map_1[k * 32:(k + 1) * 32, (j + 1) * 32:(
                                    j + 2) * 32, :] = real_B_image[j * 5 + k]
                        if not os.path.exists(folder + "/result_A2B"):
                            os.makedirs(folder + "/result_A2B")
                        imsave(
                            folder +
                            "/result_A2B/exp_epoch_{}_batch_{}.png".format(
                                epoch, i), map_1)
                        map_2 = np.zeros((32 * 10, 32 * 16, 3))
                        for j in range(0, 16, 2):
                            for d in range(10):
                                map_2[d * 32:(d + 1) * 32, j * 32:(j + 1) *
                                      32, :] = image_A[j * 5 + d]
                                map_2[d * 32:(d + 1) * 32, (j + 1) * 32:(
                                    j + 2) * 32, :] = real_A_image[j * 5 + d]
                        if not os.path.exists(folder + "/result_B2A"):
                            os.makedirs(folder + "/result_B2A")
                        imsave(
                            folder +
                            "/result_B2A/exp_epoch_{}_batch_{}.png".format(
                                epoch, i), map_2)
                        if not os.path.exists(folder + "/ckpt"):
                            os.makedirs(folder + "/ckpt")
                        torch.save(netG_A2B.state_dict(),
                                   folder + '/ckpt/netG_A2B.pth')
                        torch.save(netG_B2A.state_dict(),
                                   folder + '/ckpt/netG_B2A.pth')
                        torch.save(netD_A.state_dict(),
                                   folder + '/ckpt/netD_A.pth')
                        torch.save(netD_B.state_dict(),
                                   folder + '/ckpt/netD_B.pth')
                        i = 0
                    i = i + 1
                # Save models checkpoints

            end_time = time.time()
            print(
                'Total cost time',
                time.strftime("%H hr %M min %S sec",
                              time.gmtime(end_time - start_time)))

# TODO : plot the figure
