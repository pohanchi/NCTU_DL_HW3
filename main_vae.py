import torch 
import numpy  as np 
import PIL
from PIL import Image 
import matplotlib.pyplot as plt 
import torchvision
from vae_model.model import VAE
import argparse
import tqdm
import os
import datetime
from torchvision import transforms, datasets
import tensorboardX
from tensorboardX import SummaryWriter



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_interval', type=int, default=25, help='log_interval')
    parser.add_argument('--lr', default=[1e-4, 1e-3, 8e-3, 1e-5], type=list, help='learning rate')
    parser.add_argument('--batch_size', default=100, type=float, help='batch_size')
    parser.add_argument('--epoch', default=500, type=int, help='epochs')
    parser.add_argument('--z', default=[10,20,40,80], type=list, help='hidden')

    parser.add_argument(
        "--folder", default="./vae_summary_", type=str, help="saving_folder")
    opt = parser.parse_args()

    for latent_z in opt.z:
        for lr in opt.lr:
            now=datetime.datetime.now()
            now = now.strftime("%Y_%m_%d-%H%M")
            if not os.path.exists(opt.folder+str(now)):
                os.makedirs(opt.folder+str(now))
                folder =opt.folder+now
            f_h = open(folder + "/hyperparam.txt", "a")
            print("learning rate: {}".format(lr), file=f_h)
            print("Current time: ", datetime.datetime.now(), file=f_h)
            print("latent dim: ",latent_z, file=f_h)
            print("log_interval:", opt.log_interval, file=f_h)
            print("batch size: ", opt.batch_size, file=f_h)
            print("folder: ", folder, file=f_h)
            f_h.close()
            writer = SummaryWriter(folder)

            device = None
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')

            data_transform = transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                ])

            train_dataset = datasets.ImageFolder(root='./dataset',
                                                    transform=data_transform)
            dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=opt.batch_size, shuffle=True)

            model = VAE(latent_z).to(device)
            epoch = opt.epoch
            optim = torch.optim.Adam(model.parameters(), lr=lr) 
            model = model.train()


            for i in tqdm.tqdm(range(epoch)):
                model.train_a_epoch(dataloader,optim, opt, i, writer)
                if (i% opt.log_interval == 0):
                    model.eval_some_sample(dataloader,opt, i, folder)
                    if not os.path.exists(folder+"/vae_tuning"):
                        os.makedirs(folder+"/vae_tuning")
                    torch.save(model.state_dict(), folder+'./vae_tuning/model_batch_{}'.format(i))
            
    




