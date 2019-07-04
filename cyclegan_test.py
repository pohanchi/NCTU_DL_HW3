from torchvision.utils import save_image
import os
import sys

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
###### Definition of variables ######
# TODO : assign input_nc and output_nc
input_nc = 3
output_nc =3
device = torch.device('cpu')
# Networks
netG_A2B = Generator(input_nc, output_nc,2)
netG_B2A = Generator(output_nc, input_nc,2)

animation_root = "./dataset_2_question"
cartoon_root = "./dataset"

batchsize = 100

# Load state dicts
path = "./cyclegan/cyclegan_2019_06_12-0606/"
netG_A2B.load_state_dict(torch.load(path+'ckpt/netG_A2B.pth',map_location=device))
netG_B2A.load_state_dict(torch.load(path+'ckpt/netG_B2A.pth',map_location=device))

netG_A2B.to(device)
netG_B2A.to(device)

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Dataset loader
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
animation_set = torchvision.datasets.ImageFolder(animation_root, transform) 
cartoon_set = torchvision.datasets.ImageFolder(cartoon_root, transform) 
animation_loader = torch.utils.data.DataLoader(dataset=animation_set,batch_size=batchsize,shuffle=True)
cartoon_loader = torch.utils.data.DataLoader(dataset=cartoon_set,batch_size=batchsize,shuffle=True)


if not os.path.exists(path+'output/animation'):
    os.makedirs(path+'output/animation')
if not os.path.exists(path+'output/cartoon'):
    os.makedirs(path+'output/cartoon')

tensor_fake = torch.tensor((), dtype=torch.float64, device=device)
tensor_real = torch.tensor((), dtype=torch.float64, device=device)

i=0
for batch in zip(animation_loader, cartoon_loader):
    # Set model input
    A = batch[0][0].to(device)
    B = batch[1][0].to(device)

    target_real = tensor_real.new_ones((A.size(0), 1)).float()
    target_fake = tensor_fake.new_ones((B.size(0), 1)).float()
    real_A = A
    real_B = B

    # Generate output
    fake_B = netG_A2B(real_A).data
    fake_A = netG_B2A(real_B).data

    # Save image files
    save_image(real_A, path+'output/animation/real%04d.png' % (i+1))
    save_image(real_B, path+'output/cartoon/real%04d.png' % (i+1))
    save_image(fake_A, path+'output/animation/fake%04d.png' % (i+1))
    save_image(fake_B, path+'output/cartoon/fake%04d.png' % (i+1))

    sys.stdout.write('\rGenerated images %04d' % (i+1))
    i = i+1
    if (i==10):
        break

sys.stdout.write('\n')
