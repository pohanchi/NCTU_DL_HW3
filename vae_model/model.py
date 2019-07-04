import torch 
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import functional as F
from imageio import imsave
import os

class VAE(nn.Module):
    def __init__(self,h_dim):
        super(VAE, self).__init__()

        #encoder
        self.fc1 = nn.Linear(3*64*64, 4096)
        # 16 x 256 x 256
        self.fc2 = nn.Linear(4096, 500)
        # 16 x 256 x 256
        self.fc3 = nn.Linear(500, 150)

        #latent space
        self.W_mean = nn.Linear(150 , h_dim)
        self.W_var = nn.Linear(150, h_dim)

        #decoder 
        self.dec_w1 = nn.Linear(h_dim, 500)
        self.dec_w2 = nn.Linear(500, 4096)
        self.dec_w3 = nn.Linear(4096, 3*64*64)
        
        #activation_function
        self.activation_f = nn.ReLU()

    def encode(self, x):
        self.layer1=self.activation_f(self.fc1(x))
        self.layer2=self.activation_f(self.fc2(self.layer1))
        self.output = self.activation_f(self.fc3(self.layer2))
        self.latent_mean = self.W_mean(self.output)
        self.latent_var  = self.W_var(self.output)
        return self.latent_mean, self.latent_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        self.dec_layer1 = self.activation_f(self.dec_w1(z))
        self.dec_layer2 = self.activation_f(self.dec_w2(self.dec_layer1))
        self.pack2= self.dec_w3(self.dec_layer2)
        self.act_out = torch.sigmoid(self.pack2)
        self.pack3= self.act_out.view(-1, 3*64*64)

        return self.pack3

    def forward(self,x):
        encoder_o = self.encode(x)
        self.mu = encoder_o[0]
        self.var= encoder_o[1]
        self.z = self.reparameterize(encoder_o[0], encoder_o[1])
        self.decoder_o = self.decode(self.z)
        return self.mu, self.var, self.z, self.decoder_o

    def loss_f(self,x ):
        recon_x = self.decoder_o
        loss_first = F.mse_loss(recon_x.view(-1, 3*64*64),x, reduction='sum')
        loss_KL = -0.5 * torch.sum(1 + self.var - self.mu.pow(2) - self.var.exp())

        return loss_KL + loss_first

    def train_a_epoch(self, train_loader, optimizer,opt,epoch, writer):
        self = self.train()
        device = None
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        train_loss = 0
        print(len(train_loader))
        for batch_idx, data in enumerate(train_loader):
            
            x = data[0].to(device)
            x = x.view(-1,3*64*64)
            optimizer.zero_grad()
            mu, var, z, reconst_x  = self.forward(x)
            loss = self.loss_f(x)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % (opt.batch_size-1) == 0:
                writer.add_scalars('loss',{'train':(loss.item()/opt.batch_size)},epoch)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader.dataset)/opt.batch_size,
                    100. * batch_idx / len(train_loader),
                    loss.item()/opt.batch_size))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        return 
    
    def eval_some_sample(self, test_loader, opt, epoch, folder):
        self = self.eval()
        device = None
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        for batch_idx, data in enumerate(test_loader):
            x = data[0].to(device)
            x_input = x.view(-1,3*64*64)
            mu, var, z, reconst_x  = self.forward(x_input)
            x = x.permute(0,2,3,1).cpu().numpy()
            reconst_x = reconst_x.view(-1,3,64,64).permute(0,2,3,1).detach().cpu().numpy()
            if batch_idx % 50 == 0:
                map_ =np.zeros((64*10, 64*16,3))
                for j in range(0,16,2):
                    for i in range(10):
                        map_[i*64:(i+1)*64,j*64:(j+1)*64,:] = x[j*5+i]
                        map_[i*64:(i+1)*64,(j+1)*64:(j+2)*64,:] = reconst_x[j*5+i]
                if not os.path.exists(folder+"/result_1"):
                    os.makedirs(folder+"/result_1")
                imsave(folder+"/result_1/exp_epoch_{}_batch_{}.png".format(epoch,batch_idx),map_)

    def extract_some_sample(self, test_loader, opt, epoch, folder):
        self = self.eval()
        device = None
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        x=test_loader[0].to(device)
        x1=test_loader[1].to(device)
        x_input = x.view(-1,3*64*64)
        x1_input = x1.view(-1,3*64*64)
        mu, var, z, reconst_x  = self.forward(x_input)
        mu1, var1, z1, reconst_x1  = self.forward(x1_input)
        reconst_x = reconst_x.view(-1,3,64,64).permute(0,2,3,1).detach().cpu().numpy()
        reconst_x1 = reconst_x.view(-1,3,64,64).permute(0,2,3,1).detach().cpu().numpy()


        if batch_idx % 50 == 0:
            map_ =np.zeros((64*10, 64*16,3))
            for j in range(16):
                z2[0] = ((z1[0] - z[0]) / 16) * j
                z2[2:] = z[2:]
                for i in range(10):
                    z2[1] = ((z1[1] - z[1]) / 10) * i
                    if i and j == 0:
                        map_[i*64:(i+1)*64,j*64:(j+1)*64,:] = reconst_x
                    elif i  == 9 and j==15:
                        map_[i*64:(i+1)*64,(j+1)*64:(j+2)*64,:] = reconst_x1
                    else:
                        map_[i*64:(i+1)*64,(j+1)*64:(j+2)*64,:] = self.decoder_o(z2).view(-1,3,64,64).permute(0,2,3,1).detach().cpu().numpy()

            if not os.path.exists(folder+"/result_sample"):
                os.makedirs(folder+"/result_sample")
            imsave(folder+"/result_sample/exp_epoch_{}_batch_{}.png".format(epoch,batch_idx),map_)

            
            

                



        


        








        
        
