from __future__ import print_function
import os,time, pickle, argparse, random, sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision.utils import save_image
from IPython.display import HTML

from generator import Generator
from discriminator import Discriminator
from VGG19 import VGG19
from noise import noise_gauss as noise
from auxiliary_func import generate_data_ld
from glob import glob

import joblib #save the models


manualSeed = 999 # Set random seed for reproducibility
random.seed(manualSeed)
torch.manual_seed(manualSeed)


srcroot = './data/cartoon2D'#"/home/gines/Escritorio/TFG/data/cartoon2D" # Root directory for src_dataset
tgtroot = './data/cartoon3D'#"/home/gines/Escritorio/TFG/data/cartoon3D" # Root directory for tgt_dataset
vggroot = "./vgg19-train.pth"

workers = 2 # Number of workers for dataloader
batch_size = 2#128 # Batch size during training
num_epochs = 4 # Number of training epochs
num_epochs_pretrain = 250 #TODO CAMBIAR
lr = 0.0002 # Learning rate for optimizers
beta1 = 0.5 # Beta1 hyperparam for Adam optimizers
ngpu = 1 # Number of GPUs available. Use 0 for CPU mode.
lambda_loss = 10 # Lambda for content loss
name = 'starwars_gan'
in_size = 5 #TODO cambiar con el tamaño de la entrada!

# All training images will be resized to this size using a transformer.
image_size_w = 480 
image_size_h = 204

# Generator parameters
in_nc_g = 3 # Input channel for generator (for color images=3)
out_nc_g = 3 # Output channel for generator
ngf = 64 # Size of feature maps in generator
nblockl_layer = 8 #Number of resnet block layer for generator

# Discriminator parameters
in_nc_d = 3 # Input channel for discriminator
out_nc_d = 1 # Output channel for discriminator
ndf = 32 # Size of feature maps in discriminator


# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("Device: ",device)


# Results save path
if not os.path.isdir(os.path.join(name + '_results', 'Reconstruction')):
    os.makedirs(os.path.join(name + '_results', 'Reconstruction'))
if not os.path.isdir(os.path.join(name + '_results', 'Transfer')):
    os.makedirs(os.path.join(name + '_results', 'Transfer'))


# Dataloaders
transform = transforms.Compose([transforms.Resize((image_size_h, image_size_w)),  transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #transforms.CenterCrop(image_size),

train_src_dataloader = generate_data_ld(srcroot, 'train', transform, batch_size, shuffle=True, drop_last=True)

train_tgt_dataloader = generate_data_ld(tgtroot, 'train', transform, batch_size, shuffle=True, drop_last=True)

test_src_dataloader = generate_data_ld(srcroot, 'test', transform, 1, shuffle=True, drop_last=True)



# GENERATOR
netG = Generator(in_nc_g, out_nc_g, ngf, nblockl_layer).to(device) 

already_trained_g = False
if len(sys.argv)>1:
    already_trained_g = True
    netG.load_state_dict(torch.load(sys.argv[1]))
    
if (device.type == 'cuda') and (ngpu > 1): # Handle multi-gpu if desired
    netG = nn.DataParallel(netG, list(range(ngpu)))
#print(netG) # Print the model
netG.train()

# DISCRIMINATOR
netD = Discriminator(in_nc_d, out_nc_d, ndf).to(device)

if (device.type == 'cuda') and (ngpu > 1): # Handle multi-gpu if desired
    netD = nn.DataParallel(netD, list(range(ngpu)))
#print(netD) # Print the model
netD.train()

# VGG19
vgg = VGG19(init_weights=vggroot, feature_mode=True).to(device)

# Initialize BCELoss, L1Loss function
bce_loss = nn.BCELoss().to(device)
l1_loss = nn.L1Loss().to(device)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

G_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[num_epochs // 2, num_epochs // 4 * 3], gamma=0.1)
D_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[num_epochs // 2, num_epochs // 4 * 3], gamma=0.1)

# PRE-TRAINING
pre_train_hist = {}
pre_train_hist['Recon_loss'] = []
pre_train_hist['per_epoch_time'] = []
pre_train_hist['total_time'] = []

""" Pre-train reconstruction """
if not already_trained_g:
    print('Pre-training start!')
    start_time = time.time()
    for epoch in range(num_epochs_pretrain):
        epoch_start_time = time.time()
        Recon_losses = []
        for x, _ in train_src_dataloader:
            x = x.to(device)

            # train generator G
            optimizerG.zero_grad()

            x_feature = vgg((x + 1) / 2)
            G_, _ = netG(x)
            G_feature = vgg((G_ + 1) / 2)
            Recon_loss = 10 * l1_loss(G_feature, x_feature.detach())
            Recon_losses.append(Recon_loss.item())
            pre_train_hist['Recon_loss'].append(Recon_loss.item())

            Recon_loss.backward()
            optimizerG.step()
           
        per_epoch_time = time.time() - epoch_start_time
        pre_train_hist['per_epoch_time'].append(per_epoch_time)
        print('[%d/%d] - time: %.2f, Recon loss: %.3f' % ((epoch + 1), num_epochs_pretrain, per_epoch_time, torch.mean(torch.FloatTensor(Recon_losses))))

    total_time = time.time() - start_time
    pre_train_hist['total_time'].append(total_time)
    with open(os.path.join(name + '_results',  'pre_train_hist.pkl'), 'wb') as f:
        pickle.dump(pre_train_hist, f)

    with torch.no_grad():
        netG.eval()
        for n, (x, _) in enumerate(train_src_dataloader):
            x = x.to(device)
            G_recon, _ = netG(x)
            result = torch.cat((x[0], G_recon[0]), 2)
            path = os.path.join(name + '_results', 'Reconstruction', name + '_train_recon_' + str(n + 1) + '.png')
            plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            if n == 4:
                break

        for n, (x, _) in enumerate(test_src_dataloader):
            x = x.to(device)
            G_recon, _ = netG(x)
            result = torch.cat((x[0], G_recon[0]), 2)
            path = os.path.join(name + '_results', 'Reconstruction', name + '_test_recon_' + str(n + 1) + '.png')
            plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            if n == 4:
                break
else:
    print('Load the latest generator model, no need to pre-train')


# TRAINING LOOP
train_hist = {}
train_hist['Disc_loss'] = []
train_hist['Gen_loss'] = []
train_hist['Con_loss'] = []
train_hist['per_epoch_time'] = []
train_hist['total_time'] = []
print('Training start!')
start_time = time.time()

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.
real = torch.ones(batch_size, 1, in_size // 4, in_size // 4).to(device)
real = torch.ones(batch_size, 1, 51, 2).to(device)#TODO quitqra linea pq??
real_f = torch.ones(batch_size, 1, 51, 120).to(device)#TODO quitqra linea pq??

fake = torch.zeros(batch_size, 1, in_size // 4, in_size // 4).to(device)
fake = torch.zeros(batch_size, 1, 51, 120).to(device)#TODO quitqra linea
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    netG.train()
    G_scheduler.step()
    D_scheduler.step()
    Disc_losses = []
    Gen_losses = []
    Con_losses = []
    for (x, _), (y, _) in zip(train_src_dataloader, train_tgt_dataloader):
        e = y[:, :, :, in_size:]
        y = y[:, :, :, :in_size]
        x, y, e = x.to(device), y.to(device), e.to(device)

        # train D
        optimizerD.zero_grad()

        D_real = netD(y)
        D_real_loss = bce_loss(D_real, real)

        G_, dec= netG(x)
        print(dec.size(), ' - ', G_.size())
        D_fake = netD(G_)
        D_fake_loss = bce_loss(D_fake, fake) #min

        Disc_loss = D_real_loss + D_fake_loss  #TODO se pueden aplicar pesos en las losses
        Disc_losses.append(Disc_loss.item())
        train_hist['Disc_loss'].append(Disc_loss.item())

        Disc_loss.backward()
        optimizerD.step()

        # train G
        optimizerG.zero_grad()

        G_, _ = netG(x)
        D_fake = netD(G_)
        D_fake_loss = bce_loss(D_fake, real_f) #max

        x_feature = vgg((x + 1) / 2)
        G_feature = vgg((G_ + 1) / 2)
        Con_loss = lambda_loss * l1_loss(G_feature, x_feature.detach())

        Gen_loss = D_fake_loss + Con_loss
        Gen_losses.append(D_fake_loss.item())
        train_hist['Gen_loss'].append(D_fake_loss.item())
        Con_losses.append(Con_loss.item())
        train_hist['Con_loss'].append(Con_loss.item())

        Gen_loss.backward()
        optimizerG.step()


    per_epoch_time = time.time() - epoch_start_time
    train_hist['per_epoch_time'].append(per_epoch_time)
    print(
    '[%d/%d] - time: %.2f, Disc loss: %.3f, Gen loss: %.3f, Con loss: %.3f' % ((epoch + 1), num_epochs, per_epoch_time, torch.mean(torch.FloatTensor(Disc_losses)),
        torch.mean(torch.FloatTensor(Gen_losses)), torch.mean(torch.FloatTensor(Con_losses))))

    if epoch % 2 == 1 or epoch == num_epochs- 1:
        with torch.no_grad():
            netG.eval()
            for n, (x, _) in enumerate(train_src_dataloader):
                x = x.to(device)
                G_recon, _ = netG(x)
                result = torch.cat((x[0], G_recon[0]), 2)
                path = os.path.join(name + '_results', 'Transfer', str(epoch+1) + '_epoch_' + name + '_train_' + str(n + 1) + '.png')
                plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                if n == 4:
                    break

            for n, (x, _) in enumerate(test_src_dataloader):
                x = x.to(device)
                G_recon, _ = netG(x)
                result = torch.cat((x[0], G_recon[0]), 2)
                path = os.path.join(name + '_results', 'Transfer', str(epoch+1) + '_epoch_' + name + '_test_' + str(n + 1) + '.png')
                plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                if n == 4:
                    break

            torch.save(netG.state_dict(), os.path.join(name + '_results', 'generator_latest.pkl'))
            torch.save(netD.state_dict(), os.path.join(name + '_results', 'discriminator_latest.pkl'))

total_time = time.time() - start_time
train_hist['total_time'].append(total_time)

print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_time'])), num_epochs, total_time))
print("Training finish!... save training results")

torch.save(netG.state_dict(), os.path.join(name + '_results',  'generator_param.pkl'))
torch.save(netD.state_dict(), os.path.join(name + '_results',  'discriminator_param.pkl'))
with open(os.path.join(name + '_results',  'train_hist.pkl'), 'wb') as f:
    pickle.dump(train_hist, f)

