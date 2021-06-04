import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import os
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from IPython.display import Image
from torchvision.utils import save_image
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from discriminator import Discriminator
from generator import Generator

# from sklearn.externals import joblib
# import sklearn.external.joblib as extjoblib
import joblib #save the models



image_size = 64
#Load the data
cifar = CIFAR10(root='data', 
              train=True, 
              download=True,
              transform=Compose([Resize(image_size), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
              p


#Create a dataloader
batch_size = 100
data_loader = DataLoader(cifar, batch_size, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# DISCRIMINATOR
ngpu = 1
nchanels = 3
D = Discriminator(ngpu,nchanels).to(device)

 
# DISCRIMINATROR TRAINING
criterion = nn.BCELoss()
lr = 0.0002
beta1 = 0.5
d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

def train_discriminator(images):
    # Create the labels which are later used as input for the BCE loss
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)
        
    # Loss for real images
    outputs = D(images)
    print(outputs.size())
    d_loss_real = criterion(outputs, real_labels)
    real_score = outputs

    # Loss for fake images
    z = torch.randn(batch_size, latent_size).to(device)
    fake_images = G(z)
    outputs = D(fake_images)
    d_loss_fake = criterion(outputs, fake_labels)
    fake_score = outputs

    # Combine losses
    d_loss = d_loss_real + d_loss_fake
    # Reset gradients
    reset_grad()
    # Compute gradients
    d_loss.backward()
    # Adjust the parameters using backprop
    d_optimizer.step()
    
    return d_loss, real_score, fake_score

# output 1=real CIFAR dataset
# output 0=generated




# GENERATOR
latent_size = 64

G = Generator(ngpu, nchanels).to(device)


# GENERATOR TRAINING
g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

def train_generator():
    # Generate fake images and calculate loss
    z = torch.randn(batch_size, latent_size).to(device)
    fake_images = G(z)
    labels = torch.ones(batch_size, 1).to(device)
    g_loss = criterion(D(fake_images), labels)

    # Backprop and optimize
    reset_grad()
    g_loss.backward()
    g_optimizer.step()
    return g_loss, fake_images
    
    
# TRAINING BOTH

# Before training we are going to save some images
sample_dir = 'images_pretrainig'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

'''      
# Save some real images
for images, _ in data_loader:
    images = images.reshape(images.size(0), 3, 28, 28)
    save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'), nrow=10)
    break
'''
   
Image(os.path.join(sample_dir, 'real_images.png'))    


sample_vectors = torch.randn(batch_size, latent_size).to(device)

def save_fake_images(index):
    fake_images = G(sample_vectors)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    print('Saving', fake_fname)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=10)
    
#save_fake_images(0)
Image(os.path.join(sample_dir, 'fake_images-0000.png'))

dir_carpeta = './pretraining'

def save_fake_image_custom(images, num):
	num_int = 0
	for i in images:
		fake_fname = 'fake_images_'+str(num)+'_'+str(num_int)+'.png'
		save_image(i, os.path.join(dir_carpeta, fake_fname))
		num_int += 1
		
		fake_fname = 'fake_images_'+str(num)+'.png'
		save_image(images, os.path.join(dir_carpeta, fake_fname))


# Training
num_epochs = 300
total_step = len(data_loader)
d_losses, g_losses, real_scores, fake_scores = [], [], [], []

num = 0
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # Load a batch & transform to vectors
        images = images.to(device)#reshape(batch_size, -1).to(device)
        
        # Train the discriminator and generator
        d_loss, real_score, fake_score = train_discriminator(images)
        g_loss, fake_images = train_generator()
        save_fake_image_custom(fake_image, num)
        
        
        
        # Inspect the losses
        if (i+1) % 200 == 0:
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            real_scores.append(real_score.mean().item())
            fake_scores.append(fake_score.mean().item())
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
        
    num +=1
    # Sample and save images
    save_fake_images(epoch+1)

# Save the model checkpoints 
torch.save(G.state_dict(), 'G_cifar.ckpt')
torch.save(D.state_dict(), 'D_cifar.ckpt')

# Save the model
joblib.dump(G, 'G_train_cifar10.pkl') 
joblib.dump(D, 'D_train_cifar10.pkl') 







