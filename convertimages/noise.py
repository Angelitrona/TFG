import numpy as np
import os
from torchvision.utils import save_image
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch


def noise_gauss(image, num):
	row,col,ch= image.shape
	mean = 0 # mean of the distribution
	var = 0.1
	sigma = var**0.5 #standard deviation of the distribution
	gauss = np.random.normal(mean,sigma,(row,col,ch))
	gauss = gauss.reshape(row,col,ch)
	noisy = image + gauss
	name = 'noisy_gauss_'+str(num)+'.png'
	save_image(noisy, os.path.join(sample_dir, name), nrow=10)
	return noisy
                                     

sample_dir = './noise_images'
if not os.path.exists(sample_dir):
	os.makedirs(sample_dir)


# Root directory for dataset
dataroot = "./prueba"

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Create the dataset. We use an image folder
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
 
                                                                                                                                                                                                                                                      # Save some real images
for images, _ in dataloader:
	i=0
	for image in images:
		noise_gauss(image,i)
		i+=1
	break
 
 
 
 
 
 
 
 
