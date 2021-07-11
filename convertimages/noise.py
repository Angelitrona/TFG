import numpy as np
import os
from torchvision.utils import save_image
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch


def noise_gauss(images, device):
	num=0
	for img in images:
		ch,row,col= img.shape
		mean = 0 # mean of the distribution
		var = 0.6
		sigma = var**0.9 #standard deviation of the distribution
		gauss = np.random.normal(mean,sigma,(ch,row,col))
		gauss = gauss.reshape(ch,row,col)
		
		
		img_np = img.cpu().numpy() #data.cpu().numpy() #torch.numpy(img).cpu()
		gaus_torch = torch.from_numpy(gauss).to(device)
		result_np = img_np + gauss
		result_torch = img + gaus_torch
		
		
		img = result_torch
		#name = 'noisy_gauss_dcgan'+str(num)+'.png'
		#save_image(result_torch, os.path.join('./noise_dcgan', name), nrow=10)
	#return images
	
	
'''
sample_dir = './noise_images'         
if not os.path.exists(sample_dir):
	os.makedirs(sample_dir)
	
	
def noise_gauss_sv(image, num):
	ch,row,col= image.shape
	mean = 0 # mean of the distribution
	var = 0.1
	sigma = var**0.5 #standard deviation of the distribution
	gauss = np.random.normal(mean,sigma,(ch,row,col))
	gauss = gauss.reshape(ch,row,col)
	noisy = image + gauss
	name = 'noisy_gauss_'+str(num)+'.png'
	save_image(noisy, os.path.join(sample_dir, name), nrow=10)
	return noisy
                            


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
		noise_gauss_sv(image,i)
		i+=1
	break
 
'''
 
 
 
 
 
 
