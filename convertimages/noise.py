import numpy as np
import os
from torchvision.utils import save_image
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

