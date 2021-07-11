import os

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose

data_dir = '/home/gines/Escritorio/TFG/data'
classes = os.listdir(data_dir)
aux = classes[0]
classes[0] = classes[2]
classes[2]= classes[1]
classes[1] = aux
print(classes)

transform_custom = Compose(
    [#Resize((816, 1920)),
     Resize((816, 816)),
     ToTensor()])

dataset = ImageFolder(data_dir, transform = transform_custom)

import torch

from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

random_seed = 42
torch.manual_seed(random_seed);

val_size = 10
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)

batch_size=12

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)


import matplotlib
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import numpy as np
import torchvision
# functions to show an image

# print images

def imshow(img, lb):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(classes[lb])
    plt.show()

dataiter = iter(train_dl)
images, labels = dataiter.next()

c0 = False
c1 = False
c2 = False

for i in range(12):
    if (labels[i] == 0 and not c0):
        c0 = True
        imshow(images[i], labels[i])
    if (labels[i] == 1 and not c1):
        c1 = True
        imshow(images[i], labels[i])
    if (labels[i] == 2 and not c2):
        c2 = True
        imshow(images[i], labels[i])
 

# Model creation
import torch.nn as nn
import torch.nn.functional as F

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        print("Images shape:: " + images.shape)
        out = self(images)                  # Generate predictions
        print(out.shape)                  # Generate predictions
        print(out)
        print(labels)
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
    
class StarWarsCnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # input: 3 x 816 x 816
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 32 x 408 x 408
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 204 x 204


            nn.Flatten(), 
            nn.Linear(64*204*204, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3))
        
            #nn.Conv2d(3, 8, kernel_size=3, padding=1),
            #nn.MaxPool2d(2, 2)) # output: 64 x 16 x 16
        
    def forward(self, xb):
        return self.network(xb)

model = StarWarsCnnModel()

for images, labels in train_dl:
    print('images.shape:', images.shape)
    out = model(images)
    print('out.shape:', out.shape)
    break


# Training
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history
    
    
    
device = get_default_device()
print("We are using ", device)
if (device != "cpu"):
	train_dl = DeviceDataLoader(train_dl, device) # podemos poner shuffle = True
	#val_dl = DeviceDataLoader(val_dl, device)
	to_device(model, device);
	print("We move dataloader and the model to ", device)

num_epochs = 2
lr = 0.001
opt_func = torch.optim.Adam

print("Number epochs: ", num_epochs,'\ntrain_dl shape', len(train_dl))


def imshow(img, lb):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(lb)
    plt.show()  

"""
for batch in train_dl:
	imgs, lbs = batch
	for i in range(len(imgs)):
		lb = lbs[i]
		classs = classes[lb]
		imshow(imgs[i], classs)
	break
"""
	
history = fit(num_epochs, lr, model, train_dl, train_dl, opt_func)
history







