# -*- coding: utf-8 -*-
"""Inicio_TFG.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iZzwuVZais7YdCVtHodiWe4YO94r7Ud4

# Creación de clase imágenes

Si queremos crear una clase de datos particular: https://mlearninglab.com/2020/04/04/datasets-y-dataloader-en-pytorch/ o https://hackernoon.com/procesando-datos-para-deep-learning-datasets-visualizaciones-y-dataloaders-en-pytorch-hu1w36ly

Las clases Dataset permiten instanciar objetos con el conjunto de datos que se van a cargar. PyTorch permite crear dos tipos distintos de datasets:

* Map-style: Implementa los métodos getitem() and len() y representa un mapeo de claves/índices a valores del conjunto de datos. La clase Dataset es un ejemplo.
> * get_item() nos permite hacer "indexing", es decir que StarWarsDataSet[0] nos devuelva el primer elemento del dataset


* Iterable-style: Implementa el método iter() y representa un iterable sobre los datos. La clase IterableDataset es un ejemplo.
"""

from google.colab import drive
 
drive.mount('/content/drive') 
data_dir = './drive/MyDrive/imagenes'


transform = transforms.Compose(
    [torchvision.transforms.ToPILImage(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

import torch
from torch.utils.data import Dataset

class StarWarsDataset(Dataset):
    def __init__(self, imgs, labels, transformaciones = None):
        self.imgs = imgs
        self.labels = labels
        self.transformaciones = transformaciones

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        idx_img = self.imgs[idx]
        idx_lbl = self.labels[idx]

        if self.transformaciones: # aplicacmos la transformacion (normalizar)
            idx_img = self.transformaciones(idx_img)

            #creamos un diccionario que es lo que se devuelve
            muestra = {"imagen" : idx_img,
                      "etiqueta" : idx_lbl}
        return muestra

import tensorflow as tf
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
type(dataSet)
x_train
type(x_train)

"""Podemos crear una instancia de nuestra nueva clase

"""

dataset = NumbersDataset()
print(len(dataset))
print(dataset[100])
print(dataset[122:125])

"""La clase *torch.utils.data.DataLoader* es la clase principal para cargar los datos. 
A esta clase se le pasa como argumento un objeto *Dataset*.

Para usarla tenemos que crear una instancia de la clase *DataLoader* a la que pasamos el objeto dataset que hemos creado. Definimos un tamaño de batch de 10 y shuffle=False para que no se cambie el orden de los datos en cada epoch (recorrido completo de los datos). 

"""

batch_size=10

train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
for i, (numbers, labels) in enumerate(train_loader):
  if  i<11:
    print('Batch number %d'%(i+1))
    print(numbers, labels)

"""# Creación modelo

## Importación de datos
"""

import os
import torch
import torchvision
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

"""(https://www.adictosaltrabajo.com/2019/06/04/google-colab-python-y-machine-learning-en-la-nube/) 
Para importar los datos de google drive, no tenemos que meter en el enlace que aparece debajo, tenemos que aceptar y copiar la conytraseña que nos sale

"""

from google.colab import drive
 
drive.mount('/content/drive')

"""Ya tenemos los datos importados, vemos las carpetas que hay con las fotos, cada carpeta se convertirá en un etiqueta 0=dibujos2D 1=dibujos3D y por último 2=personas"""

data_dir = './drive/MyDrive/imagenes'

print(os.listdir(data_dir))

dataSet = ImageFolder(data_dir, transform=ToTensor())
print(dataSet.classes)

img, label = dataSet[78]
print(img.shape, label)

"""Creamos dataLoader"""

from torch.utils.data.dataloader import DataLoader

batch_size=128

train_dl = DataLoader(dataSet, batch_size, shuffle=True, num_workers=4, pin_memory=True)
#val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

"""## Clase base
Empezamos a crear nuestro modelo
"""

import torch.nn as nn
import torch.nn.functional as F

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
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
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

"""## Modelo StarWars"""

class StarWarsCnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
        
    def forward(self, xb):
        return self.network(xb)

model = StarWarsCnnModel()

for images, labels in train_dl:
    print('images.shape:', images.shape)
    out = model(images)
    print('out.shape:', out.shape)
    print('out[0]:', out[0])
    break

"""## Entrenar usando GPU

### Creación funciones auxiliares
"""

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

"""### Entrenamiento

DEfinimos funciones para entrenar
"""

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
        #result = evaluate(model, val_loader)
        #result['train_loss'] = torch.stack(train_losses).mean().item()
        #model.epoch_end(epoch, result)
        #history.append(result)
    return train_losses

"""Movemos los datos a la memoria"""

device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device) # podemos poner shuffle = True
#val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device);

"""Comenzamos el entrenamiento"""

num_epochs = 2
lr = 0.001
opt_func = torch.optim.Adam

history = fit(num_epochs, lr, model, train_dl, train_dl, opt_func)
history