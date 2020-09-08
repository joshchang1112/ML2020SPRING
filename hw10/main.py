import numpy as np
from model import *
import torch
import torch.nn as nn
import random
import sys

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from utils import set_seed, preprocess, loss_vae

set_seed(208)
train = np.load(sys.argv[1], allow_pickle=True)

x = train
num_epochs = 4
batch_size = 256
learning_rate = 5e-5

# model type: {'fcn', 'cnn', 'vae', 'cnn-vae'} 
model_type = 'fcn' 

if model_type == 'fcn' or model_type == 'vae':
    x = x.reshape(len(x), -1)

data = torch.tensor(x, dtype=torch.float)
train_dataset = TensorDataset(data)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

model_classes = {'fcn':fcn_autoencoder(), 'cnn':conv_autoencoder(), 'vae':VAE()}
model = model_classes[model_type].cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=1e-7)

print("Start Training")
best_loss = np.inf
model.train()

for epoch in range(num_epochs):
    avg_loss = 0
    for data in train_dataloader:
        # data process
        if model_type == 'cnn':
            img = data[0].transpose(3, 1).cuda()
        else:
            img = data[0].cuda()
        # ===================forward=====================
        output = model(img)
        if model_type == 'vae':
            loss = loss_vae(output[0], img, output[1], output[2], criterion)
        else:
            loss = criterion(output, img)
        avg_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_loss /= len(train_dataloader)
    scheduler.step(avg_loss)
    # ===================save====================
    if avg_loss < best_loss:
        best_loss = avg_loss
        print('Best loss:{:.6f}'.format(avg_loss))
        torch.save(model, sys.argv[2])
    # ===================log========================
    print('epoch [{}/{}], loss:{:.6f}'
            .format(epoch + 1, num_epochs, avg_loss))

