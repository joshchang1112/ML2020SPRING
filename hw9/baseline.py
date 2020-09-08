import torch
import torch.nn as nn
from autoencoder import Base_AE
from utils import same_seeds, count_parameters, cal_acc
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
from dataset import Image_Dataset
from preprocess import preprocess
#from test import inference
import torchvision.transforms as transforms

same_seeds(0)

model = Base_AE().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=1e-7) 

model.train()
n_epoch = 100

print("Reading Data")
trainX = np.load(sys.argv[1])
trainX_preprocessed = preprocess(trainX)
'''
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), #隨機將圖片水平翻轉
    transforms.RandomRotation(15), #隨機旋轉圖片
    transforms.ToTensor() #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)    
])
'''
train_dataset = Image_Dataset(trainX_preprocessed)
# 準備 dataloader, model, loss criterion 和 optimizer
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
'''
valX = np.load(sys.argv[2])
valX_preprocessed = preprocess(valX)

val_dataset = Image_Dataset(valX_preprocessed)
# 準備 dataloader, model, loss criterion 和 optimizer
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
'''

epoch_loss = 0
print("Start Training")
save_path = sys.argv[2]
# 主要的訓練過程
best_val_loss = 100
for epoch in range(n_epoch):
    epoch_loss = 0
    model.train()
    for data in train_dataloader:
        img = data
        img = img.cuda()

        output1, output = model(img)
        loss = criterion(output, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    '''
    model.eval()
    avg_loss = 0
    for data in val_dataloader:
        img = data
        img = img.cuda()
        with torch.no_grad():
            output1, output = model(img)
        
        loss = criterion(output, img)
        avg_loss += loss.item()
    
    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        torch.save(model.state_dict(), './checkpoints/best_val_loss.pth'.format(epoch+1))
    '''
    print('epoch [{}/{}], train loss:{:.5f}'.format(epoch+1, n_epoch, epoch_loss))

torch.save(model.state_dict(), save_path)
