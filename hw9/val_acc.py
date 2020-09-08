import sys
import glob
import torch
import torch.nn as nn
from autoencoder import Improved_AE
from utils import same_seeds, count_parameters, cal_acc
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
from dataset import Image_Dataset
from preprocess import preprocess
from test import inference, predict
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#checkpoints_list = glob.glob('checkpoints/checkpoint_*.pth')
checkpoints_list = ['checkpoints/checkpoint_'] * 30
epoch = []
for i in range(30):
    checkpoints_list[i] += str(i+1)+'0.pth'
    epoch.append((i+1) * 10)

print("Reading Data")
trainX = np.load(sys.argv[1])
trainX_preprocessed = preprocess(trainX)

valX = np.load('data/valX.npy')
valY = np.load('data/valY.npy')
# load data
'''
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)    
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
'''
dataset = Image_Dataset(trainX_preprocessed)
#dataset = Image_Dataset(trainX, test_transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
model = Improved_AE().cuda()

points = []
with torch.no_grad():
    for i, checkpoint in enumerate(checkpoints_list):
        print('[{}/{}] {}'.format(i+1, len(checkpoints_list), checkpoint))
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        err = 0
        n = 0
        for x in dataloader:
            x = x.cuda()
            _, rec = model(x)
            err += torch.nn.MSELoss(reduction='sum')(x, rec).item()
            n += x.flatten().size(0)
        print('Reconstruction error (MSE):', err/n)
        latents = inference(X=valX, model=model)
        pred, X_embedded = predict(latents)
        acc = cal_acc(valY, pred)
        print('Accuracy:', acc)
        points.append((err/n, acc))

'''
ps = list(zip(*points))
plt.figure(figsize=(6,6))
plt.subplot(211, title='Reconstruction error (MSE)').plot(epoch, ps[0])
plt.subplot(212, title='Accuracy (val)', xlabel='Epoch').plot(epoch, ps[1])
plt.show()
'''
