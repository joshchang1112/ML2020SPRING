import os
import cv2
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import random
import numpy as np
from cnn import VGG13, VGG16, VGG19
from mobileNet import StudentNet
from utils import readfile, set_seed
from dataset import ImgDataset
from train import training, deep_mutual_learning
import torch.nn.functional as F
import torchvision.models as models

# Set Random seed
SEED = 0
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
workspace_dir = sys.argv[1]

print("Reading data")
train_x, train_y, train_label_list = readfile(os.path.join(workspace_dir, "training"), True)
#print("Size of training data = {}".format(len(train_x)))
val_x, val_y, val_label_list = readfile(os.path.join(workspace_dir, "validation"), True)
#print("Size of validation data = {}".format(len(val_x)))

now_train_count = 0
now_val_count = 0
#print(train_label_list)
for i in range(11):
    if i == 0:
        train_X = np.concatenate((train_x[:int(train_label_list[i]*0.8), :, :, :], val_x[:val_label_list[i], :, :, :]))
        val_X = train_x[int(train_label_list[i]*0.8):train_label_list[i], :, :, :]
        train_Y = np.concatenate((train_y[:int(train_label_list[i]*0.8)], val_y[:val_label_list[i]]))
        val_Y = train_y[int(train_label_list[i]*0.8):train_label_list[i]]
    else:
        train_X = np.concatenate((train_X, train_x[now_train_count:now_train_count+int(train_label_list[i]*0.8), :, :, :]))
        train_X = np.concatenate((train_X, val_x[now_val_count:now_val_count+val_label_list[i], :, :, :]))
        val_X = np.concatenate((val_X, train_x[now_train_count+int(train_label_list[i]*0.8):now_train_count+train_label_list[i], :, :, :]))
        train_Y = np.concatenate((train_Y, train_y[now_train_count:now_train_count+int(train_label_list[i]*0.8)]))
        train_Y = np.concatenate((train_Y, val_y[now_val_count:now_val_count+val_label_list[i]]))
        val_Y = np.concatenate((val_Y, train_y[now_train_count+int(train_label_list[i]*0.8):now_train_count+train_label_list[i]]))
    now_train_count += train_label_list[i]
    now_val_count += val_label_list[i]

print("Size of training data = {}".format(len(train_X)))
print("Size of validation data = {}".format(len(val_X)))
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))

#training 時做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), #隨機將圖片水平翻轉
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomRotation(30), #隨機旋轉圖片
    transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)    
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Parameters initialize
batch_size = 32
train_set = ImgDataset(train_X, train_Y, train_transform)
val_set = ImgDataset(val_X, val_Y, test_transform)
test_set = ImgDataset(test_x, transform=train_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

teacher_net = []
teacher_net1 = VGG16().cuda()
teacher_net1.load_state_dict(torch.load('model/vgg16.model'))
teacher_net2 = VGG13().cuda()
teacher_net2.load_state_dict(torch.load('model/vgg13.model'))
# teacher_net3 = VGG19().cuda()
# teacher_net3.load_state_dict(torch.load('teacher_model/vgg19.model'))
teacher_net.append(teacher_net1)
teacher_net.append(teacher_net2)
# teacher_net.append(teacher_net3)
student_net = StudentNet(base=16).cuda()
#student_net = MobileNetV2(n_class=11).cuda()

print('Start Training')
training(teacher_net, student_net, train_loader, val_loader, test_loader, total_epoch=400)
models = []

# Deep Mutual Learning
# model1 = VGG16().cuda()
# smodel1.load_state_dict(torch.load('teacher_model/vgg16.model'))
# model2 = VGG13().cuda()
# model2.load_state_dict(torch.load('teacher_model/vgg13.model'))
# model3 = StudentNet(base=16).cuda()
# model3.load_state_dict(torch.load('student_model.bin'))

# # models.append(model1)
# models.append(model2)
# models.append(model3)
# deep_mutual_learning(models, train_loader, val_loader, total_epoch=100)
