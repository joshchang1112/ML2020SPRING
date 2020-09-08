import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import random
import numpy as np
from mobileNet import StudentNet
import sys
from utils import readfile, set_seed
from dataset import ImgDataset
from encode import decode8, decode16
import pickle


# Set Random seed
SEED = 0
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
workspace_dir = sys.argv[1]
batch_size = 32

test_transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Reading data")
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

print("Start predicting")
model = StudentNet().cuda()
#model.load_state_dict(torch.load('student_model.bin'))
model.load_state_dict(decode8('8_bit_model.pkl'))
model.eval()
pred_1 = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            pred_1.append(y)

with open(sys.argv[2], 'w') as f:
    f.write('Id,label\n')
    for i, y in  enumerate(pred_1):
        f.write('{},{}\n'.format(i, y))
