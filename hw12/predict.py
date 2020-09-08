import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import random
import cv2
import os
import sys

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import FeatureExtractor, LabelPredictor, DomainClassifier, CNN_VAE

target_transform = transforms.Compose([
    # 轉灰階: 將輸入3維壓成1維。
    transforms.Grayscale(),
    # 縮放: 因為source data是32x32，我們將target data的28x28放大成32x32。
    transforms.Resize((32, 32)),
    # 水平翻轉 (Augmentation)
    #transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    #transforms.RandomRotation(15),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor = CNN_VAE().to(device)
#feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().to(device)
feature_extractor.load_state_dict(torch.load('extractor_model.bin'))
label_predictor.load_state_dict(torch.load('predictor_model.bin'))

target_dataset = ImageFolder(os.path.join(sys.argv[1], 'test_data'), transform=target_transform)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)

result = []
label_predictor.eval()
feature_extractor.eval()
for i, (test_data, _) in enumerate(test_dataloader):
    test_data = test_data.to(device)

    class_logits = label_predictor(feature_extractor(test_data))

    x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    result.append(x)

import pandas as pd
result = np.concatenate(result)

# Generate your submission
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
df.to_csv(sys.argv[2],index=False)


