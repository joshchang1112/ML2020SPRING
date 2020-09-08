import numpy as np
import torch
import torch.nn as nn
import random
import cv2
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import FeatureExtractor, LabelPredictor, DomainClassifier, CNN_VAE
from sklearn import manifold, datasets


def plot_scatter(feat, label, savefig=None):
    """ Plot Scatter Image.
    Args:
      feat: the (x, y) coordinate of clustering result, shape: (9000, 2)
      label: ground truth label of image (0/1), shape: (9000,)
    Returns:
      None
    """
    X = feat[:, 0]
    Y = feat[:, 1]
    plt.scatter(X, Y, c = label, s=20)
    plt.legend(loc='best')
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    return

source_transform = transforms.Compose([
    # 轉灰階: Canny 不吃 RGB。
    transforms.Grayscale(),
    # cv2 不吃 skimage.Image，因此轉成np.array後再做cv2.Canny
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    #transforms.Lambda(lambda x: cv2.bitwise_or(np.uint8(np.absolute(cv2.Sobel(np.array(x), cv2.CV_64F, 1, 0))), \
                      #np.uint8(np.absolute(cv2.Sobel(np.array(x), cv2.CV_64F, 0, 1))))),
    # 重新將np.array 轉回 skimage.Image
    transforms.ToPILImage(),
    # 水平翻轉 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    transforms.RandomRotation(15),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    # 轉灰階: 將輸入3維壓成1維。
    transforms.Grayscale(),
    # 縮放: 因為source data是32x32，我們將target data的28x28放大成32x32。
    transforms.Resize((32, 32)),
    # 水平翻轉 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    transforms.RandomRotation(15),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor = CNN_VAE().to(device)
#feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().to(device)
feature_extractor.load_state_dict(torch.load('extractor_model.bin'))
#label_predictor.load_state_dict(torch.load('predictor_model.bin'))

source_dataset = ImageFolder('real_or_drawing/train_data', transform=source_transform)
target_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)

source_dataloader = DataLoader(source_dataset, batch_size=128, shuffle=False)

label_predictor.eval()
feature_extractor.eval()
for i, ((source_data,source_label), (test_data, _)) in enumerate(zip(source_dataloader, test_dataloader)):
    test_data = test_data.to(device)
    source_data = source_data.to(device)

    test_feature = feature_extractor(test_data)
    source_feature = feature_extractor(source_data)
    if i == 0:
        latents = test_feature.cpu().detach().numpy()
        label = np.zeros(128, )
    else:
        latents = np.concatenate((latents, test_feature.cpu().detach().numpy()), axis = 0)
        label = np.concatenate((label, np.zeros(128, )), axis = 0)

    latents = np.concatenate((latents, source_feature.cpu().detach().numpy()), axis = 0)
    label = np.concatenate((label, np.ones(128, )), axis = 0)
    

print(latents.shape)
X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1, perplexity=40).fit_transform(latents)
print(X_tsne.shape)
label = label[:10120]
plot_scatter(X_tsne, label, savefig='report.png')
