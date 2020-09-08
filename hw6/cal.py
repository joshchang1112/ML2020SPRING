import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable

if __name__ == '__main__':
    linf_total = 0
    fnames = []
    for i in range(200):
        fnames.append("{:03d}".format(i))
    for i in range(200):
        raw_image = np.array(Image.open(os.path.join('./data/images', fnames[i] + '.png')))
        adv_image = np.array(Image.open(os.path.join('./output', fnames[i] + '.png')))
        linf_total += np.linalg.norm((raw_image.astype('int64') - adv_image.astype('int64')).flatten(), np.inf)
    print('Avg linf:{}'.format(linf_total/200))
