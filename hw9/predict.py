import sys
import numpy as np
import torch
from autoencoder import Base_AE, Improved_AE
from utils import same_seeds, count_parameters, cal_acc
from torch.utils.data import DataLoader, Dataset
from dataset import Image_Dataset
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans, KMeans
from preprocess import preprocess
import torchvision.transforms as transforms
from test import inference, predict, save_prediction, invert

#load model
model = Improved_AE().cuda()
model.load_state_dict(torch.load(sys.argv[2]))
model.eval()

# 準備 data
trainX = np.load(sys.argv[1])

# 預測答案
latents = inference(X=trainX, model=model)
pred, X_embedded = predict(latents)

# 將預測結果存檔，上傳 kaggle
if pred[6] == 1:
    save_prediction(pred, sys.argv[3])

# 由於是 unsupervised 的二分類問題，我們只在乎有沒有成功將圖片分成兩群
# 如果上面的檔案上傳 kaggle 後正確率不足 0.5，只要將 label 反過來就行了
else:
    save_prediction(invert(pred), sys.argv[3])

