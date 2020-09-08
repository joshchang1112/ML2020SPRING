import numpy as np
import torch
from test import inference, predict
from utils import cal_acc, plot_scatter
from autoencoder import Base_AE, Improved_AE
from preprocess import preprocess

valX = np.load('data/valX.npy')
valY = np.load('data/valY.npy')

#model = Base_AE().cuda()
model = Improved_AE().cuda()
model.load_state_dict(torch.load('./improve-2.pth'))

model.eval()
latents = inference(valX, model)
pred_from_latent, emb_from_latent = predict(latents)
acc_latent = cal_acc(valY, pred_from_latent)
print('The clustering accuracy is:', acc_latent)
print('The clustering result:')
plot_scatter(emb_from_latent, valY, savefig='p1_improved.png')
'''

import matplotlib.pyplot as plt
import numpy as np

# 畫出原圖
trainX = np.load('data/trainX_new.npy')
trainX_preprocessed = preprocess(trainX)
model = Improved_AE().cuda()
model.load_state_dict(torch.load('./improve-2.pth'))

plt.figure(figsize=(10,4))
indexes = [1,2,3,6,7,9]
imgs = trainX[indexes,]
for i, img in enumerate(imgs):
    plt.subplot(2, 6, i+1, xticks=[], yticks=[])
    plt.imshow(img)

# 畫出 reconstruct 的圖
inp = torch.Tensor(trainX_preprocessed[indexes,]).cuda()
latents, recs = model(inp)
recs = ((recs+1)/2 ).cpu().detach().numpy()
recs = recs.transpose(0, 2, 3, 1)
for i, img in enumerate(recs):
    plt.subplot(2, 6, 6+i+1, xticks=[], yticks=[])
    plt.imshow(img)
  
plt.show()'''
