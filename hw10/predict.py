import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import sys

test = np.load(sys.argv[1])
y = test

# model type: {'fcn', 'cnn', 'vae', 'cnn-vae'} 
model_type = 'fcn' 
if model_type == 'fcn' or model_type == 'vae':
    y = test.reshape(len(test), -1)
else:
    y = test

batch_size=256
data = torch.tensor(y, dtype=torch.float)
test_dataset = TensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

model = torch.load(sys.argv[2]).cuda()

model.eval()
reconstructed = list()
criterion = nn.MSELoss()
encode = []
losses = []

for i, data in enumerate(test_dataloader): 
    if model_type == 'cnn':
        img = data[0].transpose(3, 1).cuda()
    else:
        img = data[0].cuda()
    
    output = model(img)

    if model_type == 'cnn':
        output = output.transpose(3, 1)
    elif model_type == 'vae':
        output = output[0]
    
    reconstructed.append(output.cpu().detach().numpy())
    # reconstructed.append(dist)
    # reconstructed.append(y_reconstructed)
    # encode.append(output.cpu().detach().numpy())
    # loss = criterion(output, img)
    # losses.append(loss.item())

reconstructed = np.concatenate(reconstructed, axis=0)
anomality = np.sqrt(np.sum(np.square(reconstructed - y).reshape(len(y), -1), axis=1))
y_pred = anomality

# Report 1

# sort_idx = sorted(range(len(losses)), key=lambda k: losses[k])
# sort_losses = sorted(losses)
# print(len(losses))
# print(sort_losses[:2] + sort_losses[-2:])
# print(sort_idx[:2] + sort_idx[-2:])

# # Plot Figure
# plt.figure(figsize=(10,4))
# indexes = [8687, 9462, 7835, 3444]
# imgs = test[indexes,]
# #imgs = imgs.reshape(4, 32, 32, 3)
# for i, img in enumerate(imgs):
#     plt.subplot(2, 4, i+1, xticks=[], yticks=[])
#     plt.imshow(img)
    
# recs = reconstructed[indexes, ]
# recs = recs.reshape(4, 32, 32, 3)

# for i, img in enumerate(recs):
#     plt.subplot(2, 6, 6+i+1, xticks=[], yticks=[])
#     plt.imshow((img * 255).astype(np.uint8))

# plt.tight_layout()
# plt.savefig('report1_2.png')

with open(sys.argv[3], 'w') as f:
    f.write('id,anomaly\n')
    for i in range(len(y_pred)):
        f.write('{},{}\n'.format(i+1, y_pred[i]))
