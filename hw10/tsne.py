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
from sklearn import manifold

def plot_scatter(feat, label=None, savefig=None):
    """ Plot Scatter Image.
    Args:
      feat: the (x, y) coordinate of clustering result, shape: (9000, 2)
      label: ground truth label of image (0/1), shape: (9000,)
    Returns:
      None
    """
    X = feat[:, 0]
    Y = feat[:, 1]
    plt.scatter(X, Y,  s=20)
    plt.legend(loc='best')
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    return

test = np.load('data/test.npy')
y = test
print(test.shape)

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

model = torch.load('models/best.pth').cuda()

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
    
    # output = model.encoder(img)


    # if model_type == 'cnn':
    #     output = output.transpose(3, 1)
    # elif model_type == 'vae':
    #     output = output[0]
    output = img

    if i == 0:
        latents = output.cpu().detach().numpy()
    else:
        latents = np.concatenate((latents, output.cpu().detach().numpy()), axis = 0)

X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1, perplexity=40).fit_transform(latents)
print(X_tsne.shape)
#label = label[:10120]
plot_scatter(X_tsne,  savefig='report.png')
