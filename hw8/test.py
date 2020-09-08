import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.sampler as sampler
import torchvision
from torchvision import datasets, transforms

import numpy as np
import sys
import os
import random
import json
import matplotlib.pyplot as plt

from config import configurations
from utils import infinite_iter, build_model, save_model
from train import train
from dataset import EN2CNDataset
from predict import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = configurations()
print ('config:\n', vars(config))

# Initialize dataset
train_dataset = EN2CNDataset(sys.argv[1], config.max_output_len, 'training')
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
train_iter = infinite_iter(train_loader)

val_dataset = EN2CNDataset(sys.argv[1], config.max_output_len, 'validation')
val_loader = DataLoader(val_dataset, batch_size=1)

test_dataset = EN2CNDataset(sys.argv[1], config.max_output_len, 'testing')
test_loader = DataLoader(test_dataset, batch_size=1)

# Build training object  
model, optimizer = build_model(config, train_dataset.en_vocab_size, train_dataset.cn_vocab_size)
loss_function = nn.CrossEntropyLoss(ignore_index=0)

# Start Testing
_, _, result = test(model, test_loader, loss_function, config.beam_size)

with open(sys.argv[2], 'w') as f:
    for i in range(len(result)):
        for j in range(len(result[i][1])):
            f.write(result[i][1][j])
        f.write('\n')


'''
# Start Training
train_losses, val_losses, bleu_scores = [], [], []
total_steps = 0
while (total_steps < config.num_steps):
    # 訓練模型
    model, optimizer, loss = train(model, optimizer, train_iter, loss_function, total_steps, config.summary_steps, train_dataset)
    train_losses += loss
    # 檢驗模型
    val_loss, bleu_score, result = test(model, val_loader, loss_function, config.beam_size)
    val_losses.append(val_loss)
    bleu_scores.append(bleu_score)

    total_steps += config.summary_steps
    print ("\r", "val [{}] loss: {:.3f}, Perplexity: {:.3f}, blue score: {:.3f}       ".format(total_steps, val_loss, np.exp(val_loss), bleu_score))
    
    # 儲存模型和結果
    if total_steps % config.store_steps == 0 or total_steps >= config.num_steps:
        save_model(model, optimizer, config.store_model_path, total_steps)
        with open(f'{config.store_model_path}/final/output_{total_steps}.txt', 'w') as f:
            for line in result:
                print (line, file=f)

x1 = [(i+1)*5 for i in range(4800)]
x2 = [(i+1)*600 for i in range(40)]
plt.figure()
plt.plot(x1, train_losses, label='train')
plt.plot(x2, val_losses, label='valid')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.xlim(0, 24000)
plt.title('Train & Validation loss')
plt.savefig('loss_dot_inv_final')

plt.figure()
plt.plot(x2, bleu_scores)
plt.xlabel('iterations')
plt.ylabel('BLEU score')
plt.title('BLEU score')
plt.savefig('BLEU_score_dot_inv_final')
'''

