import os
import torch
import argparse
import numpy as np
from torch import nn
from gensim.models import word2vec
from preprocess import Preprocess
from utils import load_training_data, load_testing_data, pad_to_len, collate_fn
from rnn import LSTM_Net
from data import TwitterDataset
from train import training
from test import testing
import random
import pandas as pd
import sys

# Setting random seed
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

# 通過torch.cuda.is_available()的回傳值進行判斷是否有使用GPU的環境，如果有的話device就設為"cuda"，沒有的話就設為"cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# 處理好各個data的路徑
train_with_label = sys.argv[1]
train_no_label = sys.argv[2]
#testing_data = os.path.join(path_prefix, 'testing_data.txt')

model_dir = 'model/'
w2v_path = os.path.join(model_dir, 'w2v_all.model') # 處理word to vec model的路徑

# 定義句子長度、要不要固定embedding、batch大小、要訓練幾個epoch、learning rate的值、model的資料夾路徑
sen_len = 20
fix_embedding = True # fix embedding during training
batch_size = 32
epoch = 5
lr = 1e-3

print("loading data ...") # 把'training_label.txt'跟'training_nolabel.txt'讀進來
train_x, y = load_training_data(train_with_label)
#train_x_no_label = load_training_data(train_no_label)
#train_x_no_label = train_x_no_label[:160000]
# 對input跟labels做預處理
#preprocess = Preprocess(train_x, sen_len, train_x_no_label, w2v_path=w2v_path)
preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
#print(preprocess.embedding.most_similar("love"))
train_x = preprocess.sentence_word2idx()
#train_x, train_x_no_label = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)
# 製作一個model的對象3model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.1, fix_embedding=fix_embedding)
model = LSTM_Net(embedding, embedding_dim=256, hidden_dim=128, num_layers=3, dropout=0.2, fix_embedding=fix_embedding)
model = model.to(device)

# 把data分為training data跟validation data(將一部份training data拿去當作validation data)
#X_train, X_val, y_train, y_val = train_x[40000:], train_x[:40000], y[40000:], y[:40000]

X_train, X_val, y_train, y_val = train_x[:160000] , train_x[160000:], y[:160000], y[160000:]

# 把data做成dataset供dataloader取用
#train_dataset = TwitterDataset(X=X_train, y=y_train, unlabel_x = train_x_no_label)
train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset = TwitterDataset(X=X_val, y=y_val)
#train_no_label_dataset = TwitterDataset(X=train_no_label_x, y=None)
# 把data 轉成 batch of tensors
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8, 
                                            collate_fn=collate_fn)
'''
train_no_label_loader = torch.utils.data.DataLoader(dataset = train_no_label_dataset,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8,
                                            collate_fn=collate_fn)
'''
val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8,
                                            collate_fn=collate_fn)

# 開始訓練
training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)

# Self-training
'''
for i in range(2):
    print("loading testing data ...")
    test_x_word = load_testing_data(train_no_label)
    preprocess = Preprocess(test_x_word, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    test_x = preprocess.sentence_word2idx()
    test_dataset = TwitterDataset(X=test_x, y=None)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8,
                                            collate_fn=collate_fn)
    print('\nload model ...')
    model = torch.load(os.path.join(model_dir, 'ckpt.model'))
    outputs = testing(batch_size, test_loader, model, device)
    pesudo_label_index = []
    for i in range(len(outputs)):
        if outputs[i] > 0.9 or outputs[i] < 0.1:
            pesudo_label_index.append(i)

    pesudo_x = [test_x_word[i] for i in pesudo_label_index]
    pesudo_y = [outputs[i] for i in pesudo_label_index]
    for i in range(len(pesudo_y)):
        if pesudo_y[i] < 0.1:
            pesudo_y[i] = 0.0
        else:
            pesudo_y[i] = 1.0

    # 對input跟labels做預處理
    print("loading data ...") # 把'training_label.txt'跟'training_nolabel.txt'讀進來
    train_x, y = load_training_data(train_with_label)
    train_x = train_x + pesudo_x
    y = y + pesudo_y

    preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x = preprocess.sentence_word2idx()
    y = preprocess.labels_to_tensor(y)

    # 製作一個model的對象
    new_model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.2, fix_embedding=fix_embedding)
    new_model = new_model.to(device) # device為"cuda"，model使用GPU來訓練(餵進去的inputs也需要是cuda tensor)

    # 把data分為training data跟validation data(將一部份training data拿去當作validation data)
    X_train, X_val, y_train, y_val = train_x[:160000]+train_x[200000:], train_x[160000:200000], torch.cat([y[:160000], y[200000:]]), y[160000:200000]

    # 把data做成dataset供dataloader取用
    train_dataset = TwitterDataset(X=X_train, y=y_train)
    val_dataset = TwitterDataset(X=X_val, y=y_val)

    batch_size=32
    # 把data 轉成 batch of tensors
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8,
                                            collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8,
                                            collate_fn=collate_fn)

    # 定義句子長度、要不要固定embedding、batch大小、要訓練幾個epoch、learning rate的值、model的資料夾路徑
    epoch = 5
    lr = 1e-3
    # 開始訓練
    training(batch_size, epoch, lr, model_dir, train_loader, val_loader, new_model, device)
'''

# Predict test data
'''
test_x = load_testing_data(testing_data)
preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()
test_dataset = TwitterDataset(X=test_x, y=None)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8,
                                            collate_fn=collate_fn)
print('\nload model ...')
model = LSTM_Net(embedding, embedding_dim=256, hidden_dim=128, num_layers=3, dropout=0.2, fix_embedding=fix_embedding).to(device)
model.load_state_dict(torch.load(os.path.join(model_dir, 'ckpt.model')))
outputs = testing(batch_size, test_loader, model, device, step=1)
# 寫到csv檔案供上傳kaggle
tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
print("save csv ...")
tmp.to_csv(os.path.join(path_prefix, 'predict.csv'), index=False)
print("Finish Predicting")
'''
