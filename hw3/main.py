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
from cnn import Classifier


def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 256, 256, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    label_count = [0] * 11
    pre = 0
    add = 0
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(256, 256))
        if label:
          y[i] = int(file.split("_")[0])
          if pre != y[i]:
              add += 1
          label_count[add] += 1
          pre = y[i]
    if label:
      return x, y, label_count
    else:
      return x

#分別將 training set、validation set、testing set 用 readfile 函式讀進來
SEED = 0
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
workspace_dir = './food-11'
print("Reading data")

#train_x, train_y, train_label_list = readfile(os.path.join(workspace_dir, "training"), True)
#print("Size of training data = {}".format(len(train_x)))
val_x, val_y, val_label_list = readfile(os.path.join(workspace_dir, "validation"), True)
'''
now_train_count = 0
now_val_count = 0
#print(train_label_list)
for i in range(11):
    if i == 0:
        train_X = np.concatenate((train_x[:int(train_label_list[i]*0.67), :, :, :], val_x[:val_label_list[i], :, :, :]))
        val_X = train_x[int(train_label_list[i]*0.67):train_label_list[i], :, :, :]
        train_Y = np.concatenate((train_y[:int(train_label_list[i]*0.67)], val_y[:val_label_list[i]]))
        val_Y = train_y[int(train_label_list[i]*0.67):train_label_list[i]]
    else:
        train_X = np.concatenate((train_X, train_x[now_train_count:now_train_count+int(train_label_list[i]*0.67), :, :, :]))
        train_X = np.concatenate((train_X, val_x[now_val_count:now_val_count+val_label_list[i], :, :, :]))
        val_X = np.concatenate((val_X, train_x[now_train_count+int(train_label_list[i]*0.67):now_train_count+train_label_list[i], :, :, :]))
        train_Y = np.concatenate((train_Y, train_y[now_train_count:now_train_count+int(train_label_list[i]*0.67)]))
        train_Y = np.concatenate((train_Y, val_y[now_val_count:now_val_count+val_label_list[i]]))
        val_Y = np.concatenate((val_Y, train_y[now_train_count+int(train_label_list[i]*0.67):now_train_count+train_label_list[i]]))

    now_train_count += train_label_list[i]
    now_val_count += val_label_list[i]
'''
#print("Size of training data = {}".format(len(train_X)))
#print("Size of validation data = {}".format(len(val_x)))

#test_x = readfile(os.path.join(workspace_dir, "testing"), False)
#print("Size of Testing data = {}".format(len(test_x)))

#training 時做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), #隨機將圖片水平翻轉
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomRotation(30), #隨機旋轉圖片
    transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)    
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

batch_size = 32
#train_set = ImgDataset(train_x, train_y, train_transform)
#val_set = ImgDataset(val_x, val_y, test_transform)
#train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

model = Classifier().to(device)
loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=7e-5) # optimizer 使用 Adam
num_epoch = 150
total_para = sum(p.numel() for p in model.parameters())
print('Parameter total:{}'.format(total_para))
best_acc = 0.0
'''
print("Start Training!")

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    if epoch == 100:
        optimizer = torch.optim.SGD(model.parameters(), lr=5e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 12800, eta_min=1e-7)
    model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].to(device)) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].to(device)) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
        optimizer.step() # 以 optimizer 用 gradient 更新參數值
        if epoch >= 100:
            scheduler.step()
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].to(device))
            batch_loss = loss(val_pred, data[1].to(device))

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()
        if val_acc > best_acc:
            torch.save(model, 'ckpt.model')
            print('saving model with acc {:.3f}'.format(val_acc/val_set.__len__()*100))
            best_acc = val_acc
        #將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))
'''
test_set = ImgDataset(val_x, val_y, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = torch.load('report_1/ckpt_1.model',  map_location='cuda:0')
model.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model(data[0].to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

plt.figure()
confmat = confusion_matrix(val_y, prediction, normalize='true')
fig, ax = plt.subplots(fig_size=(15, 15))
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=round(confmat[i, j], 4), va='center', ha='center')
plt.xlabel='Predicted Label'
plt.ylabel = 'True Label'
plt.show()
plt.savefig('Confusion.png')
#將結果寫入 csv 檔
'''
with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))
'''
