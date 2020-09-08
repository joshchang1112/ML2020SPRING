import sys
import random
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
from torchvision.utils import save_image
device = torch.device("cuda")

# 實作一個繼承 torch.utils.data.Dataset 的 Class 來讀取圖片
class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        # 圖片所在的資料夾
        self.root = root
        # 由 main function 傳入的 label
        self.label = torch.from_numpy(label).long()
        # 由 Attacker 傳入的 transforms 將輸入的圖片轉換成符合預訓練模型的形式
        self.transforms = transforms
        # 圖片檔案名稱的 list
        self.fnames = []

        for i in range(200):
            self.fnames.append("{:03d}".format(i))

    def __getitem__(self, idx):
        # 利用路徑讀取圖片
        img = Image.open(os.path.join(self.root, self.fnames[idx] + '.png'))
        # 將輸入的圖片轉換成符合預訓練模型的形式
        img = self.transforms(img)
        # 圖片相對應的 label
        label = self.label[idx]
        return img, label

    def __len__(self):
        # 由於已知這次的資料總共有 200 張圖片 所以回傳 200
        return 200

class Attacker:
    def __init__(self, img_dir, label, output_dir):
        # 讀入預訓練模型 vgg16
        self.model = models.densenet121(pretrained = True)
        self.model.to(device)
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # 把圖片 normalize 到 0~1 之間 mean 0 variance 1
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([
                        transforms.Resize((224, 224), interpolation=3),
                        transforms.ToTensor(),
                        self.normalize
                    ])
        # 利用 Adverdataset 這個 class 讀取資料
        self.output_dir = output_dir
        self.dataset = Adverdataset(img_dir, label, transform)
        #self.dataset = Adverdataset('output/', label, transform)
        self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size = 1,
                shuffle = False)

    # FGSM 攻擊
    def fgsm_attack(self, image, epsilon, data_grad):
        # 找出 gradient 的方向
        sign_data_grad = data_grad.sign()
        # 將圖片加上 gradient 方向乘上 epsilon 的 noise
        perturbed_image = image + epsilon * sign_data_grad
        return perturbed_image

    def attack(self, epsilon):
        # 存下一些成功攻擊後的圖片 以便之後顯示
        adv_examples = []
        wrong, fail, success, linf = 0, 0, 0, 0
        for i, (data, target) in enumerate(self.loader):
            print(i)
            data, target = data.to(device), target.to(device)
            data_raw = data;
            data.requires_grad = True
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]
            # 如果 class 錯誤 就不進行攻擊
            if init_pred.item() != target.item():
                wrong += 1
                print('WRONG')
                final_pred = output.max(1, keepdim=True)[1]
                inv_normalize = transforms.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                    std=[1/0.229, 1/0.224, 1/0.225]
                )
                inv_tensor = inv_normalize(data.squeeze(0))
                save_image(inv_tensor, self.output_dir+ '/{:0>3d}.png'.format(i))

                continue


            # 如果 class 正確 就開始計算 gradient 進行 FGSM 攻擊
            loss = F.nll_loss(output, target)
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            data_adv = self.fgsm_attack(data, epsilon, data_grad)
            # 再將加入 noise 的圖片丟入 model 進行測試 得出相對應的 class        
            output = self.model(data_adv)
            
            inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
            inv_tensor = inv_normalize(data_adv.squeeze(0))
            save_image(inv_tensor, self.output_dir + '/{:0>3d}.png'.format(i))
            final_pred = output.max(1, keepdim=True)[1]

            if final_pred.item() == target.item():
                # 辨識結果還是正確 攻擊失敗
                fail += 1
                print('FAIL')
            else:
                # 辨識結果失敗 攻擊成功
                success += 1
                print('SUCCESS')
            
            linf += np.linalg.norm((data.detach().cpu().numpy().astype('int64') - data_adv.detach().cpu().numpy().astype('int64')).flatten(), np.inf)
            #print(linf / (i+1))
            if len(adv_examples) < 5:
                adv_ex = data_adv * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                adv_ex = adv_ex.squeeze().detach().cpu().numpy()
                data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                data_raw = data_raw.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), data_raw , adv_ex) )
        final_acc = (fail / (wrong + success + fail))
        avg_linf = linf / 200
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}, L_INF = {}\n".format(epsilon, fail, len(self.loader), final_acc, avg_linf))
        return adv_examples, final_acc

if __name__ == '__main__':
    # Set seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
     
    # 讀入圖片相對應的 label
    df = pd.read_csv(sys.argv[1] + "/labels.csv")
    df = df.loc[:, 'TrueLabel'].to_numpy()
    label_name = pd.read_csv(sys.argv[1] + "/categories.csv")
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()
    # new 一個 Attacker class
    attacker = Attacker(sys.argv[1] + '/images', df, sys.argv[2])
    # 要嘗試的 epsilon
    epsilons = [0.1]

    accuracies, examples = [], []

    # 進行攻擊 並存起正確率和攻擊成功的圖片
    for eps in epsilons:
        ex, acc = attacker.attack(eps)
        accuracies.append(acc)
        examples.append(ex)
    
