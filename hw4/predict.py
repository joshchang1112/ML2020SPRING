import os
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from preprocess import Preprocess
from utils import load_training_data, load_testing_data, pad_to_len, collate_fn
from rnn import LSTM_Net, BILSTM_Net
from data import TwitterDataset
from train import training
from test import testing
import pandas as pd
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("loading testing data ...")

model_dir = 'model/'
w2v_path = os.path.join(model_dir, 'w2v_all.model') # 處理word to vec model的路徑
sen_len = 20
batch_size = 32
testing_data = sys.argv[1]
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


print('\nload LSTM Model ...')
#model = torch.load(os.path.join(model_dir, 'ckpt.model'))
model = LSTM_Net(embedding, embedding_dim=256, hidden_dim=128, num_layers=3, dropout=0.2, fix_embedding=True)
model = model.to(device) # device為"cuda"，model使用GPU來訓練(餵進去的inputs也需要是cuda tensor)
model.load_state_dict(torch.load(os.path.join(model_dir, 'ckpt_1.model'), map_location={'cuda:3':'cuda'}))
output1 = testing(batch_size, test_loader, model, device, step=1)

print('load BI-LSTM-1 Model ...')
#model = torch.load(os.path.join(model_dir, 'ckpt.model'))
model = BILSTM_Net(embedding, embedding_dim=256, hidden_dim=128, num_layers=3, dropout=0.2, fix_embedding=True)
model = model.to(device) # device為"cuda"，model使用GPU來訓練(餵進去的inputs也需要是cuda tensor)
model.load_state_dict(torch.load(os.path.join(model_dir, 'ckpt_2.model'), map_location={'cuda:3':'cuda'}))
output2_1 = testing(batch_size, test_loader, model, device, step=1)

print('load BI-LSTM-2 Model ...')
#model = torch.load(os.path.join(model_dir, 'ckpt.model'))
model = BILSTM_Net(embedding, embedding_dim=256, hidden_dim=128, num_layers=3, dropout=0.2, fix_embedding=True)
model = model.to(device) # device為"cuda"，model使用GPU來訓練(餵進去的inputs也需要是cuda tensor)
model.load_state_dict(torch.load(os.path.join(model_dir, 'ckpt_2-2.model'), map_location={'cuda:3':'cuda'}))
output2_2 = testing(batch_size, test_loader, model, device, step=1)

print('load BI-LSTM-3 Model ...')
#model = torch.load(os.path.join(model_dir, 'ckpt.model'))
model = BILSTM_Net(embedding, embedding_dim=256, hidden_dim=128, num_layers=3, dropout=0.2, fix_embedding=True)
model = model.to(device) # device為"cuda"，model使用GPU來訓練(餵進去的inputs也需要是cuda tensor)
model.load_state_dict(torch.load(os.path.join(model_dir, 'ckpt_2-3.model'), map_location={'cuda:3':'cuda'}))
output2_3 = testing(batch_size, test_loader, model, device, step=1)

print('load BI-LSTM-4 Model ...')
#model = torch.load(os.path.join(model_dir, 'ckpt.model'))
model = BILSTM_Net(embedding, embedding_dim=256, hidden_dim=128, num_layers=3, dropout=0.2, fix_embedding=True)
model = model.to(device) # device為"cuda"，model使用GPU來訓練(餵進去的inputs也需要是cuda tensor)
model.load_state_dict(torch.load(os.path.join(model_dir, 'ckpt_2-4.model'), map_location={'cuda:3':'cuda'}))
output2_4 = testing(batch_size, test_loader, model, device, step=1)

print('load BI-LSTM-5 Model ...')
#model = torch.load(os.path.join(model_dir, 'ckpt.model'))
model = BILSTM_Net(embedding, embedding_dim=256, hidden_dim=128, num_layers=3, dropout=0.2, fix_embedding=True)
model = model.to(device) # device為"cuda"，model使用GPU來訓練(餵進去的inputs也需要是cuda tensor)
model.load_state_dict(torch.load(os.path.join(model_dir, 'ckpt_2-5.model'), map_location={'cuda:3':'cuda'}))
output2_5 = testing(batch_size, test_loader, model, device, step=1)

print('Load LSTM with semi-supervised Model ...')
#model = torch.load(os.path.join(model_dir, 'ckpt.model'))
model = LSTM_Net(embedding, embedding_dim=256, hidden_dim=128, num_layers=3, dropout=0.2, fix_embedding=True)
model = model.to(device) # device為"cuda"，model使用GPU來訓練(餵進去的inputs也需要是cuda tensor)
model.load_state_dict(torch.load(os.path.join(model_dir, 'ckpt_3.model'), map_location={'cuda:3':'cuda'}))
output3 = testing(batch_size, test_loader, model, device, step=1)

print('load BILSTM with semi-supervised Model ...')
#model = torch.load(os.path.join(model_dir, 'ckpt.model'))
model = BILSTM_Net(embedding, embedding_dim=256, hidden_dim=128, num_layers=3, dropout=0.2, fix_embedding=True)
model = model.to(device) # device為"cuda"，model使用GPU來訓練(餵進去的inputs也需要是cuda tensor)
model.load_state_dict(torch.load(os.path.join(model_dir, 'ckpt_4.model'), map_location={'cuda:3':'cuda'}))
output4 = testing(batch_size, test_loader, model, device, step=1)

ensemble = []
for i in range(len(test_x)):
    ans = (output1[i] + 0.8* output2_1[i] + 0.8*output2_2[i] + 0.8*output2_3[i] + 0.8*output2_4[i] + 0.8*output2_5[i] + output3[i] + output4[i]) / 7
    if ans >= 0.5:
        ans = 1
    else:
        ans = 0
    ensemble.append(ans)

# 寫到csv檔案供上傳kaggle
tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":ensemble})


print("save csv ...")
tmp.to_csv(sys.argv[2], index=False)
print("Finish Predicting")
