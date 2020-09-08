import math
import torch
from torch import nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class BILSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(BILSTM_Net, self).__init__()
        # 製作 embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否將 embedding fix住，如果fix_embedding為False，在訓練過程中，embedding也會跟著被訓練
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_dim*2, 1),
                                         nn.Sigmoid() )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
  
    def forward(self, inputs, context_lens):
        #inputs = self.embedding(inputs)
        context_lens = torch.tensor(context_lens).to(self.device)
        input_lens, idx = context_lens.sort(0, descending=True)
        _, un_idx = torch.sort(idx)
        inputs = inputs[idx]
        inputs = self.embedding(inputs)
        inputs_packed = pack_padded_sequence(inputs, input_lens, batch_first=True)
        x, _ = self.lstm(inputs_packed, None)
        output, _ = pad_packed_sequence(x, batch_first=True)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最後一層的 hidden state
        
        '''
        for i in range(x.size()[0]):
            if context_lens[i] > 31:
                context_lens[i] = 31
            if i == 0:
                output = x[i, context_lens[i]-1, :].unsqueeze(0)
            else:
                output = torch.cat([output, x[i, context_lens[i]-1, :].unsqueeze(0)], dim=0)
        #print(output.size())
        '''
        output = torch.index_select(output, 0, un_idx)
        
        for i in range(output.size()[0]):
            if i == 0:
                out = output[i, context_lens[i]-1, :].unsqueeze(0)
            else:
                out = torch.cat([out, output[i, context_lens[i]-1, :].unsqueeze(0)], dim=0)
        
        # hidden: out. outputs: output
        out = self.classifier(out)
        
        return out

class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        # 製作 embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否將 embedding fix住，如果fix_embedding為False，在訓練過程中，embedding也會跟著被訓練
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=False)
        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_dim, 1),
                                         nn.Sigmoid() )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs, context_lens):
        #inputs = self.embedding(inputs)
        context_lens = torch.tensor(context_lens).to(self.device)
        input_lens, idx = context_lens.sort(0, descending=True)
        _, un_idx = torch.sort(idx)
        inputs = inputs[idx]
        inputs = self.embedding(inputs)
        inputs_packed = pack_padded_sequence(inputs, input_lens, batch_first=True)
        x, _ = self.lstm(inputs_packed, None)
        output, _ = pad_packed_sequence(x, batch_first=True)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最後一層的 hidden state

        '''
        for i in range(x.size()[0]):
            if context_lens[i] > 31:
                context_lens[i] = 31
            if i == 0:
                output = x[i, context_lens[i]-1, :].unsqueeze(0)
            else:
                output = torch.cat([output, x[i, context_lens[i]-1, :].unsqueeze(0)], dim=0)
        #print(output.size())
        '''
        output = torch.index_select(output, 0, un_idx)

        for i in range(output.size()[0]):
            if i == 0:
                out = output[i, context_lens[i]-1, :].unsqueeze(0)
            else:
                out = torch.cat([out, output[i, context_lens[i]-1, :].unsqueeze(0)], dim=0)

        # hidden: out. outputs: output
        out = self.classifier(out)

        return out

