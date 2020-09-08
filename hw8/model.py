import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
  def __init__(self, hid_dim):
    super(Attention, self).__init__()
    self.hid_dim = hid_dim
    self.general_attn = nn.Linear(hid_dim * 2, hid_dim *2)
    self.concat_attn = nn.Linear(hid_dim * 4, hid_dim *2)
    self.vector = nn.Linear(hid_dim*2, 1)

  def forward(self, encoder_outputs, decoder_hidden, type='dot'):
    # encoder_outputs = [batch size, sequence len, hid dim * directions]
    # decoder_hidden = [num_layers, batch size, hid dim]
    # 一般來說是取 Encoder 最後一層的 hidden state 來做 attention

    if type == 'dot':
        decoder_hidden = decoder_hidden[-1, :, :].unsqueeze(2)
        attention = torch.bmm(encoder_outputs, decoder_hidden)
        attention = torch.sum(torch.mul(encoder_outputs, attention), dim=1).unsqueeze(1)
         
    elif type == 'general':
        decoder_hidden = decoder_hidden[-1, :, :].unsqueeze(2)
        attention = self.general_attn(encoder_outputs)
        attention = torch.bmm(encoder_outputs, decoder_hidden)
        attention = torch.sum(torch.mul(encoder_outputs, attention), dim=1).unsqueeze(1)

    elif type == 'concat':
        decoder_hidden = decoder_hidden[-1, :, :].unsqueeze(1)
        decoder_hidden = decoder_hidden.repeat(1, encoder_outputs.size()[1], 1)
        attention = self.vector(F.tanh(self.attn(torch.cat([decoder_hidden, encoder_outputs], dim=2))))
        attention = torch.sum(torch.mul(encoder_outputs, attention), dim=1).unsqueeze(1)

    return attention

class Encoder(nn.Module):
  def __init__(self, en_vocab_size, emb_dim, hid_dim, n_layers, dropout):
    super().__init__()
    self.embedding = nn.Embedding(en_vocab_size, emb_dim)
    self.hid_dim = hid_dim
    self.n_layers = n_layers
    self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
    self.dropout = nn.Dropout(dropout)

  def forward(self, input):
    # input = [batch size, sequence len, vocab size]
    embedding = self.embedding(input)
    outputs, hidden = self.rnn(self.dropout(embedding))
    # outputs = [batch size, sequence len, hid dim * directions]
    # hidden =  [num_layers * directions, batch size  , hid dim]
    # outputs 是最上層RNN的輸出
        
    return outputs, hidden

class Decoder(nn.Module):
  def __init__(self, cn_vocab_size, emb_dim, hid_dim, n_layers, dropout, isatt):
    super().__init__()
    self.cn_vocab_size = cn_vocab_size
    self.hid_dim = hid_dim * 2
    self.n_layers = n_layers
    self.embedding = nn.Embedding(cn_vocab_size, emb_dim)
    self.isatt = isatt
    self.attention = Attention(hid_dim)
    # 如果使用 Attention Mechanism 會使得輸入維度變化，請在這裡修改
    # e.g. Attention 接在輸入後面會使得維度變化，所以輸入維度改為
    #self.input_dim = emb_dim + hid_dim * 2 if isatt else emb_dim
    self.input_dim = emb_dim
    
    self.rnn = nn.GRU(self.input_dim, self.hid_dim, self.n_layers, dropout = dropout, batch_first=True)
    self.embedding2vocab1 = nn.Linear(self.hid_dim * 2, self.hid_dim * 2)
    self.embedding2vocab2 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)
    self.embedding2vocab3 = nn.Linear(self.hid_dim * 4, self.cn_vocab_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, input, hidden, encoder_outputs):
    # input = [batch size, vocab size]
    # hidden = [batch size, n layers * directions, hid dim]
    # Decoder 只會是單向，所以 directions=1
    input = input.unsqueeze(1)
    embedded = self.dropout(self.embedding(input))
    # embedded = [batch size, 1, emb dim]
    if self.isatt:
      attn = self.attention(encoder_outputs, hidden)

    # Structure 1: concat attn with embedded input
    #output, hidden = self.rnn(torch.cat([attn, embedded], dim=2), hidden)
    # Structure 2: concat attn with output(before fc layer)
    output , hidden = self.rnn(embedded, hidden)
    output = torch.cat([output, attn], dim=2)

    # output = [batch size, 1, hid dim]
    # hidden = [num_layers, batch size, hid dim]
    output = self.embedding2vocab1(output.squeeze(1))
    output = self.embedding2vocab2(output)
    prediction = self.embedding2vocab3(output)
    # prediction = [batch size, vocab size]
    return prediction, hidden
