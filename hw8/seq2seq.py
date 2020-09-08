from model import Encoder, Decoder, Attention
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from queue import Queue
import operator

class Node(object):
    def __init__(self, hidden, previous_node, decoder_input, attn, output, log_prob, length):
        self.hidden = hidden
        self.previous_node = previous_node
        self.decoder_input = decoder_input
        self.output = output
        self.attn = attn
        self.log_prob = log_prob
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        #return self.log_prob
        return self.log_prob / float((self.length - 1)**0.7 + 1e-6) + alpha * reward


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
            
    def forward(self, input, target, teacher_forcing_ratio):
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        # teacher_forcing_ratio 是有多少機率使用正確答案來訓練
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.cn_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(input)
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        # 取的 <BOS> token
        input = target[:, 0]
        preds = []
        for t in range(1, target_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            # 決定是否用正確答案來做訓練
            teacher_force = random.random() <= teacher_forcing_ratio
            # 取出機率最大的單詞
            top1 = output.argmax(1)
            # 如果是 teacher force 則用正解訓練，反之用自己預測的單詞做預測
            input = target[:, t] if teacher_force and t < target_len else top1
            preds.append(top1.unsqueeze(1))
        
        preds = torch.cat(preds, 1)
        return outputs, preds

    def inference(self, input, target, beam_size):
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        batch_size = input.shape[0]
        input_len = input.shape[1]        # 取得最大字數
        vocab_size = self.decoder.cn_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, input_len, vocab_size).to(self.device)
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(input)
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        # 取的 <BOS> token
        input = target[:, 0]
        if beam_size == 1:
            outputs, preds = self.greedy_decode(input, input_len, hidden, encoder_outputs, outputs)
        else:
            # Beam Search
            outputs, preds = self.beam_decode(input, input_len, hidden, encoder_outputs, beam_size)


        return outputs, preds

    def greedy_decode(self, input, input_len, hidden, encoder_outputs, outputs):
        # beam size = 1
        preds = []
        for t in range(1, input_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            # 將預測結果存起來
            outputs[:, t] = output
            # 取出機率最大的單詞
            #top1 = output.topk(1)
            top1 = output.argmax(1)
            #print(top1.size())
            input = top1
            preds.append(top1.unsqueeze(1))
        
        preds = torch.cat(preds, 1)
        return outputs, preds

    def beam_decode(self, input, input_len, hidden, encoder_outputs, beam_size):
        # beam search
        root = Node(hidden, None, input, None, torch.zeros((1, 3805)).to(self.device), 0, 1)
        q = Queue()
        q.put((root.eval(), root))
        end_nodes = []
        qsize = 1
        while not q.empty():
            candidates = []
            # give up when decoding takes too long
            #if qsize > 2000: break
            for _ in range(q.qsize()):
                score, node = q.get()
                input = node.decoder_input
                hidden = node.hidden

                # EOS: 2
                #print(node.decoder_input.item())
                if node.decoder_input.item() == 2 or node.length >= 50:
                    end_nodes.append((node.log_prob, node))
                    continue

                output, hidden = self.decoder(input, hidden, encoder_outputs)
                
                # get top k candidates at this time step
                log_prob, indices = F.log_softmax(output, dim=1).topk(beam_size)

                # add log probability to previous node
                for k in range(beam_size):
                    index = indices[0][k].unsqueeze(0)
                    log_p = log_prob[0][k].item()
                    child = Node(hidden, node, index, None, output, node.log_prob + log_p, node.length + 1)
                    score = node.eval()
                    candidates.append((score, child))
            
            # select top k sentence to put them in Queue
            candidates = sorted(candidates, key=lambda x:x[0], reverse=True)
            length = min(len(candidates), beam_size)
            for i in range(length):
                q.put((candidates[i][0], candidates[i][1]))
            

        topk = 1 #  # how many sentence do you want to generate
        if len(end_nodes) == 0:
            end_nodes = [q.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(end_nodes, key=operator.itemgetter(0), reverse=True):
            utterance = []
            utterance.append(n.decoder_input.item())
            output_prob = n.output
            # back trace
            while n.previous_node != None:
                n = n.previous_node
                output_prob = torch.cat([n.output, output_prob], dim=0)
                utterance.append(n.decoder_input.item())
            
            utterance = utterance[:-1]
            utterance = utterance[::-1]
            utterances.append(utterance)
            break
        
        
        output_prob = output_prob.unsqueeze(0)
        if output_prob.size()[1] != 50:
            output_prob = torch.cat([output_prob, torch.zeros((1, 50-output_prob.size()[1], 3805)).to(self.device)], dim=1)

        return output_prob, utterances

