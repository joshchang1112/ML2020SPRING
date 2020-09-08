# 這個block用來先定義一些等等常用到的函式
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import re

def load_training_data(path='training_label.txt'):
    # 把training時需要的data讀進來
    # 如果是'training_label.txt'，需要讀取label，如果是'training_nolabel.txt'，不需要讀取label
    if 'training_label' in path:
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x

def load_testing_data(path='testing_data'):
    # 把testing時需要的data讀進來
    with open(path, 'r') as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
    return X

def evaluation(outputs, labels):
    #outputs => probability (float)
    #labels => labels
    outputs[outputs>=0.5] = 1 # 大於等於0.5為有惡意
    outputs[outputs<0.5] = 0 # 小於0.5為無惡意
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

def collate_fn(batch, padding=40633):
    if isinstance (batch[0],tuple):
        context_lens = [len(item[0]) for item in batch]
        padded_len = max(context_lens)
        data = torch.LongTensor(
          [pad_to_len(item[0], padded_len, padding) for item in batch])
        target = [item[1] for item in batch]
        target = torch.LongTensor(target)
        if len(batch[0]) == 2:
            return [data, target, context_lens]
        #print(len(batch[0][2]))
        #print(len(batch[0][2][0]))
        padded_len = 0
        for i in range(3):
            if i == 0:
                unlabel_context_lens = [[len(item[2][i]) for item in batch]]
            else:
                unlabel_context_lens.append([len(item[2][i]) for item in batch])
            if max(unlabel_context_lens[i]) > padded_len:
                padded_len = max(unlabel_context_lens[i])
        for i in range(3):
            if i == 0:
                unlabel_data = torch.LongTensor([pad_to_len(item[2][i], padded_len, padding) for item in batch]).unsqueeze(0)
            else:
                unlabel_data = torch.cat([unlabel_data, torch.LongTensor([pad_to_len(item[2][i], padded_len, padding) for item in batch]).unsqueeze(0)], dim=0)


        return [data, target, context_lens, unlabel_data, unlabel_context_lens]

    else:
        context_lens = [len(item) for item in batch]
        #padded_len = min(32, max(context_lens))
        padded_len = max(context_lens)
        data = torch.LongTensor(
          [pad_to_len(item, padded_len, padding) for item in batch])
        return [data, context_lens]

def pad_to_len(arr, padded_len, padding):
    """ Pad `arr` to `padded_len` with padding if `len(arr) < padded_len`.
    If `len(arr) > padded_len`, truncate arr to `padded_len`.
    Example:
        pad_to_len([1, 2, 3], 5, -1) == [1, 2, 3, -1, -1]
        pad_to_len([1, 2, 3, 4, 5, 6], 5, -1) == [1, 2, 3, 4, 5]
    Args:
        arr (list): List of int.
        padded_len (int)
        padding (int): Integer used to pad.
    """
    # TODO
    length_arr = len(arr)
    new_arr = arr
    if length_arr < padded_len:
        for i in range(padded_len - length_arr):
            new_arr.append(padding)
    else:
        for i in range(length_arr - padded_len):
            del new_arr[-1]
    #print(len(new_arr))
    return new_arr

