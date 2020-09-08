import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

def testing(batch_size, test_loader, model, device, step=0):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, (inputs, context_lens) in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs, context_lens)
            outputs = outputs.squeeze()
            if step == 1:
                outputs[outputs>=0.5] = 1 # 大於等於0.5為負面
                outputs[outputs<0.5] = 0 # 小於0.5為正面
                ret_output += outputs.int().tolist()
            else:
                ret_output += outputs.tolist()
    return ret_output
