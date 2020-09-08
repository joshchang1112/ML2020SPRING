import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def schedule_sampling(steps, sampling='constant'):
    
    # constant
    if sampling == 'constant':
        return 0.5
    # linear decay
    if sampling == 'linear':
        return max(0.3, 1 - steps/15000)
    #exp decay
    if sampling == 'exp':
        k = 0.9998
        return k ** steps
    #inv sigmoid decay
    if sampling == 'inv_sig':
        k = 2500
        return k / (k + np.exp(steps/k))

def train(model, optimizer, train_iter, loss_function, total_steps, summary_steps, train_dataset):
    model.train()
    model.zero_grad()
    losses = []
    loss_sum = 0.0
    for step in range(summary_steps):
        sources, targets = next(train_iter)
        sources, targets = sources.to(device), targets.to(device)
        outputs, preds = model(sources, targets, schedule_sampling(total_steps + step, 'inv_sig'))
        # targets 的第一個 token 是 <BOS> 所以忽略
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)
        loss = loss_function(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        loss_sum += loss.item()
        if (step + 1) % 5 == 0:
            loss_sum = loss_sum / 5
            print ("\r", "train [{}] loss: {:.3f}, Perplexity: {:.3f}      ".format(total_steps + step + 1, loss_sum, np.exp(loss_sum)), end=" ")
            losses.append(loss_sum)
            loss_sum = 0.0

    return model, optimizer, losses
