import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np

def teacher_selector(teachers):
    idx = np.random.randint(len(teachers))
    return teachers[idx]

def loss_fn_kd(student_outputs, labels, teacher_outputs, T=20, alpha=0.5, test=None):
    
    # 讓logits的log_softmax對目標機率(teacher的logits/T後softmax)做KL Divergence。
    for i in range(len(teacher_outputs)):
        if i == 0:
            teacher_loss = F.softmax(teacher_outputs[i]/T, dim=1) / len(teacher_outputs)
        else:
            teacher_loss += F.softmax(teacher_outputs[i]/T, dim=1) / len(teacher_outputs)

    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_outputs/T, dim=1),
                             teacher_loss) * (alpha * T * T)
    
    # 一般的Cross Entropy
    if test == None:
        hard_loss = F.cross_entropy(student_outputs, labels) * (1. - alpha)
        return hard_loss + soft_loss
    else:
        return soft_loss
    

def training(teacher_net, student_net, train, valid, test=None, total_epoch=100, alpha=0.5):
    # TeacherNet永遠都是Eval mode.
    # test means semi-supervise

    optimizer = optim.AdamW(student_net.parameters(), lr=7e-4)
    for i in range(len(teacher_net)):
        teacher_net[i].eval()
    
    now_best_acc = 0
    
    for epoch in range(total_epoch):
        epoch_start_time = time.time()
        if epoch == 300:
            optimizer = optim.SGD(student_net.parameters(), lr=2e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 35400, eta_min=1e-6)
        
        student_net.train()

        train_num, train_hit, train_loss = 0, 0, 0
        for now_step, batch_data in enumerate(train):

            optimizer.zero_grad()
            inputs, hard_labels = batch_data
            inputs = inputs.cuda()
            hard_labels = torch.LongTensor(hard_labels).cuda()
        
            # 因為Teacher沒有要backprop，所以我們使用torch.no_grad
            # 告訴torch不要暫存中間值(去做backprop)以浪費記憶體空間。
            soft_labels = []
            with torch.no_grad():
                #teacher = teacher_selector(teacher_net)
                for i in range(len(teacher_net)):
                    soft_labels.append(teacher_net[i](inputs)) 
            
            logits = student_net(inputs)
            # 使用我們之前所寫的融合soft label&hard label的loss。
            # T=20是原始論文的參數設定。
            
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            loss.backward()
            optimizer.step()    
            if epoch >= 300:
                scheduler.step()
            train_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
            train_num += len(inputs)
            train_loss += loss.item() * len(inputs)
        
        for now_step, inputs in enumerate(test):

            optimizer.zero_grad()
            inputs = inputs.cuda()
        
            # 因為Teacher沒有要backprop，所以我們使用torch.no_grad
            # 告訴torch不要暫存中間值(去做backprop)以浪費記憶體空間。
            soft_labels = []
            with torch.no_grad():
                #teacher = teacher_selector(teacher_net)
                for i in range(len(teacher_net)):
                    soft_labels.append(teacher_net[i](inputs)) 
            
            logits = student_net(inputs)
            # 使用我們之前所寫的融合soft label&hard label的loss。
            # T=20是原始論文的參數設定。
            
            loss = loss_fn_kd(logits, None, soft_labels, 20, alpha, test=1)
            loss.backward()
            optimizer.step()    

        # validation
        student_net.eval()
        
        valid_num, valid_hit, valid_loss = 0, 0, 0
        for now_step, batch_data in enumerate(valid):

            inputs, hard_labels = batch_data
            inputs = inputs.cuda()
            hard_labels = torch.LongTensor(hard_labels).cuda()
             
            soft_labels = []
            with torch.no_grad():
                #teacher = teacher_selector(teacher_net)
                for i in range(len(teacher_net)):
                    soft_labels.append(teacher_net[i](inputs)) 

            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
 
            valid_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
            valid_num += len(inputs)
            valid_loss += loss.item() * len(inputs)
        
        train_acc = train_hit / train_num 
        valid_acc = valid_hit / valid_num
        train_loss = train_loss / train_num
        valid_loss = valid_loss / valid_num
        
        # 存下最好的model。
        if valid_acc > now_best_acc:
            now_best_acc = valid_acc
            torch.save(student_net.state_dict(), 'student_model.bin')
            print('saving model with acc {:.3f}'.format(valid_acc * 100))
        print('epoch {:>3d}: {:2.2f} sec(s), train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
            epoch, time.time()-epoch_start_time, train_loss, train_acc, valid_loss, valid_acc))



def deep_mutual_learning(models, train, valid, total_epoch=100, T=1):
    
    model_num = len(models)
    optimizers = []
    schedulers = []
    now_best_acc = []
    for i in range(model_num):
        optimizers.append(optim.SGD(models[i].parameters(), lr=2e-3, momentum=0.9,
                                  weight_decay=5e-4))
        schedulers.append(optim.lr_scheduler.CosineAnnealingLR(optimizers[i], 35400, eta_min=1e-6))

    for i in range(model_num):
        now_best_acc.append(0)

    for epoch in range(total_epoch):
        epoch_start_time = time.time()

        for i in range(model_num):
            models[i].train()

        train_num, train_hit, train_loss, train_acc = [], [], [], []
        for i in range(model_num):
            train_num.append(0)
            train_hit.append(0)
            train_loss.append(0)
            train_acc.append(0)

        for now_step, batch_data in enumerate(train):

            inputs, hard_labels = batch_data
            inputs = inputs.cuda()
            hard_labels = torch.LongTensor(hard_labels).cuda()
            outputs=[]
            for i in range(model_num):
                outputs.append(models[i](inputs))
            
            for i in range(model_num):
                ce_loss = F.cross_entropy(outputs[i], hard_labels)
                kl_loss = 0
                for j in range(model_num):
                    if i != j:
                        kl_loss += nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs[i]/T, dim=1),
                                 F.softmax(outputs[i]/T, dim=1)) * (T * T)
                
                loss = ce_loss + kl_loss / (model_num - 1)
                optimizers[i].zero_grad()
                loss.backward()
                schedulers[i].step()
                optimizers[i].step()

                train_hit[i] += torch.sum(torch.argmax(outputs[i], dim=1) == hard_labels).item()
                train_num[i] += len(inputs)
                train_loss[i] += loss.item() * len(inputs)

        for i in range(model_num):
            models[i].eval()

        valid_num, valid_hit, valid_loss, valid_acc = [], [], [], []
        
        for i in range(model_num):
            valid_num.append(0)
            valid_hit.append(0)
            valid_loss.append(0)
            valid_acc.append(0)
        
        for now_step, batch_data in enumerate(valid):

            inputs, hard_labels = batch_data
            inputs = inputs.cuda()
            hard_labels = torch.LongTensor(hard_labels).cuda()
            
            outputs=[]
            for i in range(model_num):
                with torch.no_grad():
                    outputs.append(models[i](inputs))

            for i in range(model_num):
                ce_loss = F.cross_entropy(outputs[i], hard_labels)
                kl_loss = 0
                for j in range(model_num):
                    if i != j:
                        kl_loss += nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs[i]/T, dim=1),
                                 F.softmax(outputs[i]/T, dim=1)) * (T * T)
                
                loss = ce_loss + kl_loss / (model_num - 1)
            
                valid_hit[i] += torch.sum(torch.argmax(outputs[i], dim=1) == hard_labels).item()
                valid_num[i] += len(inputs)
                valid_loss[i] += loss.item() * len(inputs)
        
        for i in range(model_num):
            train_acc[i] = train_hit[i] / train_num[i]
            valid_acc[i] = valid_hit[i] / valid_num[i]
            train_loss[i] = train_loss[i] / train_num[i]
            valid_loss[i] = valid_loss[i] / valid_num[i]
        
            # 存下最好的model。
            if valid_acc[i] > now_best_acc[i]:
                now_best_acc[i] = valid_acc[i]
                torch.save(models[i].state_dict(), 'model_{}.bin'.format(i))
                print('saving model with acc {:.3f}'.format(valid_acc[i] * 100))

            print('Model {}, epoch {:>3d}: {:2.2f} sec(s), train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
                i, epoch, time.time()-epoch_start_time, train_loss[i], train_acc[i], valid_loss[i], valid_acc[i]))

