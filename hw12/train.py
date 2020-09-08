import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.

def train_epoch(source_dataloader, target_dataloader, model, optimizer, lamb):
    '''
      Args:
        source_dataloader: source data的dataloader
        target_dataloader: target data的dataloader
        lamb: 調控adversarial的loss係數。
    '''

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    # total_hit: 計算目前對了幾筆 total_num: 目前經過了幾筆
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0
    
    optimizer_F = optimizer[0]
    optimizer_C = optimizer[1]
    optimizer_D = optimizer[2]

    feature_extractor = model[0]
    label_predictor = model[1]
    domain_classifier = model[2]

    for i, ((source_data,source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):

        source_data = source_data.to(device)
        #s2 = s2.to(device)
        #s3 = s3.to(device)
        source_label = source_label.to(device)
        target_data = target_data.to(device)
        
        # 我們把source data和target data混在一起，否則batch_norm可能會算錯 (兩邊的data的mean/var不太一樣)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label1 = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).to(device)
        # 設定source data的label為1
        domain_label1[:source_data.shape[0]] = 1
        #mixed_data = mixed_data.reshape(len(mixed_data), -1)
        # Step 1 : 訓練Domain Classifier
        feature1 = feature_extractor(mixed_data)
        # 因為我們在Step 1不需要訓練Feature Extractor，所以把feature detach避免loss backprop上去。
        domain_logits = domain_classifier(feature1.detach())
        loss_1 = domain_criterion(domain_logits, domain_label1)
        running_D_loss+= loss_1.item()
        '''
        mixed_data = torch.cat([s2, target_data], dim=0)
        domain_label2 = torch.zeros([s2.shape[0] + target_data.shape[0], 1]).to(device)
        domain_label2[:s2.shape[0]] = 1
        feature2 = feature_extractor(mixed_data)
        domain_logits = domain_classifier(feature2.detach())
        loss_2 = domain_criterion(domain_logits, domain_label2)
        running_D_loss+= loss_2.item()

        mixed_data = torch.cat([s3, target_data], dim=0)
        domain_label3 = torch.zeros([s3.shape[0] + target_data.shape[0], 1]).to(device)
        domain_label3[:s3.shape[0]] = 1
        feature3 = feature_extractor(mixed_data)
        domain_logits = domain_classifier(feature3.detach())
        loss_3 = domain_criterion(domain_logits, domain_label3)
        running_D_loss+= loss_3.item()
        '''
        #loss = (loss_1 + loss_2 + loss_3) / 3
        #running_D_loss /= 3

        loss_1.backward()
        optimizer_D.step()

        # Step 2 : 訓練Feature Extractor和Domain Classifier
        class_logits = label_predictor(feature1[:source_data.shape[0]])
        domain_logits = domain_classifier(feature1)
        # loss為原本的class CE - lamb * domain BCE，相減的原因同GAN中的Discriminator中的G loss。
        loss_1 = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label1)
        running_F_loss+= loss_1.item()
        '''
        class_logits = label_predictor(feature2[:source_data.shape[0]])
        domain_logits = domain_classifier(feature2)
        # loss為原本的class CE - lamb * domain BCE，相減的原因同GAN中的Discriminator中的G loss。
        loss_2 = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label2)
        running_F_loss+= loss_2.item()

        class_logits = label_predictor(feature3[:source_data.shape[0]])
        domain_logits = domain_classifier(feature3)
        # loss為原本的class CE - lamb * domain BCE，相減的原因同GAN中的Discriminator中的G loss。
        loss_3 = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label3)
        running_F_loss+= loss_3.item()

        loss = (loss_1 + loss_2 + loss_3) / 3
        #running_F_loss /= 3
        '''
        loss_1.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')
    
    
    
    return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num

def train(source_dataloader, target_dataloader, feature_extractor, label_predictor, domain_classifier, max_epoch):
    
    optimizer_F = optim.Adam(feature_extractor.parameters(), lr=1e-4)
    optimizer_C = optim.Adam(label_predictor.parameters())
    optimizer_D = optim.Adam(domain_classifier.parameters())
    optimizer = [optimizer_F, optimizer_C, optimizer_D]
    model = [feature_extractor, label_predictor, domain_classifier]

    for epoch in range(1, max_epoch+1):
        lamb = 0.1 * get_lambda(epoch, max_epoch)
        train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, model, optimizer, lamb)


        print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc))
        if epoch % 50 == 0:
            torch.save(feature_extractor.state_dict(), 'extractor_model_{}.bin'.format(epoch))
            torch.save(label_predictor.state_dict(), 'predictor_model_{}.bin'.format(epoch))
