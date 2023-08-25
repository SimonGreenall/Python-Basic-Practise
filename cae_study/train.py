from model import ConvAutoEncoder
from data import DataLoader_CIFAR10
import torch.nn as nn
import torch.nn.functional as F

def train(epoch, net, trainset_loader, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    #分批次读入数据
    for batch_idx, (inputs, targets) in enumerate(trainset_loader):
        #优化器梯度置0
        optimizer.zero_grad()
        #数据通过网络
        outputs = net(inputs)
        #计算损失
        loss = criterion(outputs, targets)
        #反向传播
        loss.backward()
        #权重更新
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
	

