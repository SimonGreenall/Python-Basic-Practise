from model import ConvAutoEncoder
from data import DataLoader_CIFAR10
import torch.nn as nn
import torch.nn.functional as F

#在深度学习中，通常使用监督学习来训练模型。监督学习需要输入和输出之间的对应关系。
# 在这种情况下，inputs是模型的输入数据，而targets是模型的期望输出数据。
# 例如，在图像分类任务中，输入可能是图像数据，而目标可能是该图像所属的类别。

def train(epoch, net, trainset_loader, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train() #设置为train模式
    losses = []
    criterion = nn.CrossEntropyLoss() #定义损失函数
    #分批次读入数据
    for (inputs, targets) in trainset_loader: 
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

        losses.append(loss)
    
    print("Loss after epoch %d: %f" % (epoch, loss))
    return losses
    
	

