from utills import * #超参数准备
import torch.optim as optim
from train import train
from predict import predict
from data import DataLoader_CIFAR10
from model import ConvAutoEncoder

if __name__ == "__main__":
	#数据准备
	print("准备数据")
	data = DataLoader_CIFAR10() #加载CIFAR10
	dt = data.combine_dataset() #训练集和测试集合并
	trainset, valset, testset = data.split_dataset(dt) #分割数据集
	trainset_loader, valset_loader, testset_loader = data.upload_dataset(trainset, valset, testset) #得到三个数据集

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	#构建模型
	net = ConvAutoEncoder()
	optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) #构建优化器
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

	#训练
	train(10,net,trainset_loader,optimizer)
	
	