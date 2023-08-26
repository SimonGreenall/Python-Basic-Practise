from utills import * #超参数准备
import torch.optim as optim #准备优化器
from train import train #准备训练函数
from predict import predict
from data import DataLoader_CIFAR10 #准备数据集
from model import ConvAutoEncoder #准备我们写好的CAE模型
import matplotlib.pyplot as plt #准备画图工具

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
	optimizer = optim.SGD(net.parameters(), lr=l_r, momentum=0.9, weight_decay=5e-4) #构建SGD优化器

	#训练
	losses = []
	for i in range(epoch):
		loss = train(i+1,net,trainset_loader,optimizer)
		losses.append(loss)
	
	plt.plot(loss,color='r')
	plt.xlabel('iter')
	plt.ylabel('loss')
	plt.grid()
	plt.show()
	print("success")
	
	
	