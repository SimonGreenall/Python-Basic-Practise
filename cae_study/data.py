import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

class DataLoader_CIFAR10():
	
	def __init__(self):
		#数据预处理	
		self.transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
    		transforms.RandomHorizontalFlip(),
    		transforms.ToTensor(),
    		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
		self.transform_test = transforms.Compose([
			transforms.ToTensor(),
    		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
		
		self.train_data = dset.CIFAR10(root='../data', train=True, transform=self.transform_train ,download=False)
		self.test_data = dset.CIFAR10(root='../data', train=False, transform=self.transform_test ,download=False)
	
	def combine_dataset(self):
		dt = ConcatDataset([self.train_data,self.test_data]) #将读取到的CIFAR10训练集和测试集合并形成新的数据集
		return dt

	def split_dataset(self,dt):
		trainset, valset, testset = data.random_split(dt,[48000,6000,6000]) #将合并的数据集按8:1:1分割成训练集，验证集，测试集
		return trainset, valset, testset

	def upload_dataset(self,trainset, valset, testset):
		trainset = trainset
		valset = valset
		testset = testset
		trainset_loader = data.DataLoader(trainset, batch_size=64, shuffle=True) 
		valset_loader = data.DataLoader(valset, batch_size=64, shuffle=True) 
		testset_loader = data.DataLoader(testset, batch_size=64, shuffle=True) #这样的参数设置，默认sampler=None,然后shuffle==True,那么sampler = RandomSampler(dataset)
		return trainset_loader, valset_loader, testset_loader