import torch.nn as nn
import torch.nn.functional as F

class ConvAutoEncoder(nn.Module):
	def __init__(self):
		super(ConvAutoEncoder,self).__init__()
		self.conv1 = nn.Conv2d(3,8,3) #表示输入图像通道数为3，卷积产生的通道数为8，卷积核尺寸为3×3
		self.bn1 = nn.BatchNorm2d(8) #在第一层卷积后应用批量归一化
		self.drop_out1 = nn.Dropout2d(0.25) #将元素置0的概率为0.25
		self.conv2 = nn.Conv2d(8,12,3) #表示输入图像通道数为8，卷积产生的通道数为12，卷积核尺寸为3×3
		self.bn2 = nn.BatchNorm2d(12) #在第二层卷积后应用批量归一化
		self.drop_out2 = nn.Dropout2d(0.5) #将元素置0的概率为0.5
		self.conv3 = nn.Conv2d(12,16,3) #表示输入图像通道数为12，卷积产生的通道数为16，卷积核尺寸为3×3
		self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2) #实现一个2×2的上采样层
		self.conv4 = nn.Conv2d(16,12,3) #表示输入图像通道数为16，卷积产生的通道数为12，卷积核尺寸为3×3
		self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2) #实现一个2×2的上采样层
		self.conv5 = nn.Conv2d(12,3,3) #表示输入图像通道数为12，卷积产生的通道数为3，卷积核尺寸为3×3

	def forward(self, x):
		x = self.conv1(x) #输入经过第一个卷积层（在_init_里设定好Conv2D的参数，赋给conv1，然后就能通过x = self.conv1(x)的方式让inputs通过conv1，这肯定是继承的模块设定好的，而且在train.py里还不用特地调用net.forward(inputs)，就直接net(inputs)，就能自动对上这里的x，这肯定也是继承的父类模块设定好的，不要纠结！）
		x = F.relu(x) #激活
		x = F.max_pool2d(x,2) #对输入进行最大池化，且最大池化窗口为2×2
		x = self.bn1(x) #批归一化
		x = self.drop_out1(x) #dropout防止过拟合
		x = self.conv2(x) #输入经过第二个卷积层
		x = F.relu(x) #激活
		x = F.max_pool2d(x,2) #对输入进行最大池化，且最大池化窗口为2×2
		x = self.bn2(x) #批归一化
		x = self.drop_out2(x) #dropout防止过拟合
		x = self.conv3(x) #输入经过第三个卷积层
		x = F.relu(x) #激活
		x = self.upsample1(x) #第一次上采样
		x = self.conv4(x) #输入经过第四个卷积层
		x = F.relu(x) #激活
		x = self.upsample2(x) #第二次上采样
		x = self.conv5(x) #输入经过第五个卷积层
		x = F.softmax(x) #最后softmax输出
		return x


