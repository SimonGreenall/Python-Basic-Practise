import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training') #创建参数解析器
parser.add_argument('--lr', default=0.1, type=float, help='learning rate') #学习率
args = parser.parse_args() #使以上这些关于parser的代码生效
