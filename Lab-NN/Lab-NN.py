import numpy as np
import matplotlib.pyplot as plt

class NeutralNetwork():
    def __init__(self) -> None:
         pass
    
    def initialize_parameters(self,n_x=8,n_h=3,n_y=8):
        W1 = np.random.randn(n_h,n_x) * 0.01 #初始化隐藏层W，因为隐藏层有3个结点，输入有8个，所以W1是一个（3，8）的矩阵，乘0.01是因为通常要把w的初始值设定为很小很小的非零随机数
        b1 = np.zeros((n_h,1)) #初始化隐藏层b，因为隐藏层有三个结点，所以b为（3，1)的矩阵
        W2 = np.random.randn(n_y,n_h) * 0.01 #初始化输出层W，输出层有8个结点，输入是隐藏层的三个计算结果作为输入，所以W2是一个（8，3）的矩阵
        b2 = np.zeros((n_y,1)) #初始化输出层b

        # 使用断言确保我的数据格式是正确的,括号内条件为false的时候会触发异常
        assert(W1.shape == ( n_h , n_x ))
        assert(b1.shape == ( n_h , 1 ))
        assert(W2.shape == ( n_y , n_h ))
        assert(b2.shape == ( n_y , 1 ))

        parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2} #用字典保存好我们初始化的值
    
        return parameters #返回
    
    def forward_propagation(self,X,parameters):
            # 从字典 “parameters” 中检索每个参数
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]

            # 实现前向传播计算A2(输出结果)
            Z1 = np.dot(W1,X) + b1
            A1 = np.tanh(Z1)
            Z2 = np.dot(W2,A1) + b2
            A2 = 1/(1+np.exp(-Z2)) #sigmoid函数

            #使用断言确保我的数据格式是正确的
            assert(A2.shape == X.shape)
    
            cache = {"Z1": Z1,
                    "A1": A1,
                    "Z2": Z2,
                    "A2": A2}
    
            return A2, cache
    
    def compute_cost(self,A2,Y):
         #样本数量
         m=Y.shape[1]
         # 计算交叉熵代价
         logprobs = Y*np.log(A2) + (1-Y)* np.log(1-A2)
         cost = -1/m * np.sum(logprobs)
         # 确保损失是我们期望的维度
         cost = np.squeeze(cost)
         #用断言确保cost是float类型，不是的话会抛出异常
         assert(isinstance(cost, float))
         return cost
    
    def backward_propagation(self,parameters,cache,X,Y):
         #输入样本的数量
         m=X.shape[1]
         # 首先，从字典“parameters”中检索W1和W2
         W1=parameters["W1"]
         W2=parameters["W2"]
         #  从字典“cache”中检索A1和A2
         A1=cache["A1"]
         A2=cache["A2"]
         # 反向传播:计算 dW1、db1、dW2、db2
         dZ2= A2 - Y
         dW2 = 1 / m * np.dot(dZ2,A1.T)
         db2 = 1 / m * np.sum(dZ2,axis=1,keepdims=True)
         dZ1 = np.dot(W2.T,dZ2) * (1-np.power(A1,2))
         dW1 = 1 / m * np.dot(dZ1,X.T)
         db1 = 1 / m * np.sum(dZ1,axis=1,keepdims=True)

         grads = {"dW1": dW1,
                  "db1": db1,
                  "dW2": dW2,
                  "db2": db2}
         
         return grads
    
    def update_parameters(self,parameters, grads, learning_rate = 1.2):
         # 从字典“parameters”中检索每个参数
         W1 = parameters["W1"]
         b1 = parameters["b1"]
         W2 = parameters["W2"]
         b2 = parameters["b2"]
    
         # 从字典“梯度”中检索每个梯度
         dW1 = grads["dW1"]
         db1 = grads["db1"]
         dW2 = grads["dW2"]
         db2 = grads["db2"]

         # 每个参数的更新规则
         W1 = W1 - learning_rate * dW1
         b1 = b1 - learning_rate * db1
         W2 = W2 - learning_rate * dW2
         b2 = b2 - learning_rate * db2

         parameters = {"W1": W1,
                       "b1": b1,
                       "W2": W2,
                       "b2": b2}
         
         return parameters
    
    def train(self,X,Y,epoch):
         parameters = self.initialize_parameters()
         W1 = parameters["W1"]
         b1 = parameters["b1"]
         W2 = parameters["W2"]
         b2 = parameters["b2"]

         #开始循环（梯度下降）
         for i in range(0,epoch):
              for j in range(0,1000): #每个epoch训练一千次
                   #前向传播
                   A2, cache = self.forward_propagation(X, parameters)
                   #计算成本
                   cost = self.compute_cost(A2, Y)
                   #反向传播
                   grads = self.backward_propagation(parameters, cache, X, Y)
                   #更新参数
                   parameters = self.update_parameters(parameters, grads)
              
              print ("Cost after epoch %i: %f" %(i, cost))

         return parameters
    
    def predict(self,parameters,X):
         # 使用前向传播计算概率，并使用 0.5 作为阈值将其分类为 0/1。
         A2, cache = self.forward_propagation(X, parameters)
         predictions = np.round(A2)
    
         return predictions
    
lab_nn=NeutralNetwork()
x=np.eye(8)
y=np.eye(8)
lab_nn.train(x,y,10)