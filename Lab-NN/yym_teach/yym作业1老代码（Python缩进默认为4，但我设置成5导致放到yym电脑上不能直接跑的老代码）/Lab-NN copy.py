import numpy as np
import matplotlib.pyplot as plt
from utils import *
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
    
#     def compute_cost_with_regularization(self,A2,Y,parameters,lambd,flag):
#          m=Y.shape[1] #样本数量存到m里去
#          logprobs = Y*np.log(A2) + (1-Y)* np.log(1-A2)
#          if(flag==True):
#               W1 = parameters["W1"]
#               W2 = parameters["W2"]
#               L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2))) / (2 * m) #L2正则项
#               cost = -1/m * np.sum(logprobs) + L2_regularization_cost #正则化损失
#               cost = np.squeeze(cost)
#               assert(isinstance(cost, float))
#               return cost
#          else:
#               cost = -1/m * np.sum(logprobs)
#               cost = np.squeeze(cost)
#               assert(isinstance(cost, float))
#               return cost
    
    def backward_propagation(self, X, Y, cache, parameters): #反向传播(未使用正则化)
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
    
#     def backward_propagation_with_regularization(self, X, Y, cache, parameters,lambd): #正则化反向传播
#          m = X.shape[1] #输入样本数量
#          A1=cache["A1"]
#          A2=cache["A2"]
#          W1=parameters["W1"]
#          W2=parameters["W2"]
#          dZ2 = A2 - Y
#          dW2 = (1 / m) * np.dot(dZ2,A1.T) + ((lambd * W2) / m)
#          db2 = (1 / m) * np.sum(dZ2,axis=1,keepdims=True)
#          dA1 = np.dot(W2.T,dZ2)
#          dZ1 = np.multiply(dA1,np.int64(A1 > 0))
#          dW1 = (1 / m) * np.dot(dZ1,X.T) + ((lambd * W1) / m)
#          db1 = (1 / m) * np.sum(dZ1,axis=1,keepdims=True)
#          grads = { "dW1": dW1, 
#                    "db1": db1, 
#                    "dW2": dW2, 
#                    "db2": db2}
#          return grads
    
    def update_parameters(self,parameters, grads, learning_rate):
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
    
    # def train(self,X,Y,epoch,lambd,learning_rate,losses):
    #      losses1 = []
    #      losses2 = []
    #      parameters = self.initialize_parameters()

    #      #开始循环（梯度下降） 有正则
    #      for i in range(0,epoch):
    #           for _ in range(0,1000): #每个epoch训练一千次,用_可以避免开辟新变量
    #                #前向传播
    #                A2, cache = self.forward_propagation(X, parameters)
    #                #计算成本
    #                cost = self.compute_cost_with_regularization(A2,Y,parameters,lambd)
    #                #反向传播
    #                grads = self.backward_propagation_with_regularization(X, Y, cache, parameters,lambd)
    #                #更新参数
    #                parameters = self.update_parameters(parameters, grads, learning_rate)

    #                losses1.append(cost)

    #           print ("Cost with regularization after epoch %i: %f" %(i, cost))
          
    #       #开始循环（梯度下降） 无正则
    #      for i in range(0,epoch):
    #           for _ in range(0,1000): #每个epoch训练一千次,用_可以避免开辟新变量
    #                #前向传播
    #                A2, cache = self.forward_propagation(X, parameters)
    #                #计算成本
    #                cost = self.compute_cost(A2,Y)
    #                #反向传播
    #                grads = self.backward_propagation(X, Y, cache, parameters)
    #                #更新参数
    #                parameters = self.update_parameters(parameters, grads, learning_rate)

    #                losses2.append(cost)
              
    #           print ("Cost after epoch %i: %f" %(i, cost))
          
    #      plt.plot(losses1,color='red',label='with regularization')
    #      plt.plot(losses2,color='blue',label='no regularization')
    #      plt.xlabel('epoch')
    #      plt.ylabel('loss')
    #      plt.legend(loc="upper right")
    #      plt.show()

    #      return parameters

    def train(self,X,Y,epoch,learning_rate,decay_rate,losses):
         parameters = self.initialize_parameters()
          
          #开始循环（梯度下降） 无正则
         for i in range(0,epoch):
              for j in range(0,1000): #每个epoch训练一千次,用_可以避免开辟新变量
                   #前向传播
                   A2, cache = self.forward_propagation(X, parameters)
                   #计算成本
                   cost = self.compute_cost(A2,Y)
                   #反向传播
                   grads = self.backward_propagation(X, Y, cache, parameters)
                   #更新参数
                   decay_learning_rate = learning_rate * np.power(decay_rate,(i*1000+j)/1000) #实现学习率衰减
                   parameters = self.update_parameters(parameters, grads, decay_learning_rate)

                   losses.append(cost)
              
              print ("Cost after epoch %i: %f" %(i, cost))

         return parameters,losses
    
    def predict(self,parameters,X):
         # 使用前向传播计算概率，并使用 0.5 作为阈值将其分类为 0/1。
         A2, cache = self.forward_propagation(X, parameters)
         predictions = np.round(A2)
    
         return predictions

if __name__ == "__main__":
    lab_nn=NeutralNetwork()
    x = y = np.eye(8)
    learning_rates = LEARNING_RATES
    decay_rates = DECAY_RATES
    losses = []
    losses_all = []
    for learning_rate in learning_rates:
         print("start training with learning_rate = " + str(learning_rate))
         lab_nn.train(x, y, 10, learning_rate,1, losses)
         losses_all.append(losses)
    i = 0
    for loss in losses_all:
         i+=1
         plt.plot(loss,label='learning_rate='+str(learning_rates[i-1]))
         plt.xlabel('iteration')
         plt.ylabel('loss')
         plt.legend(loc="upper right")
     
    plt.show()

    losses1 = []
    losses_all1 = []
    for decay_rate in decay_rates:
         print("start training with decay_rate = " + str(decay_rate))
         lab_nn.train(x, y, 10, 1.2,decay_rate, losses1)
         losses_all1.append(losses1)
    i = 0
    for loss in losses_all1:
         i+=1
         plt.plot(loss,label='decay_rate='+str(decay_rates[i-1]))
         plt.xlabel('iteration')
         plt.ylabel('loss')
         plt.legend(loc="upper right")

    plt.show()       