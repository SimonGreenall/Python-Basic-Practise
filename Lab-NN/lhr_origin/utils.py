import numpy as np
import matplotlib.pyplot as plt
from train import train,train_with_batch_change,train_test
import math
import model

LEARNING_RATES = [0.1, 0.01, 0.05, 0.005,1.2, 1, 1.3]
DECAY_RATES = [0.98, 0.97,  0.96, 0.95, 0.94]
LOSS_FUNCTIONS = ['cee','mse','mae']
BETA_1 = [0.5, 0.6, 0.7, 0.8, 0.9]
BETA_2 = [0.995, 0.996, 0.997, 0.998, 0.999]
BATCH_SIZES = [1,4,8,16]
EPOCHS = [1,3,5,7,9]

def compute_indexes(predicted,expected): #模块化计算性能指标
	TP = np.sum(np.logical_and(predicted == 1,expected == 1))
	TN = np.sum(np.logical_and(predicted == 0,expected == 0))
	FP = np.sum(np.logical_and(predicted == 1,expected == 0))
	FN = np.sum(np.logical_and(predicted == 0,expected == 1))
	accuracy = (TP + TN) / (TP + TN + FP + FN)
	recall = TP / (TP + FN)
	precision = TP / (TP + FP)
	F1 = 2 * precision * recall / (precision + recall)
	return accuracy,recall,precision,F1


def plot(save_fig):
	plt.grid()
	plt.savefig(save_fig)
	plt.show()
	print("succeed")

# def random_mini_batches(X, Y, mini_batch_size):    
#     """
#     从（X，Y）中创建一个随机的mini-batch列表
    
#     参数：
#         X - 输入数据，维度为(输入节点数量，样本的数量)
#         Y - 对应的是X的标签，【1 | 0】（蓝|红），维度为(1,样本的数量)
#         mini_batch_size - 每个mini-batch的样本数量
        
#     返回：
#         mini-bacthes - 一个同步列表，维度为（mini_batch_X,mini_batch_Y）
        
#     """
    
#     m = X.shape[1]                  # number of training examples
#     mini_batches = []
        
#     # Step 1: Shuffle (X, Y)
#     permutation = list(np.random.permutation(m))
#     shuffled_X = X[:, permutation]
#     shuffled_Y = Y[:, permutation].reshape((1,m))

#     """
#     #注：
#     #如果你不好理解的话请看一下下面的伪代码，看看X和Y是如何根据permutation来打乱顺序的。
#     x = np.array([[1,2,3,4,5,6,7,8,9],
#                   [9,8,7,6,5,4,3,2,1]])
#     y = np.array([[1,0,1,0,1,0,1,0,1]])

#     random_mini_batches(x,y)
#     permutation= [7, 2, 1, 4, 8, 6, 3, 0, 5]
#     shuffled_X= [[8 3 2 5 9 7 4 1 6]
#                  [2 7 8 5 1 3 6 9 4]]
#     shuffled_Y= [[0 1 0 1 1 1 0 1 0]]
#     """
    
#     # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
#     num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
#     for k in range(0, num_complete_minibatches):
#         mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
#         mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
        
#         """
#         #注：
#         #如果你不好理解的话请单独执行下面的代码，它可以帮你理解一些。
#         a = np.array([[1,2,3,4,5,6,7,8,9],
#                       [9,8,7,6,5,4,3,2,1],
#                       [1,2,3,4,5,6,7,8,9]])
#         k=1
#         mini_batch_size=3
#         print(a[:,1*3:(1+1)*3]) #从第4列到第6列
#         '''
#         [[4 5 6]
#          [6 5 4]
#          [4 5 6]]
#         '''
#         k=2
#         print(a[:,2*3:(2+1)*3]) #从第7列到第9列
#         '''
#         [[7 8 9]
#          [3 2 1]
#          [7 8 9]]
#         '''

#         #看一下每一列的数据你可能就会好理解一些
#         """

#         mini_batch = (mini_batch_X, mini_batch_Y)
#         mini_batches.append(mini_batch)

#     #如果训练集的大小刚好是mini_batch_size的整数倍，那么这里已经处理完了
#     #如果训练集的大小不是mini_batch_size的整数倍，那么最后肯定会剩下一些，我们要把它处理了
#     # Handling the end case (last mini-batch < mini_batch_size)
#     if m % mini_batch_size != 0:
#         #获取最后剩余的部分
#         mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
#         mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]

#         mini_batch = (mini_batch_X, mini_batch_Y)
#         mini_batches.append(mini_batch)

#     return mini_batches
	
class Adam():
	def initialize_adam(self,parameters) :
		"""
		初始化v和s，它们都是字典类型的变量，都包含了以下字段：
			- keys: "dW1", "db1", ..., "dWL", "dbL" 
			- values：与对应的梯度/参数相同维度的值为零的numpy矩阵
		
		参数：
			parameters - 包含了以下参数的字典变量：
				parameters["W" + str(l)] = Wl
				parameters["b" + str(l)] = bl
		返回：
			v - 包含梯度的指数加权平均值，字段如下：
				v["dW" + str(l)] = ...
				v["db" + str(l)] = ...
			s - 包含平方梯度的指数加权平均值，字段如下：
				s["dW" + str(l)] = ...
				s["db" + str(l)] = ...
		
		"""
		
		L = len(parameters) // 2 # number of layers in the neural networks
		v = {}
		s = {}
		
		# Initialize v, s. Input: "parameters". Outputs: "v, s".
		for l in range(L):
		### START CODE HERE ### (approx. 4 lines)
			v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l+1)].shape)
			v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l+1)].shape)
			s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l+1)].shape)
			s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l+1)].shape)
			### END CODE HERE ###
		
		return v, s

	def update_parameters_with_adam(parameters, grads, t, learning_rate,
                                beta1, beta2 ,  epsilon):
		"""
		使用Adam更新参数
		
		参数：
			parameters - 包含了以下字段的字典：
				parameters['W' + str(l)] = Wl
				parameters['b' + str(l)] = bl
			grads - 包含了梯度值的字典，有以下key值：
				grads['dW' + str(l)] = dWl
				grads['db' + str(l)] = dbl
			v - Adam的变量，第一个梯度的移动平均值，是一个字典类型的变量
			s - Adam的变量，平方梯度的移动平均值，是一个字典类型的变量
			t - 当前迭代的次数
			learning_rate - 学习率
			beta1 - 动量，超参数,用于第一阶段，使得曲线的Y值不从0开始（参见天气数据的那个图）
			beta2 - RMSprop的一个参数，超参数
			epsilon - 防止除零操作（分母为0）
		
		返回：
			parameters - 更新后的参数
			v - 第一个梯度的移动平均值，是一个字典类型的变量
			s - 平方梯度的移动平均值，是一个字典类型的变量
		"""
		L = len(parameters) // 2                 # number of layers in the neural networks
		v_corrected = {}                         # Initializing first moment estimate, python dictionary
		s_corrected = {}                         # Initializing second moment estimate, python dictionary
		v, s = ({'dW1': np.array([[ 0.,  0.,  0., 0., 0.,  0.,  0., 0.],[ 0.,  0.,  0., 0., 0.,  0.,  0., 0.],[ 0.,  0.,  0., 0., 0.,  0.,  0., 0.]]), 
	             'dW2': np.array([[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.]]), 
				 'db1': np.array([[ 0.],[ 0.],[ 0.]]), 
				 'db2': np.array([[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.]])}, 
				{'dW1': np.array([[ 0.,  0.,  0., 0., 0.,  0.,  0., 0.],[ 0.,  0.,  0., 0., 0.,  0.,  0., 0.],[ 0.,  0.,  0., 0., 0.,  0.,  0., 0.]]), 
     			 'dW2': np.array([[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.]]), 
				 'db1': np.array([[ 0.],[ 0.],[ 0.]]),  
				 'db2': np.array([[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.]])})
		# Perform Adam update on all parameters
		for l in range(L):
			# Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
			### START CODE HERE ### (approx. 2 lines)
			v["dW" + str(l + 1)] = beta1*v["dW" + str(l + 1)] +(1-beta1)*grads['dW' + str(l+1)]
			v["db" + str(l + 1)] = beta1*v["db" + str(l + 1)] +(1-beta1)*grads['db' + str(l+1)]
			### END CODE HERE ###

			# Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
			### START CODE HERE ### (approx. 2 lines)
			v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)]/(1-(beta1)**t)
			v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)]/(1-(beta1)**t)
			### END CODE HERE ###

			# Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
			### START CODE HERE ### (approx. 2 lines)
			s["dW" + str(l + 1)] =beta2*s["dW" + str(l + 1)] + (1-beta2)*(grads['dW' + str(l+1)]**2)
			s["db" + str(l + 1)] = beta2*s["db" + str(l + 1)] + (1-beta2)*(grads['db' + str(l+1)]**2)
			### END CODE HERE ###

			# Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
			### START CODE HERE ### (approx. 2 lines)
			s_corrected["dW" + str(l + 1)] =s["dW" + str(l + 1)]/(1-(beta2)**t)
			s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)]/(1-(beta2)**t)
			### END CODE HERE ###

			# Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
			### START CODE HERE ### (approx. 2 lines)
			parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)]-learning_rate*(v_corrected["dW" + str(l + 1)]/np.sqrt( s_corrected["dW" + str(l + 1)]+epsilon))
			parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)]-learning_rate*(v_corrected["db" + str(l + 1)]/np.sqrt( s_corrected["db" + str(l + 1)]+epsilon))
			### END CODE HERE ###
			
		return parameters

class TUNING():
	def __init__(self):
		self.x = self.y = np.eye(8)
		self.learning_rates = LEARNING_RATES
		self.decay_rates = DECAY_RATES
		self.loss_functions = LOSS_FUNCTIONS
		self.beta1_rates = BETA_1
		self.beta2_rates = BETA_2
		self.epochs = EPOCHS
		self.batch_sizes = BATCH_SIZES
		self.losses_all_lr = []
		self.losses_all_decay = []
		self.losses_all_lf = []
		self.losses_all_beta1 = []
		self.losses_all_beta2 = []
		self.losses_all_epoch = []
		self.losses_all_batch_size = []
		self.losses_all_best = []
		self.fig,self.axs= plt.subplots(3,3,figsize=(30,30))
		
	
	def lr_tuning(self):
		#学习率调参
		print("start computing loss with learning rate change")
		for learning_rate in self.learning_rates:
			losses = []
			print("start training with learning_rate = " + str(learning_rate))
			train(self.x, self.y, 10, learning_rate, 1, 'cee', 'SGD', 0.9, 0.999,losses)
			self.losses_all_lr.append(losses)
		i = 0
		print("start ploting loss with learning rate change")
		for loss in self.losses_all_lr:
			i += 1
			if (i==1):
				self.axs[0][0].plot(loss,color='r',label='learning_rate='+str(self.learning_rates[i-1]))
			elif(i==2):
				self.axs[0][0].plot(loss,color='g',linestyle='--',label='learning_rate='+str(self.learning_rates[i-1]))
			elif(i==3):
				self.axs[0][0].plot(loss,color='b',linestyle='--',marker='*',label='learning_rate='+str(self.learning_rates[i-1]))
			elif(i==4):
				self.axs[0][0].plot(loss,color='y',linestyle='-.',label='learning_rate='+str(self.learning_rates[i-1]))
			elif(i==5):
				self.axs[0][0].plot(loss,color='k',linestyle=':',label='learning_rate='+str(self.learning_rates[i-1]))						
			self.axs[0][0].legend(loc="upper right")
		# plot('F:/Python-project/Python-Basic-Practise/Lab-NN/lhr_origin/figs/l_r')
		self.axs[0][0].set_xlabel('iter')
		self.axs[0][0].set_ylabel('loss')
		self.axs[0][0].set_title('lr vs loss')

	def decay_tuning(self):
		#学习率衰减调参
		print("start computing loss with decay rate change")
		for decay_rate in self.decay_rates:
			losses = []
			print("start training with learning_rate = " + str(decay_rate))
			train(self.x, self.y, 10, 1.2, decay_rate, 'cee', 'SGD', 0.9, 0.999, losses)
			self.losses_all_decay.append(losses)
		i = 0
		print("start ploting loss with decay rate change")
		for loss in self.losses_all_decay:
			i += 1
			if (i==1):
				self.axs[0][1].plot(loss,color='r',label='decay_rate='+str(self.decay_rates[i-1]))
			elif(i==2):
				self.axs[0][1].plot(loss,color='g',linestyle='--',label='decay_rate='+str(self.decay_rates[i-1]))
			elif(i==3):
				self.axs[0][1].plot(loss,color='b',linestyle='--',marker='*',label='decay_rate='+str(self.decay_rates[i-1]))
			elif(i==4):
				self.axs[0][1].plot(loss,color='y',linestyle='-.',label='decay_rate='+str(self.decay_rates[i-1]))
			elif(i==5):
				self.axs[0][1].plot(loss,color='k',linestyle=':',label='decay_rate='+str(self.decay_rates[i-1]))						
			self.axs[0][1].legend(loc="upper right")
		# plot('F:/Python-project/Python-Basic-Practise/Lab-NN/lhr_origin/figs/d_r')
		self.axs[0][1].set_xlabel('iter')
		self.axs[0][1].set_ylabel('loss')
		self.axs[0][1].set_title('dr vs loss')

	def loss_function_tuning(self):
		#损失函数调参
		print("start computing loss with loss_function change")
		for loss_function in self.loss_functions:
			losses = []
			print("start training with loss_function = " + str(loss_function))
			train(self.x, self.y, 10, 1.2, 0.98, loss_function, 'SGD', 0.9, 0.999, losses)
			self.losses_all_lf.append(losses)
		i = 0
		print("start ploting loss with loss_function change")
		for loss in self.losses_all_lf:
			i += 1
			if (i==1):
				self.axs[0][2].plot(loss,color='r',label='loss_function='+self.loss_functions[i-1])
			elif(i==2):
				self.axs[0][2].plot(loss,color='g',linestyle='--',label='loss_function='+self.loss_functions[i-1])
			elif(i==3):
				self.axs[0][2].plot(loss,color='b',linestyle='--',marker='*',label='loss_function='+self.loss_functions[i-1])					
			self.axs[0][2].legend(loc="upper right")
		# plot('F:/Python-project/Python-Basic-Practise/Lab-NN/lhr_origin/figs/l_f')
		self.axs[0][2].set_xlabel('iter')
		self.axs[0][2].set_ylabel('loss')
		self.axs[0][2].set_title('lf vs loss')

	def beta1_tuning(self):
		#Adam优化器beta1调参
		print("start computing loss with beta1 change")
		for beta1_rate in self.beta1_rates:
			losses = []
			print("start training with beta1_rate = " + str(beta1_rate))
			train(self.x, self.y, 10, 1.2, 1, 'cee', 'ADAM', beta1_rate, 0.999,losses)
			self.losses_all_beta1.append(losses)
		i = 0
		print("start ploting loss with beta1 change")
		for loss in self.losses_all_beta1:
			i += 1
			if (i==1):
				self.axs[1][0].plot(loss,color='r',label='beta1='+str(self.beta1_rates[i-1]))
			elif(i==2):
				self.axs[1][0].plot(loss,color='g',linestyle='--',label='beta1='+str(self.beta1_rates[i-1]))
			elif(i==3):
				self.axs[1][0].plot(loss,color='b',linestyle='--',marker='*',label='beta1='+str(self.beta1_rates[i-1]))
			elif(i==4):
				self.axs[1][0].plot(loss,color='y',linestyle='-.',label='beta1='+str(self.beta1_rates[i-1]))
			elif(i==5):
				self.axs[1][0].plot(loss,color='k',linestyle=':',label='beta1='+str(self.beta1_rates[i-1]))						
			self.axs[1][0].legend(loc="upper right")
		# plot('F:/Python-project/Python-Basic-Practise/Lab-NN/lhr_origin/figs/beta1')
		self.axs[1][0].set_xlabel('iter')
		self.axs[1][0].set_ylabel('loss')
		self.axs[1][0].set_title('beta1 vs loss')

	def beta2_tuning(self):
		#Adam优化器beta2调参
		print("start computing loss with beta2 change")
		for beta2_rate in self.beta2_rates:
			losses = []
			print("start training with beta2_rate = " + str(beta2_rate))
			train(self.x, self.y, 10, 1.2, 1, 'cee', 'ADAM', 0.9, beta2_rate,losses)
			self.losses_all_beta2.append(losses)
		i = 0
		print("start ploting loss with beta2 change")
		for loss in self.losses_all_beta2:
			i += 1
			if (i==1):
				self.axs[1][1].plot(loss,color='r',label='beta2='+str(self.beta2_rates[i-1]))
			elif(i==2):
				self.axs[1][1].plot(loss,color='g',linestyle='--',label='beta2='+str(self.beta2_rates[i-1]))
			elif(i==3):
				self.axs[1][1].plot(loss,color='b',linestyle='--',marker='*',label='beta2='+str(self.beta2_rates[i-1]))
			elif(i==4):
				self.axs[1][1].plot(loss,color='y',linestyle='-.',label='beta2='+str(self.beta2_rates[i-1]))
			elif(i==5):
				self.axs[1][1].plot(loss,color='k',linestyle=':',label='beta2='+str(self.beta2_rates[i-1]))						
			self.axs[1][1].legend(loc="upper right")
		# plot('F:/Python-project/Python-Basic-Practise/Lab-NN/lhr_origin/figs/beta2')
		self.axs[1][1].set_xlabel('iter')
		self.axs[1][1].set_ylabel('loss')
		self.axs[1][1].set_title('beta2 vs loss')
	
	def epoch_tuning(self):
		#epoch调参
		print("start computing loss with epoch change")
		for epoch in self.epochs:
			losses = []
			print("start training with epoch = " + str(epoch))
			train(self.x, self.y, epoch, 1.2, 1, 'cee', 'ADAM', 0.9, 0.999,losses)
			self.losses_all_epoch.append(losses)
		i = 0
		print("start ploting loss with epoch change")
		for loss in self.losses_all_epoch:
			i += 1
			if (i==1):
				self.axs[1][2].plot(loss,color='r',label='epoch='+str(self.epochs[i-1]))
			elif(i==2):
				self.axs[1][2].plot(loss,color='g',linestyle='--',label='epoch='+str(self.epochs[i-1]))
			elif(i==3):
				self.axs[1][2].plot(loss,color='b',linestyle='--',marker='*',label='epoch='+str(self.epochs[i-1]))
			elif(i==4):
				self.axs[1][2].plot(loss,color='y',linestyle='-.',label='epoch='+str(self.epochs[i-1]))
			elif(i==5):
				self.axs[1][2].plot(loss,color='k',linestyle=':',label='epoch='+str(self.epochs[i-1]))						
			self.axs[1][2].legend(loc="upper right")
		# plot('F:/Python-project/Python-Basic-Practise/Lab-NN/lhr_origin/figs/beta2')
		self.axs[1][2].set_xlabel('iter')
		self.axs[1][2].set_ylabel('loss')
		self.axs[1][2].set_title('epoch vs loss')

	def batch_tuning(self):
		#batch_size调参
		print("start computing loss with batch_size change")
		for batch_size in self.batch_sizes:
			losses = []
			print("start training with batch_size = " + str(batch_size))
			train_with_batch_change(self.x,self.y,10000,1.2,losses,batch_size)
			self.losses_all_batch_size.append(losses)
		i = 0
		print("start ploting loss with batch_size change")
		for loss in self.losses_all_batch_size:
			i += 1
			if (i==1):
				self.axs[2][0].plot(loss,color='r',label='batch_size='+str(self.batch_sizes[i-1]))
			elif(i==2):
				self.axs[2][0].plot(loss,color='g',linestyle='--',label='batch_size='+str(self.batch_sizes[i-1]))
			elif(i==3):
				self.axs[2][0].plot(loss,color='b',linestyle='--',marker='*',label='batch_size='+str(self.batch_sizes[i-1]))
			elif(i==4):
				self.axs[2][0].plot(loss,color='y',linestyle='-.',label='batch_size='+str(self.batch_sizes[i-1]))					
			self.axs[2][0].legend(loc="upper right")
		self.axs[2][0].set_xlabel('iter')
		self.axs[2][0].set_ylabel('loss')
		self.axs[2][0].set_title('batch_size vs loss')

	def best_parameters(self):
		print("start best")
		losses = []
		parameters,losses=train(self.x, self.y, 10, 1.2, 0.98, 'cee', 'SGD', 0.9, 0.999,losses)
		self.losses_all_best.append(losses)
		for loss in self.losses_all_best:
			self.axs[2][1].plot(loss,color='r',label='best')
			self.axs[2][1].legend(loc="upper right")
		self.axs[2][1].set_xlabel('iter')
		self.axs[2][1].set_ylabel('loss')
		self.axs[2][1].set_title('best vs loss')
		self.fig.delaxes(self.axs[2,2])
		plot('F:/Python-project/Python-Basic-Practise/Lab-NN/lhr_origin/figs/all')
		print("best parameters: \n epoch = 10 \n learning_rate = 1.2\n decay_rate = 0.98 \n loss_function = 'cee' \n beta1 = 0.9 \n beta2 = 0.999")

		#测试训练效果
		test= [[1,0,1],
	 		   [1,0,0],
			   [0,0,0],
			   [0,0,0],
			   [0,0,0],
			   [0,0,0],
			   [0,0,0],
			   [0,0,0]]
		test_X = test_Y = np.array(test)
		train_test(test_X, test_Y, parameters, 10, 1.2, 1, 'cee', 'ADAM', 0.9, 0.999,losses)
		