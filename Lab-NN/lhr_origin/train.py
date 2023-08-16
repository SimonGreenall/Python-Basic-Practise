import utils
import numpy as np
import pandas as pd
import model
from TEST import predict

def train(X, Y, epoch, learning_rate, decay_rate, loss_function, OPTIMIZER, beta1, beta2, losses):
    parameters = model.NeutralNetwork().initialize_parameters()
    epoches=[]
    accuracies=[] #存放每个epoch后预测的准确率
    recalls=[] #存放每个epoch后预测的召回率
    precisiones=[] #存放每个epoch后预测的精确度
    F1_S=[] #存放每个epoch后预测的F1

    #开始循环（梯度下降） 无正则
    for i in range(0, epoch):
        for j in range(0, 1000):  # 每个epoch训练一千次,用_可以避免开辟新变量
            #前向传播
            A2, cache = model.NeutralNetwork().forward_propagation(X, parameters)
            #计算成本
            cost = model.NeutralNetwork().compute_cost(A2, Y, loss_function)
            #反向传播
            grads = model.NeutralNetwork().backward_propagation(X, Y, cache, parameters)
            #更新参数
            if(OPTIMIZER == 'SGD'):
                decay_learning_rate = learning_rate * np.power(decay_rate, (i*1000+j)/1000)  # 实现学习率衰减
                parameters = model.NeutralNetwork().update_parameters(parameters, grads, decay_learning_rate)
            elif(OPTIMIZER == 'ADAM'):
                parameters = utils.Adam.update_parameters_with_adam(parameters,grads,i*1000+j+1,learning_rate,beta1,beta2,1e-8)

            losses.append(cost)


        print("Cost after epoch %i: %f" % (i, cost))
        predicted = np.array(predict(parameters, X)).flatten().astype(int) #将预测结果变为一维数组并且将其中元素转换为int作为计算指标的输入
        expected = np.array(Y).flatten().astype(int) #将标签变为一维数组并且将其中元素转换为int作为计算指标的输入
        accuracy,recall,precision,F1 = utils.compute_indexes(predicted,expected)
        accuracies.append(accuracy)
        recalls.append(recall)
        precisiones.append(precision)
        F1_S.append(F1)
        epoches.append(i)

        # print("accuracy, precision, recall, F1 after epoch %i: %f %f %f %f" % (i, accuracy, precision, recall, F1))
    data={'epoch': epoches,
          'accuracy': accuracies,
          'recall': recalls,
          'precision': precisiones,
          'F1': F1_S}
    df = pd.DataFrame(data)
    print(df)
    return parameters, losses

def train_with_batch_change(X,Y,epoch,learning_rate, losses,batch_size):
    parameters = model.NeutralNetwork().initialize_parameters()
    m = X.shape[1] # 样本数量

    #开始循环(梯度下降) 无正则
    for i in range(epoch):
        #将数据随机打乱
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]

        #分成多个小批量进行训练
        num_batches = m // batch_size
        for j in range(num_batches):
            #获取当前小批量的数据
            start = j * batch_size
            end = (j + 1) * batch_size
            mini_batch_X = shuffled_X[:,start:end]
            mini_batch_Y = shuffled_Y[:,start:end]
            #前向传播
            A2, cache = model.NeutralNetwork().forward_propagation(mini_batch_X, parameters)
            #计算成本
            cost = model.NeutralNetwork().compute_cost(A2, mini_batch_Y, 'cee')
            #反向传播
            grads = model.NeutralNetwork().backward_propagation(mini_batch_X, mini_batch_Y, cache, parameters) 
            #更新参数
            parameters = model.NeutralNetwork().update_parameters(parameters, grads, learning_rate)
            losses.append(cost)

            print("Cost after epoch %i: %f" % (i, cost))
    return parameters,losses