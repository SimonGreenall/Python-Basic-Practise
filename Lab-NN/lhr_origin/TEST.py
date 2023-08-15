import numpy as np
import model

def predict(parameters, X):
    # 使用前向传播计算概率，并使用 0.5 作为阈值将其分类为 0/1。
    A2, cache = model.NeutralNetwork().forward_propagation(X, parameters)
    predictions = np.round(A2)

    return predictions