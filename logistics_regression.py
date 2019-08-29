import numpy as np
import random

# 1. S曲线函数
def sigmoid(x):
    return 1/(1 + np.exp(x))


# 2. 随机梯度上升算法
def stocgradAscent(dataset, labels, iter):
    """
    改进的随机梯度上升法
    :param dataset: 矩阵
    :param labels: 矩阵
    :param iter: int
    :return: 权重
    """
    m, n = np.shape(dataset)
    weights = np.ones(n)
    for i in range(iter):
        dataindex = range(m)
        for j in range(m):
            index = random.uniform(0, len(dataindex))
            alphal = 4/(1.0 + i + j) + 0.01
            h = sigmoid(dataset[index]*weights)
            error = labels[index] - h
            weights = weights + alphal * error * dataset[index]
            del(dataindex[index])
    return weights


# 3. 分类
def classify(x, weights):
    """

    :param x: 矩阵
    :param weights: 矩阵
    :return:
    """
    p = sigmoid((sum(x*weights)))
    if p > 0.5:
        return 1.0
    else:
        return 0.0


# 4. 训练集数据读取
def data(filename):
    dataset = []
    labels = []
    fh = open(filename)
    for line in fh.readlines():
        linearr = line.strip().split('\t')
        datalist = [1.0]
        for i in range(len(linearr)-1):
            datalist.append(linearr[i])
        dataset.append(list)
        labels.append(int(linearr[-1]))
    return np.mat(dataset), np.mat(labels).transpose()









