import collections
from math import log
import numpy as np

def calc_probabilities(data):
    """
    计算几个特征下不同分类标签的概率（条件概率）
    :param data:
    :return: 字典
    """
    data_list = []
    num_features = len(data[0])
    for i in range(num_features):
        data_list.extend([data[i] for data in data])
    data_dic = collections.Counter(data_list)
    for key in data_dic:
        data_dic[key] = float(data_dic[key]) / len(data_list)
    return data_dic


def classify(dataset, labels, testdata):
    """
    预测分类
    :param dataset: list，前提是每个特征的分类标签不能相同
    :param labels: list
    :param testdata: list
    :return: 分类情况
    """
    num_eaxmple = len(dataset)
    data0 = []
    data1 = []
    for i in range(num_eaxmple):
        if labels[i] == -1:
            data0.append(dataset[i])
        else:
            data1.append(dataset[i])
    # 不同类别下的数据集对应的条件概率
    data0_dic = calc_probabilities(data0)
    data1_dic = calc_probabilities(data1)
    # 不同类别的先验概率
    py0 = collections.Counter(labels)[-1] / float(num_eaxmple)
    py1 = collections.Counter(labels)[1] / float(num_eaxmple)
    # 计算预测数据的各类别的概率
    p_class0 = 0.0
    p_class1 = 0.0
    # 避免下溢出，采取对数函数求概率
    for test in testdata:
        p_class0 += log(data0_dic[test])
    p_class0 = p_class0 * log(py0)
    for test in testdata:
        p_class1 += log(data1_dic[test])
    p_class1 = p_class1 * log(py1)
    if p_class0 > p_class1:
        return -1
    else:
        return 1


dataset = [[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'],
           [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'],
           [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L'], ]
labels = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
testdata = [2, 'S']
result = classify(dataset, labels, testdata)
print(result)
