import numpy as np
import collections

def classfiy(traindata, testdata, label, k):
    """
    knn算法
    :param traindata: array，训练集
    :param testdata: array，测试数据，一行多列
    :param label: list，类别
    :param k: 参数
    :return: 测试数据的分类类别
    """
    datasetsize = traindata.shape[0]
    Y = np.tile(testdata, (datasetsize, 1))   # 准备数据格式，方便计算距离
    diss = ((traindata - Y) ** 2).sum(axis=1) * 0.5   # 计算距离
    sortindex = np.argsort(diss)   # 距离排序，返回索引
    class_label = []   # k个分类标签列表
    # 多数表决
    for i in range(k):
        votelabel = label[sortindex[i]]
        class_label.append(votelabel)
    class_y = collections.Counter(class_label)   # 重复元素计数，返回字典
    y_label = max(zip(class_y.values(), class_y.keys()))   #返回结果为元组
    return y_label[1]


def error(data, label, k):
    """
    错误率计算,整个数据集按照7:3的比率划分训练集和验证集
    :param data: array,整个数据
    :param label: list, 类别
    :param k: 参数
    :return: 错误率，直接print
    """
    num = int(0.7*data.shape[0])
    traindata = data[:num]
    testdata = data[num:]
    train_label = label[:num]
    test_label = label[num:]
    errorcount = 0
    for i in range(testdata.shape[0]):
        testlabel = classfiy(traindata, testdata[i], train_label, k)
        if testlabel != test_label[i]:
            errorcount += 1
    print(errorcount)
    error_rate = errorcount/testdata.shape[0]
    print(error_rate)
    print('错误率为%s' % error_rate)


def text_parse(filename,n):
    """
    TXT文本解析，类似Excel表格数据
    :param filename:
    :param n: 特征数
    :return: 数据，类别
    """
    fr = open(filename)
    numberofline = len(fr.readlines())
    traindata = np.zeros((numberofline, n))
    label = []
    index = 0
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip()
        list = line.split("\t")
        traindata[index, :] = list[:n]
        label.append(list[-1])
        index += 1
    return traindata, label


def normalize(data):
    """
    标准化
    :param data:
    :return:
    """
    min_data = np.tile(data.min(0), (data.shape[0], 1))
    max_data = np.tile(data.max(0), (data.shape[0], 1))
    data = (data - min_data)/(max_data - min_data)
    return data



# def main():
#     traindata = np.array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])
#     testdata = np.array([1, 1])
#     label = ['a','a','b','b']
#     print("未知数据分类为%s" % classfiy(traindata, testdata,label, 3))


