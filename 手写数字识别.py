from os import listdir
import numpy as np
import knn


def imgvector(filename):
    """
    将每个TXT文档内容转化成一位数组
    :param filename: 文件名
    :return: 一位数组
    """
    fh = open(filename)
    data = np.zeros((1, 1024))
    for i in range(32):
        linestr = fh.readline()
        for j in range(32):
            data[0, 32*i+j] = int(linestr[j])
    return data


def handwritting():
    """
    读取文件夹下每个文件，准备数据集和标签
    :return:
    """
    train_filelist = listdir('trainingDigits')
    train_labels = []
    m = len(train_filelist)
    traindata = np.zeros((m, 1024))
    for i in range(m):
        filename_str = train_filelist[i]
        filename = filename_str.split('.')[0]
        label = filename.split('_')[0]
        train_labels.append(label)
        traindata[i, :] = imgvector('trainingDigits/%s' % filename_str)
    return traindata, train_labels


def error_calu(traindata, train_labels):
    """
    错误率计算
    :param traindata:
    :param train_labels:
    :return:
    """
    test_filelist = listdir('testDigits')
    error = 0
    n = len(test_filelist)
    for i in range(n):
        filename_str = test_filelist[i]
        filename = filename_str.split('.')[0]
        y_label = filename.split('_')[0]
        testdata = imgvector('testDigits/%s' % filename_str)
        result = knn.classfiy(traindata, testdata, train_labels, 50)
        if result != y_label:
            error += 1
        else:
            error = 0
    error_rate = error/int(n)
    print('错误率为%s' % error_rate)


def test(traindata, train_label, predictdata):
    """
    预测
    :param traindata:
    :param train_label:
    :param predictdata:
    :return:
    """
    predictdata = imgvector('')
    result = knn.classfiy(traindata, predictdata, train_label, 10)






