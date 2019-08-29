import collections
import math


def calc_entropy(dataset):
    """
    计算熵
    :param dataset: list
    :return: 熵
    """
    labelcounts = {}
    for example in dataset:
        currentlabel = example[-1]
        if currentlabel not in labelcounts.keys():
            labelcounts[currentlabel] = 1
        else:
            labelcounts[currentlabel] += 1
    entropy = 0.0
    for key in labelcounts:
        prob = float(labelcounts[key]) / (len(dataset))
        entropy -= prob * math.log(prob, 2)
    return entropy


def split_dataset(dataset, axis, value):
    """
    根据特征（axis)和值（values）获得新数据集
    :param dataset: list
    :param axis: 特征索引
    :param value: 特征值的某一个
    :return: 划分后的数据集list
    """
    new_dataset = []
    for example in dataset:
        if example[axis] == value:
            subexample = example[:axis]
            subexample.extend(example[axis+1:])
            new_dataset.append(subexample)
    return new_dataset


def feature_to_split(dataset):
    """
    根据信息增益最大的原则选择最优划分特征,ID3算法
    :param dataset: list
    :return: 最优特征索引
    """
    numfeature = len(dataset[0]) - 1
    base_entropy = calc_entropy(dataset)
    best_infogain = 0.0
    best_feature_index = -1
    for i in range(numfeature):
        feature_list = [example[i] for example in dataset]
        feature_values = set(feature_list)
        new_entropy = 0.0
        for values in feature_values:
            sub_dataset = split_dataset(dataset, i, values)
            prob = len(sub_dataset)/len(dataset)
            new_entropy += prob * calc_entropy(sub_dataset)
        infogain = base_entropy - new_entropy
        if infogain > best_infogain:
            best_infogain = infogain
            best_feature_index = i
    return best_feature_index


def create_tree(dataset, labels):
    """
    生成决策树
    :param dataset: list
    :param labels: list
    :return: 字典形式呈现的决策树模型
    """
    class_list = [example[-1] for example in dataset]
    # 判断类别标签是否一致
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 当特征集为空时，多数表决（list）
    if len(dataset[0]) == 1:
        class_y = collections.Counter(class_list)  # 重复元素计数，返回字典
        y_label = max(zip(class_y.values(), class_y.keys()))  # 返回结果为元组
        return y_label[1]

    best_feature_index = feature_to_split(dataset)
    best_feature_label = labels[best_feature_index]
    mytree = {best_feature_label: {}}
    del(labels[best_feature_index])   # 从特征标签中删除已经用于划分的标签
    feature_list = [example[best_feature_index] for example in dataset]
    unique_values = set(feature_list)
    for values in unique_values:
        sublabels = labels[:]
        mytree[best_feature_label][values] = create_tree(split_dataset(
            dataset, best_feature_index, values), sublabels)
    return mytree


def classify(tree_model, labels, testdata):
    """
    根据生成的决策树模型预测
    :param tree_model: {'有自己的房子': {'否': {'年龄': {'中年': '否', '青年': '否', '老年': '是'}}, '是': '是'}}
    :param labels: 特征标签，list  ['年龄', '有工作', '有自己的房子', '信贷情况']
    :param testdata: 预测数据，list  ['青年', '否', '否', '非常好']
    :return: 分类标签
    """
    first_str = list(tree_model.keys())[0]
    second_dict = tree_model[first_str]
    feature_index = labels.index(first_str)
    for key in second_dict.keys():
        if testdata[feature_index] == key:
            if isinstance(second_dict[key], dict):
                class_label = classify(second_dict[key], labels, testdata)
            else:
                class_label = second_dict[key]

    return class_label



