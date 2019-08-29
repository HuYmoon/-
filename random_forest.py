import decision_tree as dtree
import numpy as np
import random


def random_forest(dataset, labels, m, h):
    """
    随机森林
    :param dataset: list
    :param labels: list
    :param m: 基分类器的个数
    :param h: 随机选取的特征个数 ，一般为log2(d)+1
    :return: 基分类器结果，list
    """
    example_num = len(dataset)
    feature_num = len(labels)
    mytree_set = []
    for i in range(m):
        row_index = list(np.random.randint(0, example_num, example_num))   # 重复随机抽样
        data = [dataset[j] for j in row_index]
        col_index = random.sample(list(np.arange(0, feature_num, 1)), h)  #不重复的随机变量
        newdata = []
        for example in data:
            col_data = [example[j] for j in col_index]
            coldata = col_data.append(example[-1])
            newdata.append(col_data)
        newlabels = [labels[j] for j in col_index]
        mytree_set.append(dtree.create_tree(newdata, newlabels))
    return mytree_set


def classify(tree_model, testlabels, testdata):
    """
    预测，多数投票
    :param tree_model: 各基分类器的树结果，list
    :param testlabels: 测试数据的特征标签，list
    :param testdata:  测试数据，list
    :return: 组合分类器结果
    """
    vote = {}
    for tree in tree_model:
        # 使用异常捕捉原因：随机性导致构造的决策树可能未包含某一特征的所有值，导致最后无法预测，对于这类树，直接投0
        try:
            label = dtree.classify(tree, testlabels, testdata)
            if label not in vote.keys():
                vote[label] = 1
            else:
                vote[label] += 1
        except:
            continue
    result = max(zip(vote.values(), vote.keys()))[1]
    return result

def main():
    data = [['青年', '否', '否', '一般', '否'],
            ['青年', '否', '否', '好', '否'],
            ['青年', '是', '否', '好', '是'],
            ['青年', '是', '是', '一般', '是'],
            ['青年', '否', '否', '一般', '否'],
            ['中年', '否', '否', '一般', '否'],
            ['中年', '否', '否', '好', '否'],
            ['中年', '是', '是', '好', '是'],
            ['中年', '否', '是', '非常好', '是'],
            ['中年', '否', '是', '非常好', '是'],
            ['老年', '否', '是', '非常好', '是'],
            ['老年', '否', '是', '好', '是'],
            ['老年', '是', '否', '好', '是'],
            ['老年', '是', '否', '非常好', '是'],
            ['老年', '否', '否', '一般', '否'],]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    vote = {}
    for i in range(10):
        mytree = random_forest(data, labels, 10, 3)
        testdata = ['青年', '否', '否', '非常好']
        testlabel = ['年龄', '有工作', '有自己的房子', '信贷情况']   # 在生成决策树模型的时候labels有所改动
        result = classify(mytree, testlabel, testdata)
        if result not in vote.keys():
            vote[result] = 1
        else:
            vote[result] += 1
    result_finally = max(zip(vote.values(), vote.keys()))[1]
    print(result_finally)


if __name__ == '__main__':
    main()