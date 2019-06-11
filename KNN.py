import numpy as np
import pandas as pd
from pandas import DataFrame


def train(tdata, pdata):
    # 1.确定k
    k = int(input("请输入KNN中的k的取值："))
    # 2.确定距离
    # 2.1 确定p值，度量距离的方法
    print("-" * 20)
    print("【1】.曼哈顿距离  【2】.欧式距离  【3】.切比雪夫距离")
    p = int(input("请输入你想选择的度量距离的方法；"))
    lp = 0
    di = []   # 创建训练集和实例之间的距离的列表
    y = []
    # 2.2 根据距离找出最邻近的k个点
    print(type(tdata.iloc[1, 2]))   # 切片出来是一个数组
    if p == 1 or p == 2:
        for i in range(np.shape(tdata)[0]):
            for j in range(np.shape(tdata)[1]-1):
                lp += ((np.abs(tdata.iloc[i, j] - (pdata.iloc[j]))) ** p) ** 1/p
            di.append(lp)
            y.append(tdata.iloc[i, -1])
    else:
         for i in range(np.shape(tdata)[0]):
            for j in range(np.shape(tdata)[1]-1):
                 lp += np.max(np.abs(tdata.iloc[i, j] - (pdata.iloc[j])))
            di.append(lp)
            y.append(tdata.iloc[i, -1])

    # 3.分类决策规则
    y = DataFrame(y)
    index = [i for i in range(np.shape(tdata)[0])]
    di = DataFrame(di, index=index)
    diy = pd.merge(di, y, left_index=True, right_index=True, how="outer")   # 将距离和y合并
    print("-"*20)
    print("所以的距离值以及对应的分类标号")
    print(diy)
    dik = diy.sort_index(by="0_x")[0: k]   # 数据框根据某一列排序
    print("-"*20)
    print("前k个距离值以及对应的分类标号")
    print(dik)
    y_data = dik.iloc[:, 1].value_counts(ascending=False)  # 数据框某一列值的统计，返回一个数组
    print("-"*20)
    print(y_data)
    print("输入的数据属于：", end="")
    print(y_data.idxmax())


def main():
    data = pd.ExcelFile("感知机.xlsx")
    tdata = data.parse("Sheet1")
    pdata = DataFrame([3.4, 4, 3.2, 5, 2.6])
    train(tdata, pdata)


if __name__ == "__main__":
    main()
