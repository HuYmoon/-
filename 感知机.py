import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D


def train(t_data):
    # 1.处理输入的数据行例情况,方便后面随机梯度下降法的使用
    row = np.shape(t_data)[0]
    col = np.shape(t_data)[1]

    # 2.初始化w,b,n
    global w, b
    w = np.zeros((1, col-1))
    b = 0
    n = 1

    # 3.随机梯度下降法
    print("------------------------------")
    print("1.原始形式   2.对偶形式")
    choose = int(input("请输入你想执行的算法编号:"))

    #   原始形式
    if choose == 1:
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for i in range(row):
                xi = t_data.iloc[i, :col-1]
                yi = t_data.iloc[i, col-1]
                v = yi * (np.dot(w, xi) + b)
                if v <= 0:
                    w = w + n * (np.dot(yi, xi))
                    b = b + n * yi
                    wrong_count = +1
            if wrong_count == 0:
                is_wrong = True

    #   对偶形式
    elif choose == 2:
        # aerfa = DataFrame(np.zeros((1, row)))  # 初始化α
        # print(aerfa)
        aerfa = [0 for i in range(row)]   # 初始化α，用列表的形式
        t_data_array = np.array(t_data)   # 将数据框转化为数组
        # gram = np.dot(t_data[:, 0:-1], t_data[:, 0:-1])
        # print(gram)
        # multiple是对应元素相乘，dot和matmul是矩阵乘法，？？?为什么要转成数组才能用，直接dot报错
        Gram = np.matmul(t_data_array[:, 0:-1], t_data_array[:, 0:-1].T)
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for i in range(row):
                temp = 0
                xi = t_data.iloc[i, :col-1]
                yi = t_data.iloc[i, col-1]
                for j in range(row):
                    temp += aerfa[j] * t_data_array[j, -1] * Gram[j, i]
                temp += b
                if yi * temp <= 0:
                    aerfa[i] = aerfa[i] + 1
                    b = b + yi
                    wrong_count = +1
            if wrong_count == 0:
                is_wrong = True
    else:
        print("请正确输入")

    # 4.输出感知机模型(使用公式展示）
    # print(w, b)   输出的w是[[2. 2.]],列表里面的列表
    print("感知机模型为   ", end="")
    w_array = w[0]
    w_list = list(w_array)   # w[0]是数组形式，无法用len函数，需要先转换为列表
    # 遍历每一个系数，负数和正数的连接方式，最后一个系数的连接方式
    for i in range(len(w_list)):
        if i < len(w_list)-1:
            if w_list[i] >= 0:
                wx = str(w_list[i]) + "*" + "x" + str(i + 1) + str("+")
                print(wx, end="")
            else:
                 wx = str("(") + str(w_list[i]) + str(")") + "*" + "x" + str(i + 1) + str("+")
                 print(wx, end="")
        else:
            if w_list[i] >= 0:
                wx = str(w_list[i]) + "*" + "x" + str(i + 1)
                print(wx, end="")
            else:
                wx = str("(") + str(w_list[i]) + str(")") + "*" + "x" + str(i + 1)
            print(wx, end="")
    print("=0")


def image(t_data):
    # 将正类和负类区分
    row = np.shape(t_data)[0]
    col = np.shape(t_data)[1]
    objdata = t_data[t_data["y"] > 0]
    subjdata = t_data[t_data["y"] < 0]

    # 画图
    if np.shape(t_data)[1] - 1 == 2:
        plt.scatter(objdata["x1"], objdata["x2"], label="1")
        plt.scatter(subjdata["x1"], subjdata["x2"], label="-1")
        plt.xlabel("x1")
        plt.xlabel("x2")
        x = np.linspace(0, 5, 10)   # 视数据情况而定
        plt.plot(x, -(b + w[0][0] * x)/w[0][1])
        plt.title("二维散点图")
        plt.show()
    elif np.shape(t_data)[1] - 1 == 3:
        # 二维
        fig = plt.figure()
        # 二位转化为三维
        ax = Axes3D(fig)
        xs = t_data["x1"]
        ys = t_data["x2"]
        zs = t_data["x3"]
        ax.scatter(xs, ys, zs, c="r", cmap="rainbow")
        plt.xlabel("x1")
        plt.ylabel("x2")
        ax.set_zlabel("x3")
        plt.show()
    else:
        print("变量x超过二维，无法显示二维散点图")


def test(test_data):
    count = 0
    row = np.shape(test_data)[0]
    col = np.shape(test_data)[1]
    for j in range(row):
        ty = np.dot(w, test_data.iloc[j, :col-1]) + b
        if ty * test_data.iloc[j, col-1] < 0:
            count = +1
    accruRate = 1 - (count / np.shape(test_data)[0])
    return accruRate


def predict(tsdata):
    ts = np.dot(w, tsdata) + b
    if ts > 0:
        print("输入的数据为正类")
    else:
        print("输入的数据为负类")


def main():
    # 读入excel数据
    data = pd.ExcelFile("感知机.xlsx")

    # 输入测试集数据
    table = data.parse("Sheet1")
    t_data = DataFrame(table)
    print(t_data)

    # 调用方法
    train(t_data)
    image(t_data)

    # 输入测试集数据
    table1 = data.parse("Sheet2")
    tsdata = DataFrame(table1)
    acc = test(tsdata)   # 正确率
    print("模型准确率为 %.02f" % acc)

    # 输入预测数据
    pdata = DataFrame([3.4, 4, 3.2, 5, 2.6])
    predict(pdata)


if __name__ == "__main__":
    main()






