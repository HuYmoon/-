import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pandas import DataFrame
import statsmodels.api as sm


def factor_corr(data):
    """
    计算估值因子与市值因子之间的相关系数
    :param data:时间(第一列）+估值因子+市值因子（最后一列）
    :return:
    corr:相关系数
    """
    # 1.创建一个新的和corr函数输出结果相同的数据框，方便pd.merge
    n = data.iloc[:, 0].unique()   # 提取出所有的时间取值(形式为2018)
    m = np.min(np.array(n))   # 提取最小时间
    arr = ["因子"]   # 输出结果的列
    corrnew = DataFrame(np.zeros((data.shape[1]-2, 2)), columns=list("ab"))
    for i in range(len(n)):
        data_time = data[data.iloc[:, 0].isin([m + i])]
        corr = data_time.iloc[:, 1:-1].corrwith(data_time.iloc[:, -1])   # 每一列和最后一列求corr
        corr = DataFrame(corr).reset_index()    # 索引变成列,改列名
        corr = corr.rename(columns={"index": "因子"})
        if i == 0:
            corrnew = pd.merge(corrnew, corr, left_index=True, right_index=True, how="outer")   #数据框直接合并
        else:
            corrnew = pd.merge(corrnew, corr.iloc[:, -1], left_index=True, right_index=True)   #数据框直接合并
        arr.append(str(i+m))

    # 2.删除最开始创建的数据框
    corr = corrnew.iloc[:, 2:]
    # 3.求行均值，并合并
    mdata = DataFrame(corr.iloc[:, 1:])
    mean = mdata.mean(axis=1)
    mean = DataFrame(mean)
    corr = pd.merge(corr, mean, left_index=True, right_index=True, how="outer")   #数据框直接合并
    # 4.修改列名
    arr.append("均值")
    corr.columns = arr
    # 5.打印数据
    return corr

def factor_C(data):
    """
    计算各估值因子的相关强度,月度相关系数均值和标准差
    :param data:36个月月末截面所有个股的数据汇总,假设列为:时间+股票型号+因子
    :return:None
    Cab: 合估值因子间的相关强度
    mean: 各估值因子间的月度相关系数序列均值
    std: 各估值因子间的月度系数序列标准差
    """
    # 1.月度相关系数及其均值，标准差(36为月份）
    # 数据框相加必须列名(columns)、行名(index)相同
    fnum = len(data.shape[1]-2)   # 因子数量
    mnum = data.iloc[:, 1].unique()   # 月份数量
    name = data.columns.values[1:]
    corr = DataFrame(np.zeros((fnum, fnum)), columns=name, index=name)
    sgmx2 = DataFrame(np.zeros((fnum, fnum)), columns=name, index=name)
    for i in range(len(mnum)):
        # 挑选出同一个月底的所有股票数据
        datacorr = DataFrame(data[data.iloc[:, 0].isin([mnum[i]])].iloc[:, 2:])
        corr0 = datacorr.corr()
        corr = corr0 + corr
        # 求相关系数的标准差（sigema的x平方-n*x均值的平方)
        sgmx2 += corr0.mul(corr0)
    mean = corr / mnum
    cov = sgmx2 - mnum*mean.mul(mean)   # 数据框对应位置相乘
    arrcov = np.array(cov)   # pandas里面没有找到开方，转成数组开方
    std = DataFrame(np.sqrt(arrcov), columns=name, index=name)
    arrcorr = np.array(corr)
    # 2.取数据框的列名，返回结果是list
    Cab = DataFrame(np.divide(arrcorr, arrcov), columns=name, index=name)
    return Cab, mean, std
    # print("相关强度", Cab)
    # print("月度相关系数均值", mean)
    # print("月度相关系数标准差", std)


def signal_factor_test(data, current_price, market_values):
    """
    单因子测试：回归、IC
    :param data:
    data: 时间+行业(列名industry)+因子+股票收益率(y)
    current_price： 个股流通市值,当作wls的权重
    market_values: 市值因子,IC回归的变量之一
    :return:
    table_reg: 估值因子回归测试结果
    table_ic： 估值因子IC值分析
    """
    # 1.数据处理
    industy = pd.get_dummies(data.iloc[:, 1])   # 类别特征抽取
    y = DataFrame(data.iloc[:, -1])  # 提取回归模型中的Y,先添加类别特征数据框，再加回去（让因变量在最后一列）
    yname = y.columns.values
    newdata = data.drop(["industry", yname[0]], axis=1)   # 删除行业列和Y
    newdata = pd.merge(newdata, industy, left_index=True,
                       right_index=True, how="outer")  # 数据合并(时间+因子+行业虚拟变量)
    # wls回归的数据整理
    newdata_reg = pd.merge(newdata, y, left_index=True,
                           right_index=True, how="outer")   # 数据合并(时间+因子+行业虚拟变量+y)
    newdata_wls = pd.merge(newdata_reg, current_price, left_index=True,
                           right_index=True, how="outer")   # 数据合并(时间+因子+行业虚拟变量+y+权重股票流通价值)
    # IC回归的数据整理
    newdata_mfv = pd.merge(newdata, market_values, left_index=True,
                           right_index=True, how="outer")  # 数据合并(时间+因子+行业虚拟变量+市值因子)
    newdata_ic = pd.merge(newdata_mfv, y, left_index=True,
                          right_index=True, how="outer")   # 数据合并(时间+因子+行业虚拟变量+市值因子+y)

    # 2.拟合回归方程,提取结果,生成表格
    time = data.iloc[:, 0].unique()   # 提取数据中所有的时间类型
    fnum = data.shape[1]-3   # 因子数量
    table_reg = DataFrame(np.zeros((1, 7)), columns=["因子", "|t|均值", "|t|>2占比", "t均值", "t均值/t标准差",
                                                     "因子收益率均值", "因子收益率序列t检验"])
    table_ic = DataFrame(np.zeros((1, 6)), columns=["因子", "IC序列均值", "IC序列标准差",
                                                    "IR比率", "IC>0占比", "|IC|>0.02占比"])
    fig, ax = plt.subplots(1, 2)
    for i in range(fnum):
        # 每次因子循环列表清空
        tlist = []   # t值列表
        rlist = []   # 因子收益率列表
        iclist = []   # ic值列表
        for j in range(len(time)):

            # wls,创建属于某一个因子的其中一个一个截面的所有数据集(股票+因子+虚拟变量+y+股票流通价值)
            newdata2 = newdata_wls[newdata_wls.iloc[:, 0].isin([time[j]])]
            # IC的ols,创建属于某一个因子的其中一个截面的所有数据集(股票+因子+虚拟变量+市值因子+y)
            newdata3 = newdata_ic[newdata_ic.iloc[:, 0].isin([time[j]])]
            # 创建回归方程中的自变量名字列表
            col = list(industy.columns.values)
            factor_data = DataFrame(newdata2.iloc[:, i + 1])
            global factor_name   # 全局变量
            factor_name = list(factor_data.columns.values)[0]
            col.append(factor_name)  # 单因子自变量的名字列表,list.append没有返回值,直接修改col
            # wls回归
            y = newdata2.iloc[:, -2]
            x = sm.add_constant(newdata2.loc[:, col])   # 增加常数项
            reg = sm.WLS(y, x, weights=newdata2.iloc[:, -1])   # loc是列名索引,exog增加截距
            model = reg.fit()
            # IC的ols回归
            col.append(list(DataFrame(market_values).columns.values)[0])
            x_ic = sm.add_constant(newdata3.loc[:, col])
            reg_ic = sm.WLS(y, x_ic)
            model_ic = reg_ic.fit()
            # 提取wls回归模型结果
            tvalues = DataFrame(model.tvalues).iloc[-1, :]
            weight = DataFrame(model.params)
            tlist.append(DataFrame(model.tvalues).iloc[-1, :])
            rlist.append(weight.iloc[-1, :])
            # 提取IC的ols回归模型结果
            iclist.append(np.sqrt(1-model_ic.rsquared))
        # wls结果合并
        tarr = np.array(tlist)
        rarr = np.array(rlist)
        table = {"因子": factor_name,
                 "|t|均值": np.mean(np.abs(tarr)),
                 "|t|>2占比": list(np.where(np.abs(tarr)>2, 1, 0)).count(1)/len(time),
                 "t均值": np.mean(tarr),
                 "t均值/t标准差": np.mean(tarr)/np.std(tarr),
                 "因子收益率均值": np.mean(rarr),
                 "因子收益率序列t检验": np.std(rarr)}
        table_reg0 = DataFrame(table, columns=["因子", "|t|均值", "|t|>2占比", "t均值", "t均值/t标准差",
                                               "因子收益率均值", "因子收益率序列t检验"], index=[i+1])


        ax[0].plot(time, rarr.cumsum(), label=factor_name)

        # IC结果合并
        table_reg = pd.concat([table_reg, table_reg0])   # 纵向合并
        icarr = np.array(iclist)
        table1 = {"因子": factor_name,
                  "IC序列均值": np.mean(icarr),
                  "IC序列标准差": np.std(icarr),
                  "IR比率": np.mean(icarr)/np.std(icarr),
                  "IC>0占比": list(np.where(icarr>0, 1, 0)).count(1)/len(icarr),
                  "|IC|>0.02占比": list(np.where(np.abs(icarr)>0.02, 1, 0)).count(1)/len(icarr)}

        table_ic0 = DataFrame(table1, columns=["因子", "IC序列均值", "IC序列标准差", "IR比率",
                                               "IC>0占比", "|IC|>0.02占比"], index=[i+1])
        table_ic = pd.concat([table_ic, table_ic0])   # 纵向合并
        ax[1].plot(time, icarr.cumsum(), "b-")
    table_reg = table_reg.iloc[1:, :].set_index("因子")   # set_index 列变索引
    table_ic = table_ic.iloc[1:,:].set_index("因子")
    plt.legend(loc="upper center")
    plt.show()
    return table_reg, table_ic


def main():
    pd.set_option("display.max_columns", 10)
    data = pd.read_csv("onefactor.csv")
    # print(type(data))
    # factor_corr(data)
    market = DataFrame(data.iloc[:, 0])
    market = market.rename(columns={"time":"市值因子"})
    t = signal_factor_test(data, market, market)   # 接收返回结果
    print(t[0])
    print(t[1])


if __name__ == '__main__':
    main()