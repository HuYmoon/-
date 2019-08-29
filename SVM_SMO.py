import numpy as np
import random
import math
# 1.定义数据集的类
class optStruct:
    def __init__(self, dataset, labels, c, toler):
        """

        :param dataset: 数据
        :param labels: 分类标签
        :param c: 惩罚系数
        :param toler:
        """
        self.dataset = dataset
        self.labels = labels
        self.c = c
        self.toler = toler
        self.b = 0
        self.m = np.shape(dataset)[0]
        alphas = np.zeros(self.m, 1)
        self.alphas = np.mat(alphas)
        ecache = np.zeros(self.m, 2)
        self.ecache = np.mat(ecache)


# 2.随意选择第二个变量，j
def selectJrand(i, m):
    j = i
    while j == i:
        i = int(random.uniform(0, m))
    return j


# 3.计算Ek
def calcEk(k, os):
    gxk = np.multiply(os.alphas, os.labels).T*(os.dataset*os.dataset[k, :].T) + os.b
    Ek = gxk - float(os.labels[k])
    return Ek


# 4.根据SMO算法选择第二个变量，j
def selectJ(i, os, Ei):
    maxk = -1
    maxdeltaE = 0
    Ej = 0
    os.ecache[i] = [1, Ei]
    validecache = np.nonzero(os.ecache[:, 1].A)[0]   # ???
    if len(validecache) > 1:
        for k in validecache:
            if k == i:
                continue
            Ek = calcEk(k, os)
            deltaE = np.abs(Ek - Ei)
            if deltaE > maxdeltaE:
                maxdeltaE = deltaE
                maxk = k
                Ej = Ek
        return maxk, Ej
    else:
        j = selectJrand(i, os.m)
        Ej = calcEk(j, os)
        return j, Ej


# 5.更新Ek
def updateEk(os, k):
    Ek = calcEk(k, os)
    os.ecache[k] = [1, Ek]


# 6.剪辑alpha
def clipalphas(alphaj, H, L):
    if alphaj > H:
        alphaj = H
    elif alphaj < L:
        alphaj = L
    else:
        alphaj = alphaj
    return alphaj


# 7.内层循环，计算b,Ei
def innerLoop(i, os):
    Ei = calcEk(i, os)
    if ((os.labels[i]*Ei < -os.toler) and (os.alphas[i] < os.c)) or ((os.labels[i]*Ei > os.toler) and (os.alphas[i] > 0)):
        j, Ej = selectJ(i, os, Ei)
        alphaIold = os.alphas[i].copy()
        alphaJold = os.alphas[j].copy()
        if os.labels[i] != os.labels[j]:
            L = np.max(0, alphaJold - alphaIold)
            H = np.min(os.c, os.c + alphaJold - alphaIold)
        else:
            L = np.max(0, alphaJold + alphaIold - oc.c)
            H = np.min(os.c, alphaIold + alphaJold)
        eta = os.dataset[i, :]*os.dataset[i, :].T + os.dataset[j, :]*os.dataset[j, :].T - 2*os.dataset[i, :]*os.dataset[j, :].T
        if eta <= 0:
            return 0

        os.alphas[j] = alphaJold + os.labels[j]*(Ei - Ej)/eta
        os.alphas[j] = clipalphas(os.alphas[j], H, L)
        updateEk(os, j)
        if np.abs(Ej - Ei) < 0.0001:
            return 0

        os.alphas[i] = alphaIold + os.labels[i]*os.labels[j]*(alphaJold, os.alphas[j])
        updateEk(os, i)
        b1new = os.b - Ei - os.labels[i]*os.dataset[i, :]*os.dataset[i, :]*(os.alphas[i] - alphaIold) - os.labels[j]*\
                os.dataset[j, :]*os.dataset[i, :]*(os.alphas[j] - alphaJold)
        b2new = os.b - Ej - os.labels[i]*os.dataset[i, :]*os.dataset[j, :]*(os.alphas[i] - alphaIold) - os.labels[j]*\
                os.dataset[j, :]*os.dataset[j, :]*(os.alphas[j] - alphaJold)
        if (0 < os.alphas[i] < os.c) and (0 < os.alphas[j] < os.c):
            bnew = b1new
        else:
            bnew = (b1new + b2new)/2.0
        return 1
    else:
        return 0


# 8.外循环
def SMO(dataset, lables, c, toler, maxiter):
    os = optStruct(dataset, lables.transpose(), c, toler)
    iter = 0
    entireset = True
    alphachanged = 0
    while (iter < maxiter) and ((alphachanged > 0) or (entireset)):
        alphachanged = 0
        if entireset:
            for i in range(os.m):
                alphachanged += innerLoop(i, os)
            iter += 1
        else:
            nonboundis = np.nonzero((os.alphas.A > 0) * (os.alphas.A < c))[0]
            for i in nonboundis:
                alphachanged += innerLoop(i, os)
            iter += 1
        if entireset:
            entireset = False
        elif (alphachanged == 0):
            entireset = True
    return os.b, os.alphas









