# encoding: utf-8
import matplotlib.pyplot as plt
from sklearn.datasets import *
import random
import pandas as pd
import os
import math
import numpy
from datetime import datetime
import csv

"""
任务：在[分布,面积,形状,位置,]这个高维空间中采样
生成数据，旋转、平移另外考虑，得到含有两个聚类的数据(中心可以重叠)
分布：高斯，随机，均匀，(柯西分布，学生分布，F分布)
形状：线段，圆形，S形，螺旋形，圆形，正方形，五边形，六边形，弓形，扇形，螺旋形，通过包围盒控制在一定的范围内，形状要自然
位置：中心点，方向
面积：需要保证密度不变，对于高斯分布需要改变bandwidth
密度：平均密度=外围边框积分/面积，
距离：通过控制某一个聚类平移一段距离实现。
"""


def savetocsv(pxlist, pylist, lab, name):
    sample = []
    le = len(pxlist)
    for i in range(le):
        sample.append([pxlist[i], pylist[i], lab[i]])
    test = pd.DataFrame(columns=['X', 'Y', 'label'], data=sample)
    path = os.path.join('C:/Users/liblnuex/Desktop/VCS/generatedata/', name + '.csv')
    # print(path)
    test.to_csv(path, encoding='utf-8')


def squaregen(tot):
    cen1 = [[-4, -2], [-4, -1], [-4, 0], [-4, 1], [-4, 2], [-4, 3], [-4, 4],
            [-3, 4], [-2, 4], [-1, 4], [0, 4], [1, 4], [2, 4],
            [2, 3], [2, 2], [2, 1], [2, 0], [2, -1], [2, -2]]

    cen2 = [[-2, 2], [-2, 1], [-2, 0], [-2, -1], [-2, -2], [-2, -3], [-2, -4],
            [-1, -4], [0, -4], [1, -4], [2, -4], [3, -4], [4, -4],
            [4, -3], [4, -2], [4, -1], [4, 0], [4, 1], [4, 2]]

    l1 = int(len(cen1))
    #print(l1)
    delta = random.uniform(0.1, 0.5)
    for i in range(int(l1)):
        #print(cen1[i][1])
        cen1[i][1] += delta

    t_cluster_std = random.uniform(0.2, 0.6)
    t_n_samples = random.randint(100, 500)
    #print(t_cluster_std)
    #print(t_n_samples)
    x1, y1 = make_blobs(n_samples=t_n_samples, n_features=2, cluster_std=t_cluster_std, centers=cen1)
    x2, y2 = make_blobs(n_samples=t_n_samples, n_features=2, cluster_std=t_cluster_std, centers=cen2)
    pxlist = x1[:, 0].tolist() + x2[:, 0].tolist()
    pylist = x1[:, 1].tolist() + x2[:, 1].tolist()
    t1 = [0 for x in range(t_n_samples)]
    t2 = [1 for x in range(t_n_samples)]
    t = t1 + t2
    sita = random.uniform(0, math.pi / 3)
    tpxlist = pxlist
    tpylist = pylist
    le = len(pxlist)
    op = random.choice([1, 2, 3, 4, 5])
    for j in range(le):
        pxlist[j] = math.cos(sita) * tpxlist[j] + math.sin(sita) * tpylist[j]
        pylist[j] = - math.sin(sita) * tpxlist[j] + math.cos(sita) * tpylist[j]
        if op == 3:
            pylist[j] = -pylist[j]
    plt.scatter(pxlist, pylist, c=t, alpha=0.5)
    plt.show()
    # savetocsv(pxlist, pylist, t, str(tot) + "square")


def trianglegen(tot):
    cen1 = [[-1, 4], [-2, 3], [-3, 2], [-4, 1], [-5, 0], [-6, -1],
            [-5, -1], [-4, -1], [-3, -1], [-2, -1], [-1, -1], [0, -1]]
    cen2 = [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1],
            [5, 0], [4, -1], [3, -2], [2, -3], [1, -4]]

    l1 = int(len(cen1))
    # print(l1)
    delta = random.uniform(0.1, 0.5)
    for i in range(int(l1)):
        # print(cen1[i][1])
        cen1[i][0] -= delta

    t_cluster_std = random.uniform(0.2, 0.6)
    t_n_samples = random.randint(100, 500)
    # print(t_cluster_std)
    x1, y1 = make_blobs(n_samples=t_n_samples, n_features=2, cluster_std=t_cluster_std, centers=cen1)
    x2, y2 = make_blobs(n_samples=t_n_samples, n_features=2, cluster_std=t_cluster_std, centers=cen2)
    pxlist = x1[:, 0].tolist() + x2[:, 0].tolist()
    pylist = x1[:, 1].tolist() + x2[:, 1].tolist()
    t1 = [0 for x in range(t_n_samples)]
    t2 = [1 for x in range(t_n_samples)]
    t = t1 + t2
    sita = random.uniform(0, math.pi/3)
    tpxlist = pxlist
    tpylist = pylist
    le = len(pxlist)
    op = random.choice([1, 2, 3, 4, 5])
    for j in range(le):
        pxlist[j] = math.cos(sita) * tpxlist[j] + math.sin(sita) * tpylist[j]
        pylist[j] = - math.sin(sita) * tpxlist[j] + math.cos(sita) * tpylist[j]
        if op == 3:
            pylist[j] = -pylist[j]
    plt.scatter(pxlist, pylist, c=t, alpha=0.5)
    plt.show()
    # savetocsv(pxlist, pylist, t, str(tot) + "triangle")


def sgen(tot):

    cen1 = [[-4, 0], [-3, 1], [-2, 2], [-1, 1] , [0, 0], [1, -1], [2, -2], [3, -1], [4, 0]]
    cen2 = [[-4, -3],[-3, -2],  [-2, -1], [-1, -2], [0, -3], [1, -4], [2, -5], [3, -4], [4, -3]]

    l1 = int(len(cen1))
    print(l1)
    delta = random.uniform(0.1, 0.5)
    for i in range(int(l1)):
        # print(cen1[i][1])
        cen1[i][0] -= delta

    t_cluster_std = random.uniform(0.2, 0.6)
    t_n_samples = random.randint(100, 500)
    # print(t_cluster_std)
    x1, y1 = make_blobs(n_samples=t_n_samples, n_features=2, cluster_std=t_cluster_std, centers=cen1)
    x2, y2 = make_blobs(n_samples=t_n_samples, n_features=2, cluster_std=t_cluster_std, centers=cen2)
    pxlist = x1[:, 0].tolist() + x2[:, 0].tolist()
    pylist = x1[:, 1].tolist() + x2[:, 1].tolist()
    t1 = [0 for x in range(t_n_samples)]
    t2 = [1 for x in range(t_n_samples)]
    t = t1 + t2
    sita = random.uniform(0, math.pi/3)
    tpxlist = pxlist
    tpylist = pylist
    le = len(pxlist)
    op = random.choice([1, 2, 3, 4, 5])
    for j in range(le):
        pxlist[j] = math.cos(sita) * tpxlist[j] + math.sin(sita) * tpylist[j]
        pylist[j] = - math.sin(sita) * tpxlist[j] + math.cos(sita) * tpylist[j]
        if op == 3:
            pylist[j] = -pylist[j]
    plt.scatter(pxlist, pylist, c=t, alpha=0.5)
    plt.show()
    # savetocsv(pxlist, pylist, t, str(tot) + "triangle")



def circlegen(tot):

    cen1 = [[0, 3],[-0.75,2.25] ,[-1.5, 1.5], [-2.25,0.75],[-3, 0], [-2.75, -0.75], [-1.5, -1.5],[-0.75,-2.75] ,[0, -3],[0.75,-2.75], [1.5, -1.5],[2.75,-0.75], [3, 0], [2.75,0.75], [1.5,1.5]]
    cen2 = [[0, 6],[-1.5,5.5],   [-3, 3],    [-5.5,1.5],  [-6, 0],  [-5.5,-1.5],     [-3, -3],    [-1.5,-5.5], [0, -6], [1.5,-5.5],[3, -3], [5.5,-1.5], [6, 0], [5.5,1.5], [3, 3]]

    l1 = int(len(cen1))
    # print(l1)
    # delta = random.uniform(0.1, 0.5)
    # for i in range(int(l1)):
    #     # print(cen1[i][1])
    #     cen1[i][0] -= delta

    t_cluster_std = random.uniform(0.2, 0.6)
    t_n_samples = random.randint(100, 500)
    # print(t_cluster_std)
    x1, y1 = make_blobs(n_samples=t_n_samples, n_features=2, cluster_std=t_cluster_std, centers=cen1)
    x2, y2 = make_blobs(n_samples=t_n_samples, n_features=2, cluster_std=t_cluster_std, centers=cen2)
    pxlist = x1[:, 0].tolist() + x2[:, 0].tolist()
    pylist = x1[:, 1].tolist() + x2[:, 1].tolist()
    t1 = [0 for x in range(t_n_samples)]
    t2 = [1 for x in range(t_n_samples)]
    t = t1 + t2
    # sita = random.uniform(0, math.pi/3)
    sita = random.uniform(0, 0.001)
    tpxlist = pxlist
    tpylist = pylist
    le = len(pxlist)
    op = random.choice([1, 2, 3, 4, 5])
    for j in range(le):
        pxlist[j] = math.cos(sita) * tpxlist[j] + math.sin(sita) * tpylist[j]
        pylist[j] = - math.sin(sita) * tpxlist[j] + math.cos(sita) * tpylist[j]
        if op == 3:
            pylist[j] = -pylist[j]
    plt.scatter(pxlist, pylist, c=t, alpha=0.5)
    plt.show()
    # savetocsv(pxlist, pylist, t, str(tot) + "triangle")


def trianglegen(tot):

    cen1 = [[0, 2], [-1,1], [-2,0], [-3,-1], [-4,-2],  [-2,-2], [0,-2], [2,-2], [4,-2],[3,-1],[2,0], [1,1]]
    cen2 = [[0, 6], [-1.5,4.5], [-3,3], [-4,2], [-6,0],[-7.5,-1.5], [-9,-3],[-10.5,-4.5], [-12,-6],[-9,-6] ,[-5,-6] ,[-6,-6],[-3,-6],[-1,-6], [0,-6],[3,-6], [4,-6], [6,-6], [9,-6], [12,-6],[10.5,-4.5],[9,-3],[-7.5,-1.5],[6,0],[4.5,1.5], [3,3]]

    l1 = int(len(cen1))
    # print(l1)
    delta = random.uniform(1, 2)
    for i in range(int(l1)):
        # print(cen1[i][1])
        cen1[i][0] -= delta

    t_cluster_std = random.uniform(0.2, 0.6)
    t_n_samples = random.randint(100, 500)
    # print(t_cluster_std)
    x1, y1 = make_blobs(n_samples=t_n_samples, n_features=2, cluster_std=t_cluster_std, centers=cen1)
    x2, y2 = make_blobs(n_samples=t_n_samples, n_features=2, cluster_std=t_cluster_std, centers=cen2)
    pxlist = x1[:, 0].tolist() + x2[:, 0].tolist()
    pylist = x1[:, 1].tolist() + x2[:, 1].tolist()
    t1 = [0 for x in range(t_n_samples)]
    t2 = [1 for x in range(t_n_samples)]
    t = t1 + t2
    # sita = random.uniform(0, math.pi/3)
    sita = random.uniform(0, 0.001)
    tpxlist = pxlist
    tpylist = pylist
    le = len(pxlist)
    op = random.choice([1, 2, 3, 4, 5])
    for j in range(le):
        pxlist[j] = math.cos(sita) * tpxlist[j] + math.sin(sita) * tpylist[j]
        pylist[j] = - math.sin(sita) * tpxlist[j] + math.cos(sita) * tpylist[j]
        if op == 3:
            pylist[j] = -pylist[j]
    plt.scatter(pxlist, pylist, c=t, alpha=0.5)
    plt.show()
    # savetocsv(pxlist, pylist, t, str(tot) + "triangle")





if __name__ == '__main__':
    random.seed(datetime.now())
    """
    # 正方形缺口嵌套
    for i in range(10):
        squaregen(i+1)

    # 三角形缺口嵌套
    for i in range(10):
        trianglegen(i+1)
    """
    """
    # S形嵌套
    for i in range(10):
        sgen(i+1)

    # 圆形嵌套
    for i in range(10):
        circlegen(i+1)
    """

    # 三角形嵌套
    for i in range(10):
        trianglegen(i+1)



