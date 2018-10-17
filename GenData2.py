import matplotlib.pyplot as plt
from sklearn.datasets import *
import random
import pandas as pd
import os
import math
import numpy
from datetime import datetime
import csv

def savetocsv(pxlist, pylist, lab, name):
    sample = []
    le = len(pxlist)
    for i in range(le):
        sample.append([pxlist[i], pylist[i], lab[i]])
    test = pd.DataFrame(columns=['X', 'Y', 'label'], data=sample)
    path = os.path.join('C:/Users/liblnuex/Desktop/VCS/generatedata/', name + '.csv')
    # print(path)
    test.to_csv(path, encoding='utf-8')

if __name__ == '__main__':
    random.seed(datetime.now())


    cen1 = [[-4, -2], [-4, -1], [-4, 0], [-4, 1], [-4, 2], [-4, 3], [-4, 4],
            [-3, 4], [-2, 4], [-1, 4], [0, 4], [1, 4], [2, 4],
            [2, 3], [2, 2], [2, 1], [2, 0], [2, -1], [2, -2]]
    cen2 = [[-2, 2], [-2, 1], [-2, 0], [-2, -1], [-2, -2], [-2, -3], [-2, -4],
            [-1, -4], [0, -4], [1, -4], [2, -4], [3, -4], [4, -4],
            [4, -3], [4, -2], [4, -1], [4, 0], [4, 1], [4, 2]]

    for i in range(10):
        t_cluster_std = random.uniform(0.2, 0.6)
        print(t_cluster_std)
        x1, y1 = make_blobs(n_samples=250, n_features=2, cluster_std=t_cluster_std, centers=cen1)
        x2, y2 = make_blobs(n_samples=250, n_features=2, cluster_std=t_cluster_std, centers=cen2)
        pxlist = x1[:, 0].tolist() + x2[:, 0].tolist()
        pylist = x1[:, 1].tolist() + x2[:, 1].tolist()
        t1 = [0 for x in range(250)]
        t2 = [1 for x in range(250)]
        t = t1 + t2
        sita = random.uniform(0, math.pi/2.0)
        tpxlist = pxlist
        tpylist = pylist
        le = len(pxlist)
        for j in range(le):
            pxlist[j] = math.cos(sita)*tpxlist[j] + math.sin(sita)*tpylist[j]
            pylist[j] = - math.sin(sita)*tpxlist[j] + math.cos(sita)*tpylist[j]
        plt.scatter(pxlist, pylist, c=t, alpha=0.5)
        plt.show()

    cen1 = [[-1, 4], [-2, 3], [-3, 2], [-4, 1], [-5, 0], [-6, -1],
            [-5, -1], [-4, -1], [-3, -1], [-2, -1], [-1, -1], [0, -1]]
    cen2 = [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1],
            [5, 0], [4, -1], [3, -2], [2, -3], [1, -4]]

    for i in range(10):
        t_cluster_std = random.uniform(0.2, 0.6)
        print(t_cluster_std)
        x1, y1 = make_blobs(n_samples=250, n_features=2, cluster_std=t_cluster_std, centers=cen1)
        x2, y2 = make_blobs(n_samples=250, n_features=2, cluster_std=t_cluster_std, centers=cen2)
        pxlist = x1[:, 0].tolist() + x2[:, 0].tolist()
        pylist = x1[:, 1].tolist() + x2[:, 1].tolist()
        t1 = [0 for x in range(250)]
        t2 = [1 for x in range(250)]
        t = t1 + t2
        sita = random.uniform(0, math.pi)
        tpxlist = pxlist
        tpylist = pylist
        le = len(pxlist)
        for j in range(le):
            pxlist[j] = math.cos(sita)*tpxlist[j] + math.sin(sita)*tpylist[j]
            pylist[j] = - math.sin(sita)*tpxlist[j] + math.cos(sita)*tpylist[j]
        plt.scatter(pxlist, pylist, c=t, alpha=0.5)
        plt.show()

    # tot = 1
    # for i in range(1):
    #     t_cluster_std = random.uniform(0.2, 0.6)
    #     print(t_cluster_std)
    #     sita = random.uniform(15, 90)
    #     sita = math.radians(sita)
    #     cen1 = []
    #     cen2 = []
    #     step = numpy.arange(-2, 2.5, .25)
    #     print(step)
    #     for j in step:
    #         cen1.append([j, math.tan(sita) * j])
    #         cen2.append([j, math.tan(-sita) * j])
    #     X1, y1 = make_blobs(n_samples=250, n_features=2, cluster_std=t_cluster_std, centers=cen1)
    #     # plt.scatter(X1[:, 0], X1[:, 1], c='b', alpha=0.5)
    #     X2, y2 = make_blobs(n_samples=250, n_features=2, cluster_std=t_cluster_std, centers=cen2)
    #     # plt.scatter(X2[:, 0], X2[:, 1], c='c', alpha=0.5)
    #     pxlist = X1[:, 0].tolist() + X2[:, 0].tolist()
    #     pylist = X1[:, 1].tolist() + X2[:, 1].tolist()
    #     t1 = [0 for x in range(250)]
    #     t2 = [1 for x in range(250)]
    #     t1 = t1 + t2
    #     plt.scatter(pxlist, pylist, c=t1, alpha=0.5)
    #     plt.show()
    #     # savetocsv(pxlist, pylist, t1, str(tot) + "intersection")
    #     tot = tot + 1