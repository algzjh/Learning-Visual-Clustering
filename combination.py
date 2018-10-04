import matplotlib.pyplot as plt
import math
import pandas as pd
import os
import csv

def getvector():
    namelist = ['Circle', 'Ellipse', 'Helix', 'Hexagon', 'Line', 'Pentagon', 'Rectangle', 'SLine', 'Trapezoid', 'Triangle']
    L1 = []
    for name in namelist:
        # print(name)
        L2 = []
        path = 'C:/Users/liblnuex/Desktop/VCS/datapoint/' + name + '.csv'
        # print(path)
        csv_reader = csv.reader(open(path))
        first = True
        for row in csv_reader:
            if first:
                first = False
                continue
            L2.append(row[1])
            L2.append(row[2])
            # print(row[1],row[2])
        L1.append(L2)
    # print(len(L1),len(L1[0]))
    return L1


def paint(vec, coe):
    newv = []
    for j in range(len(vec[0])):
        newv.append(0)
        for i in range(len(vec)):
            print(i)
            newv[j] = coe[i] * float(vec[i][j])
    x = []
    y = []
    for i in range(len(newv)):
        if i%2 == 1:
            y.append(newv[i])
        else:
            x.append(newv[i])
    plt.axis([-8, 8, -8, 8])
    plt.grid(True)
    plt.scatter(x, y, alpha=0.5)
    plt.show()

if __name__ == '__main__':
    L1 = getvector()
    paint([L1[0], L1[9]],[0.5, 0.5])
    paint([L1[6], L1[2]], [0.0, 1]) # Rectangle Helix
    paint([L1[7], L1[6]], [0.9,0.1])

