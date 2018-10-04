import matplotlib.pyplot as plt
import math
import pandas as pd
import os


def paint(sample):
    x = []
    y = []
    for i in sample:
        x.append(i[0])
        y.append(i[1])
    plt.axis([-8, 8, -8, 8])
    plt.grid(True)
    plt.scatter(x, y, alpha=0.5)
    plt.show()

def savetocsv(pxlist, pylist, op, name):
    sample = []
    le = len(pxlist)
    if op == 1:
        k = int(le/50)
        for i in range(50):
            sample.append([pxlist[k*i], pylist[k*i]])
        paint(sample)
    if op == 2:
        k = int(le/25)
        delta = 0.25
        for i in range(25):
            sample.append([pxlist[k*i], pylist[k*i]])
        for i in range(25):
            sample.append([pxlist[le-1-k*i]+delta, pylist[le-1-k*i]+delta])
        paint(sample)
    test = pd.DataFrame(columns=['X', 'Y'], data=sample)
    path = os.path.join('C:/Users/liblnuex/Desktop/VCS/datapoint/', name + '.csv')
    print(path)
    test.to_csv(path, encoding='utf-8')

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


def getdistance(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    return math.sqrt(dx*dx+dy*dy)


class Line:
    pxlist = []
    pylist = []

    def __init__(self, vlist):
        self.a = vlist[0]
        self.b = vlist[1]

    def subdivide(self):
        self.pxlist = []
        self.pylist = []
        step = 0
        sx = self.a.x
        sy = self.a.y
        tx = self.b.x
        ty = self.b.y
        while step <= 1:
            self.pxlist.append(sx + step * (tx - sx))
            self.pylist.append(sy + step * (ty - sy))
            step = step + 0.01

    def draw(self):
        self.subdivide()
        plt.axis([-8, 8, -8, 8])
        plt.grid(True)
        plt.plot(self.pxlist, self.pylist)
        plt.show()
        savetocsv(self.pxlist, self.pylist, 2, 'Line')


class Triangle:
    vlist = []
    pxlist = []
    pylist = []
    xmin = 1e9
    xmax = -1e9
    ymin = 1e9
    ymax = -1e9

    def __init__(self, vlist):
        self.vlist = vlist

    def subdivide(self):
        self.vlist.append(self.vlist[0])
        for i in range(len(self.vlist)-1):
            sx = self.vlist[i].x
            sy = self.vlist[i].y
            tx = self.vlist[i+1].x
            ty = self.vlist[i+1].y
            step = 0
            while step <= 1:
                self.pxlist.append(sx + step*(tx-sx))
                self.pylist.append(sy + step*(ty-sy))
                step = step + 0.01

    def draw(self):
        self.subdivide()
        plt.axis([-8, 8, -8, 8])
        plt.grid(True)
        plt.plot(self.pxlist, self.pylist)
        plt.show()
        savetocsv(self.pxlist, self.pylist, 1, 'Triangle')


class Circle:
    c = Point()
    r = 0
    pxlist = []
    pylist = []
    xmin = 1e9
    xmax = -1e9
    ymin = 1e9
    ymax = -1e9

    def __init__(self, c, r):
        self.c = c
        self.r = r

    def subdivide(self):
        step = 0
        while step <= 1:
            self.pxlist.append(self.r * math.cos(2 * math.pi * step))
            self.pylist.append(self.r * math.sin(2 * math.pi * step))
            step = step + 0.001

    def draw(self):
        self.subdivide()
        plt.axis([-8, 8, -8, 8])
        plt.grid(True)
        plt.plot(self.pxlist, self.pylist)
        plt.show()
        savetocsv(self.pxlist, self.pylist, 1, 'Circle')


class Rectangle:
    vlist = []
    pxlist = []
    pylist = []

    def __init__(self, vlist):
        self.vlist = vlist

    def subdivide(self):
        self.pxlist = []
        self.pylist = []
        self.vlist.append(self.vlist[0])
        for i in range(len(self.vlist) - 1):
            sx = self.vlist[i].x
            sy = self.vlist[i].y
            tx = self.vlist[i + 1].x
            ty = self.vlist[i + 1].y
            step = 0
            while step <= 1:
                self.pxlist.append(sx + step * (tx - sx))
                self.pylist.append(sy + step * (ty - sy))
                step = step + 0.01

    def draw(self, name):
        self.subdivide()
        plt.axis([-8, 8, -8, 8])
        plt.grid(True)
        plt.plot(self.pxlist, self.pylist)
        plt.show()
        savetocsv(self.pxlist, self.pylist, 1, name)


class SLine:
    pxlist = []
    pylist = []

    def __init__(self):
        pass

    def subdivide(self):
        step = -4
        while step <= 4:
            self.pxlist.append(step)
            self.pylist.append(4*math.sin(math.pi/4.0*(step+4.0)))
            step = step + 0.01

    def draw(self):
        self.subdivide()
        plt.axis([-8, 8, -8, 8])
        plt.grid(True)
        plt.plot(self.pxlist, self.pylist)
        plt.show()
        savetocsv(self.pxlist, self.pylist, 2, 'SLine')



class Helix:
    pxlist = []
    pylist = []
    xmin = 1e9
    xmax = -1e9
    ymin = 1e9
    ymax = -1e9

    def __init__(self):
        pass

    def subdivide(self):
        step = 0
        while step <= 2:
            r = 4.0/3.0 * (1 + step)
            self.pxlist.append(r * math.cos(step * 2 * math.pi))
            self.pylist.append(r * math.sin(step * 2 * math.pi))
            step = step + 0.01

    def draw(self):
        self.subdivide()
        plt.axis([-8, 8, -8, 8])
        plt.grid(True)
        plt.plot(self.pxlist, self.pylist)
        plt.show()
        savetocsv(self.pxlist, self.pylist, 2, 'Helix')


class ellipse:
    pxlist = []
    pylist = []
    xmin = 1e9
    xmax = -1e9
    ymin = 1e9
    ymax = -1e9

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def subdivide(self):
        self.pxlist = []
        self.pylist = []
        step = 0
        while step <= 1:
            self.pxlist.append(self.a * math.cos(2 * math.pi * step))
            self.pylist.append(self.b * math.sin(2 * math.pi * step))
            step = step + 0.001

    def draw(self):
        self.subdivide()
        plt.axis([-8, 8, -8, 8])
        plt.grid(True)
        plt.plot(self.pxlist, self.pylist)
        plt.show()
        savetocsv(self.pxlist, self.pylist, 1, 'Ellipse')


class Trapezoid:
    vlist = []
    pxlist = []
    pylist = []

    def __init__(self, vlist):
        self.vlist = vlist

    def subdivide(self):
        self.vlist.append(self.vlist[0])
        for i in range(len(self.vlist) - 1):
            sx = self.vlist[i].x
            sy = self.vlist[i].y
            tx = self.vlist[i + 1].x
            ty = self.vlist[i + 1].y
            step = 0
            while step <= 1:
                self.pxlist.append(sx + step * (tx - sx))
                self.pylist.append(sy + step * (ty - sy))
                step = step + 0.01

    def draw(self):
        self.subdivide()
        plt.axis([-8, 8, -8, 8])
        plt.grid(True)
        plt.plot(self.pxlist, self.pylist)
        plt.show()
        savetocsv(self.pxlist, self.pylist, 1, 'Trapezoid')



if __name__ == '__main__':
    vlist = [Point(-4, -4), Point(4, -4), Point(0, 4)]
    tri = Triangle(vlist)
    tri.draw()
    cir = Circle(Point(0, 0), 4)
    cir.draw()
    vlist = [Point(-4, -4), Point(4, -4), Point(4, 4), Point(-4, 4)]
    rec = Rectangle(vlist)
    rec.draw('Rectangle')
    sli = SLine()
    sli.draw()
    hel = Helix()
    hel.draw()
    ell = ellipse(4,2)
    ell.draw()
    # ell2 = ellipse(2,4)
    # ell2.draw()
    vlist = [Point(-4, -4), Point(4, -4), Point(2, 4), Point(-2, 4)]
    tra = Trapezoid(vlist)
    tra.draw()
    a = 4 * math.sin(36/180.0 * math.pi)
    b = 4 * math.cos(36/180.0 * math.pi)
    c = 2 * a * math.cos(72/180.0 * math.pi)
    d = 2 * a * math.sin(72/180.0 * math.pi)
    vlist = [Point(-a, -b), Point(a, -b), Point(a + c, -b + d), Point(0, 4), Point(-a - c, -b + d)]
    pen = Rectangle(vlist)
    pen.draw('Pentagon')
    a = 4 * math.cos(60/180.0 * math.pi)
    b = 4 * math.sin(60/180.0 * math.pi)
    vlist = [Point(-a, -b), Point(a, -b), Point(4, 0), Point(a, b), Point(-a, b), Point(-4, 0)]
    hex = Rectangle(vlist)
    hex.draw('Hexagon')
    vlist = [Point(-4, 0), Point(4,0)]
    lin = Line(vlist)
    lin.draw()












