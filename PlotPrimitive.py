import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from math import sin,cos,pi
from scipy.interpolate import spline
import numpy as np
from scipy import optimize

"""
class Point:
    x=0;y=0
    def __init__(self,x,y):
        self.x = x
        self.y = y
class Line: 
"""

patches = []

def create_circle():
    circle = plt.Circle((0,0),radius=5,fill=False)
    return circle

def create_rectangle():
    rectangle = plt.Rectangle((0,0),4,3,fill=False)
    return rectangle

def create_RegularPolygon():
    poly = mpatches.RegularPolygon((0,0),5,3,fill=False)
    return poly

def create_ellipse():
    ellipse = mpatches.Ellipse((0,0),15,2,fill=False)
    return ellipse

def create_wedge():
    wedge = mpatches.Wedge((0,0),5,0,60,fill=False)
    return wedge

def create_HollowSLine():
    path = mpath.Path
    path_data = [
        (path.MOVETO,[0,0]),
        (path.CURVE4, [2, 3]),
        (path.CURVE4, [4, -3]),
        (path.CURVE4, [6, 0]),
        (path.CURVE4, [4, -2.5]),
        (path.CURVE4, [2, 3.5]),
        (path.CURVE4, [0, 0])
    ]
    codes,verts = zip(*path_data)
    path = mpath.Path(verts,codes)
    patch = mpatches.PathPatch(path,fill=False)
    return patch


"""
def create_HollowHelix():
    path = mpath.Path
    path_data = [
        (path.MOVETO, [0, 0]),
        (path.CURVE4, [3, 2]),
        (path.CURVE4, [0, 4]),
        (path.CURVE4, [-4, 3]),
        (path.CURVE4, [-5, 0]),
        (path.CURVE4, [-4, -4]),
        (path.CURVE4, [0, -6]),
        (path.CURVE4, [3, -5]),
        (path.CURVE4, [7, 0]),
        (path.CURVE4, [5, 6]),
        (path.CURVE4, [0, 8]),
        (path.CURVE4, [-5, 7]),
        (path.CURVE4, [-9, 5]),
        (path.CURVE4, [-11, 0]),
        (path.CURVE4, [-8, -8]),
        (path.CURVE4, [-4, -9]),
        (path.CURVE4, [0, -11]),
        (path.CURVE4, [3, -11]),
        (path.CURVE4, [6, -9]),
        (path.CURVE4, [8, -7]),
        (path.CURVE4, [10, 0]),
        (path.CURVE4, [11, 4])
    ]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, fill=False)
    return patch
"""


def create_HollowHelix():
    plt.subplot(111,polar=True)
    plt.ylim([0,30])
    N = 4
    theta = np.arange(0,N*np.pi,np.pi/100)
    plt.plot(theta,theta*2)
    plt.show()
    #X = [0,2,0,-2.6,0,4,3,0,-4,-6,-5,-3,0,3,6,9,7,4,1,-2,-4,-4,-3,-1,1,3,1,0]
    #Y = [0,1,2.4,0,-3.14,0,3,4.78,3,0,-4,-5,-6,-5,-3,0,-1,-3,-4,-4,-2,0,2,3,3,1,0,0]
    #plt.plot(X, Y)
    #plt.show()
    #XNew = np.linspace(X.min(),X.max(),300)
    #YNew = spline(X,Y,XNew)
    #plt.plot(XNew,YNew)
    #plt.show()



def show_shape(patch):
   patches.append(patch)
   ax=plt.gca()
   ax.add_patch(patch)
   plt.axis('scaled')
   plt.xlim(-8,8)
   plt.ylim(-6,6)
   plt.show()


if __name__ == '__main__':
    c=create_circle()
    show_shape(c)
    c=create_wedge()
    show_shape(c)
    c=create_rectangle()
    show_shape(c)
    c=create_ellipse()
    show_shape(c)
    c=create_RegularPolygon()
    show_shape(c)
    c=create_HollowSLine()
    show_shape(c)
    create_HollowHelix()









