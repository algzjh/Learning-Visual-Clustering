import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np

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
        (path.CURVE4, [2, 2]),
        (path.CURVE4, [4, -2]),
        (path.CURVE4, [6, 0]),
        (path.CURVE4, [4, -1.5]),
        (path.CURVE4, [2, 2.5]),
        (path.CURVE4, [0, 0])
    ]
    codes,verts = zip(*path_data)
    path = mpath.Path(verts,codes)
    patch = mpatches.PathPatch(path,fill=False)
    return patch


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

# def create_star():



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
    c=create_HollowHelix()
    show_shape(c)

    #plt.axis('equal')
    #plt.axis('on')
    #plt.tight_layout()
    #plt.show()









