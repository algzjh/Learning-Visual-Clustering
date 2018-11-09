# encoding: utf-8
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import math
import random
import numpy
import pandas as pd
import os
import csv


def savetocsv(pxlist, pylist, op, name):
    sample = []
    tot = len(pxlist)
    for i in range(tot):
        sample.append([pxlist[i], pylist[i]])
    test = pd.DataFrame(columns = ['X', 'Y'], data = sample)
    path = os.path.join('C:/Users/liblnuex/Desktop/VCS/singlecluster/', name + '.csv')
    test.to_csv(path, encoding = 'utf-8')


def dcmp(x):
    """
    compare the floating number to zero
    :param x: a float number 
    :return:  [-1,0,1]
    """
    if math.fabs(x) <= 1e-4:
        return 0
    elif x > 0:
        return 1
    else:
        return -1


def draw_skeleton(pxlist, pylist):
    """
    draw a scatterplot
    :param pxlist: the x-coordinate of the scatter
    :param pylist: the y-coordinate of the scatter
    :return: a scatter diagram
    :rtype: void
    :param pxlist: x coordinates   
    :param pylist: y coordinates
    """
    plt.scatter(pxlist, pylist, alpha=0.5)  # 绘制出图形
    plt.show()


def getdist(x1, y1, x2, y2):
    """
    calculate the distance between two points
    :param x1: the x-coordinate of the first point
    :param y1: the y-coordinate of the first point
    :param x2: the x-coordinate of the second point
    :param y2: the y-coordinate of the second point
    :return: 
    """
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


def gettrianglearea(x1, y1, x2, y2, x3, y3):
    """
    calculate the area of the triangle
    :rtype: float
    :param x1: vertices1.x
    :param y1: vertices1.y
    :param x2: vertices2.x
    :param y2: vertices2.y
    :param x3: vertices3.x
    :param y3: vertices3.y
    :return: S the area of the triangle
    """
    S = 0.5 * (x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2)
    S = math.fabs(S)
    return S


def gettriangleperimeter(x1, y1, x2, y2, x3, y3):
    """
    Calculate the perimeter of the triangle
    :param x1: vertices1.x
    :param y1: vertices1.y
    :param x2: vertices2.x
    :param y2: vertices2.y
    :param x3: vertices3.x
    :param y3: vertices3.y
    :return: perimeter of the triangle
    """
    return getdist(x1, y1, x2, y2) + \
           getdist(x2, y2, x3, y3) + \
           getdist(x3, y3, x1, y1)


def point_in_polygon(nvert, vertx, verty, testx, testy):
    """
    Determine if the point is inside the polygon
    :param nvert: the number of polygon vertex
    :param vertx: the x-coordinate set of vertexes
    :param verty: the y-coordinate set of vertexes
    :param testx: the x-coordinate of the test vertex
    :param testy: the y-coordinate of the test vertex
    :return:   1: in polygon 0: not in polygon 
    """
    crossing = 0
    i = 0
    j = nvert - 1
    while i < nvert:
        if (verty[i] > testy) != (verty[j] > testy):
            if testx < float((vertx[j] - vertx[i]) * (testy - verty[i])) / (verty[j] - verty[i]) + vertx[i]:
                crossing = crossing + 1
        j = i
        i = i + 1
    return crossing % 2 != 0


def get_bounding_box(pxlist, pylist):
    """
    get the bounding bos of the polygon
    :param pxlist: the x-coordinate set of vertexes
    :param pylist: the y-coordinate set of vertexes
    :return: the min & max of x/y coordinate
    """
    minx = 1e9
    maxx = -1e9
    miny = 1e9
    maxy = -1e9
    tot = len(pxlist)
    for i in range(tot):
        minx = min(pxlist[i], minx)
        maxx = max(pxlist[i], maxx)
        miny = min(pylist[i], miny)
        maxy = max(pylist[i], maxy)
    return minx, maxx, miny, maxy


def get_box_distribution(minx, maxx, miny, maxy, cluster_density, cluster_distribution, cluster_shape):
    """

    :param minx: the minimum of the x coordinate
    :param maxx: the maximum of the x coordinate
    :param miny: the minimum of the y coordinate
    :param maxy: the maximum of the y coordinate
    :param cluster_density: literal meaning
    :param cluster_distribution: literal meaning, 1:uniform, 2:random, 3:Gaussian
    :param cluster_shape: literal meaning, 1:圆形、2:椭圆形、3:三角形、4:四边形、5:五边形、6:六边形、7:七边形
    :return: get the required  points distribution within the bounding box
    """
    rectangle_area = (maxx - minx)*(maxy-miny)
    num_of_points = cluster_density*rectangle_area/(math.pi*3*3)
    # print("num_of_points: ", num_of_points)
    if cluster_distribution == 1:
        square_side = math.sqrt(rectangle_area / num_of_points)
        square_side_div_two = int(square_side / 2)
        # print("square_side: ", square_side)
        col_cnt = int((maxx - minx) / square_side)
        row_cnt = int((maxy - miny) / square_side)
        row_step = 1
        col_step = 1
        if cluster_shape == 2:
            row_step = 5
        if cluster_shape == 3:
            row_step = 2
            col_step = 2
        box_x = []
        box_y = []
        for i in range(0, row_cnt, row_step):
            for j in range(0, col_cnt, col_step):
                box_x.append(minx + square_side_div_two + square_side * j)
                box_y.append(miny + square_side_div_two + square_side * i)
        return box_x, box_y
    elif cluster_distribution == 2:
        square_side = math.sqrt(rectangle_area / num_of_points)
        square_side_div_two = int(square_side / 2)
        # print("square_side: ", square_side)
        col_cnt = int((maxx - minx) / square_side)
        row_cnt = int((maxy - miny) / square_side)
        row_step = 1
        col_step = 1
        if cluster_shape == 2:
            row_step = 2
            col_step = 2
        box_x = []
        box_y = []
        for i in range(0, row_cnt, row_step):
            for j in range(0, col_cnt, col_step):
                box_x.append(random.uniform(minx, maxx))
                box_y.append(random.uniform(miny, maxy))
        return box_x, box_y
    else:
        square_side = math.sqrt(rectangle_area / num_of_points)
        square_side_div_two = int(square_side / 2)
        # print("square_side: ", square_side)
        col_cnt = int((maxx - minx) / square_side)
        row_cnt = int((maxy - miny) / square_side)
        if cluster_shape == 1:
            box_x = 60*numpy.random.randn(1, max(col_cnt*80, row_cnt*80)) + int((maxx+minx)/2)
            box_y = 60*numpy.random.randn(1, max(col_cnt*80, row_cnt*80)) + int((maxy+miny)/2)
        elif cluster_shape == 2:
            box_x = 300 * numpy.random.randn(1, max(col_cnt * 80, row_cnt * 80)) + int((maxx + minx) / 2)
            box_y = 300 * numpy.random.randn(1, max(col_cnt * 80, row_cnt * 80)) + int((maxy + miny) / 2)
        elif cluster_shape == 3 or cluster_shape == 4 or cluster_shape == 5 or cluster_shape == 6 or cluster_shape == 7:
            box_x = 100 * numpy.random.randn(1, max(col_cnt * 80, row_cnt * 80)) + int((maxx + minx) / 2)
            box_y = 100 * numpy.random.randn(1, max(col_cnt * 80, row_cnt * 80)) + int((maxy + miny) / 2)
        # print("type: ", type(box_x))
        print(box_x[0])
        print(type(box_x[0]))
        box_x = box_x[0].tolist()
        box_y = box_y[0].tolist()
        return box_x, box_y


def counterclockwise_cmp(point):
    """
    vector = [point[0], point[1]]
    lenvector = math.hypot(vector[0], vector[1])
    if lenvector == 0:
        return -math.pi, 0
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod = normalized[1]
    diffprod = normalized[0]
    angle = math.atan2(diffprod, dotprod)
    if angle < 0:
        return 2*math.pi+angle, lenvector
    return angle, lenvector
    """
    angle = math.atan2(point[1], point[0])
    if angle < 0:
        return 2*math.pi + angle
    return angle


def filter_points(pxlist, pylist, box_x, box_y, central_point):
    tot = len(box_x)
    nvert = len(pxlist)
    pts = []
    for i in range(nvert):
        pts.append([pxlist[i]-central_point[0], pylist[i]-central_point[1]])
    pts.sort(key=counterclockwise_cmp)
    # sorted(pts, key=counterclockwise_cmp)
    minx = 1e9
    maxx = -1e9
    miny = 1e9
    maxy = -1e9
    for i in range(nvert):
        pxlist[i] = pts[i][0] + central_point[0]
        pylist[i] = pts[i][1] + central_point[1]
        minx = min(minx, pxlist[i])
        maxx = max(maxx, pxlist[i])
        miny = min(miny, pylist[i])
        maxy = max(maxy, pylist[i])
    plt.plot(pxlist, pylist)
    draw_skeleton(box_x, box_y)
    fillx = []
    filly = []
    print(minx, maxx, miny, maxy)
    # print("tot: ", len(box_x))
    count = 0
    for i in range(tot):
        if box_x[i] < minx or box_x[i] > maxx or box_y[i] < miny or box_y[i] > maxy:
            count = count + 1
            continue
        # print("xixi")
        if point_in_polygon(nvert, pxlist, pylist, box_x[i], box_y[i]) == 1:
            fillx.append(box_x[i])
            filly.append(box_y[i])
    print(count)
    return fillx, filly


def gen_distribution(cluster_area, cluster_density, cluster_distribution, pxlist, pylist, central_point, cluster_shape):
    """

    :param cluster_area: 
    :rtype: a set of points of a particular distribution
    :param cluster_density: float
    :param cluster_distribution: int {1:uniform, 2:random, 3:Gaussian} 以区域中心点为中心
    :param pxlist: The x-coordinate of the boundary point
    :param pylist: The y-coordinate of the boundary point
    :param central_point: center
    """
    minx, maxx, miny, maxy = get_bounding_box(pxlist, pylist)
    box_x, box_y = get_box_distribution(minx, maxx, miny, maxy, cluster_density, cluster_distribution, cluster_shape)
    draw_skeleton(box_x, box_y)
    fillx, filly = filter_points(pxlist, pylist, box_x, box_y, central_point)
    # print("haha: ", len(fillx), len(filly))
    draw_skeleton(fillx, filly)
    return fillx, filly

def get_final_position(px, py, vx, vy, noise_amount, noise_width, central_point):
    """

    :rtype: Coordinates of points affected by noise points
    :param px: The original x-coordinate of the point
    :param py: The original y-coordinate of the point
    :param vx: The x-coordinates of the noise point set
    :param vy: The y-coordinates of the noise point set
    :param noise_amount: The number of the noise points
    """
    """
    Find the coordinates for the equilibrium position
    Using methods similar to gradient descent
    F1 = k1*q1*q2/r/r
    F2 = -k2*x
    The result is poor, so try to take a new approach
    The original point moves a certain distance to the noise point
    depending on the distance from the noise point
    the farther it is, the smaller the distance you move
    """
    now_x = px
    now_y = py
    noise_width *= 2.0
    min_width = 5
    k = float(noise_width*noise_width*min_width*min_width)/(min_width*min_width - noise_width*noise_width)
    b = -1.0*k/(min_width*min_width)
    # k = float(noise_width * noise_width) / (4.0 * noise_width * noise_width - 1.0)
    # b = 1.0-4.0*k
    cx = central_point[0]
    cy = central_point[1]
    delta_x = 0
    delta_y = 0
    for i in range(noise_amount):
        d = getdist(vx[i], vy[i], px, py)
        d2 = getdist(vx[i], vy[i], cx, cy)
        force = k/(1.0*d*d) + b
        # if dcmp(d2-d) > 0:
        #     force -=0.1
        alpha = math.atan2(vy[i]-py, vx[i]-px)
        delta_x += force*math.cos(alpha)
        delta_y += force*math.sin(alpha)
        """
        tx = force*math.cos(alpha)
        if tx < 0:
            delta_x += max(tx, d*math.cos(alpha))
        else:
            delta_x += min(tx, d*math.cos(alpha))
        ty = force*math.sin(alpha)
        if ty < 0:
            delta_y += max(ty, d*math.sin(alpha))
        else:
            delta_y += min(ty, d*math.sin(alpha))
        """
    # delta_x *= 2.0
    # delta_y *= 2.0
    # print("delta_x: ", delta_x)
    # print("delta_y: ", delta_y)
    move_dis = math.sqrt(delta_x*delta_x + delta_y*delta_y)
    for i in range(noise_amount):
        td = getdist(now_x, now_y, vx[i], vy[i])
        if move_dis > td:
            move_dis = td
            delta_x = vx[i] - now_x
            delta_y = vy[i] - now_y
    # now_x += 5000*delta_x
    # now_y += 5000*delta_y
    now_x += 10*delta_x
    now_y += 10*delta_y
    return now_x, now_y


def get_noise_effect(pxlist, pylist, noise_amount, noise_width, central_point):
    """
    
    :param central_point: [float, float]
    :param pxlist: The x-coordinate of the set of points
    :param pylist: The y-coordinate of the set of points
    :param noise_amount: The number of the noise points
    :param noise_width: The width range in which noise occurs
    :return: The set of skeleton points after deformation
    """
    """
    The noise points are regarded as a magnet, the skeleton points are 
    fixed in the initial position by the rubber band, and finally the 
    skeleton points are in the state of force balance.
    """
    min_noise_dis = 5
    noise_x = []
    noise_y = []
    tot = len(pxlist)
    noise_sum_x = 0
    noise_sum_y = 0
    cx = central_point[0]
    cy = central_point[1]
    for i in range(noise_amount):
        pos = random.randint(0, tot-1)
        pos2 = (pos+1) % tot
        if dcmp(pxlist[pos]-pxlist[pos2]) == 0:  # 斜率不存在, 水平移动
            dir = random.choice([-1, 1])
            delta = random.uniform(min(min_noise_dis, noise_width), max(min_noise_dis, noise_width))
            noise_x.append(pxlist[pos]+dir*delta)
            noise_y.append(pylist[pos])
        else:  # 斜率存在，在法线方向
            norm_k = -1.0/((pylist[pos2]-pylist[pos])/(pxlist[pos2]-pxlist[pos]))
            norm_x = 1.0/math.sqrt(1.0+norm_k*norm_k)
            norm_y = norm_k/math.sqrt(1.0+norm_k*norm_k)
            dir = random.choice([-1, 1])
            delta = random.uniform(min(min_noise_dis, noise_width), max(min_noise_dis, noise_width))
            noise_x.append(pxlist[pos]+dir*norm_x*delta)
            noise_y.append(pylist[pos]+dir*norm_y*delta)
        noise_sum_x += noise_x[i]
        noise_sum_y += noise_y[i]

    for i in range(tot):
        pxlist[i], pylist[i] = get_final_position(pxlist[i], pylist[i], noise_x, noise_y, noise_amount,
                                                  noise_width, central_point)
        """
        sumx = 0
        sumy = 0
        for j in range(noise_amount):
            d = getdist(pxlist[i], pylist[i], noise_x[j], noise_y[j])
            vecx = noise_x[j] - pxlist[i]
            vecy = noise_y[j] - pylist[i]
            beta = math.atan2(vecy, vecx)
            sumx += math.cos(math.atan2(vecy, vecx))*0.5*float((noise_width-d))/noise_width
            sumy += math.sin(math.atan2(vecy, vecx))*0.5*float((noise_width-d))/noise_width
            # sumx += math.cos(math.atan2(vecy, vecx))*0.3*float((noise_width*noise_width-d*d))/(noise_width*noise_width)
            # sumy += math.sin(math.atan2(vecy, vecx))*0.3*float((noise_width*noise_width-d*d))/(noise_width*noise_width)
        pxlist[i] += sumx
        pylist[i] += sumy
        """
    """
    for i in range(tot):
    new_x = pxlist[i]
    new_y = pylist[i]
    delta_x = 0
    delta_y = 0
    for j in range(noise_amount):
        dis = getdist(noise_x[j], noise_y[j], pxlist[i], pylist[i])
        force = 1/(dis*dis)
        delta_x += force*math.fabs(math.cos((noise_y[j]-pylist[i])/(noise_x[j]-pxlist[i])))
        delta_y += force*math.fabs(math.sin((noise_y[j]-pylist[i])/(noise_x[j]-pxlist[i])))
    pxlist[i] += delta_x
    pylist[i] += delta_y
    """
    # for i in range(noise_amount):
    #     pxlist.append(noise_x[i])
    #     pylist.append(noise_y[i])
    # for i in range(tot):
    #     pxlist[i] = float((pxlist[i]+noise_sum_x))/(noise_amount+1)
    #     pylist[i] = float((pylist[i]+noise_sum_y))/(noise_amount+1)
    return pxlist, pylist


def draw_circle_skeleton(cluseter_angle, cluster_area, cluster_density,
                         cluster_distribution, noise_amount, noise_width, central_point, cluster_shape):
    """

    :rtype: list point sets skeleton
    :param cluster_area: float
    :param cluseter_angle: float [-180,180]
    :param central_point: [float, float]
    """
    """
        (x-a)^2 + (y-b)^2 = r^2;
        pi*r*r = cluster_area --> r = sqrt(cluster_area/pi)
        x = a + r*sin(sita)
        y = b + r*cos(sita)
    """
    circle_sets = []
    pxlist = []
    pylist = []
    a = central_point[0]
    b = central_point[1]
    r = math.sqrt(cluster_area / math.pi)
    alpha = cluseter_angle / 180.0 * math.pi
    # print("r: ", r)
    # 0 2*pi* 1/200  2*pi*2/200 ...  2*pi*199/200
    for i in range(50):
        sita = 2 * math.pi * i / 50
        x = a + r * math.sin(sita)
        y = b + r * math.cos(sita)
        new_x = (x - a) * math.cos(alpha) - (y - b) * math.sin(alpha) + a
        new_y = (x - a) * math.sin(alpha) + (y - b) * math.cos(alpha) + b
        point = [new_x, new_y]
        pxlist.append(new_x)
        pylist.append(new_y)
        circle_sets.append(point)
    draw_skeleton(pxlist, pylist)
    pxlist, pylist = get_noise_effect(pxlist, pylist, noise_amount, noise_width, central_point)
    draw_skeleton(pxlist, pylist)
    pxlist, pylist = gen_distribution(cluster_area, cluster_density, cluster_distribution, pxlist, pylist, central_point, cluster_shape)
    draw_skeleton(pxlist, pylist)
    return circle_sets


def draw_oval_skeleton(cluster_angle, cluster_area, cluster_density,
                       cluster_distribution, noise_amount, noise_width, central_point):
    """

    :rtype: list point sets skeleton
    :param cluster_area: float
    :param cluster_angle: float [-180,180]
    :param central_point: [float, float]
    """

    """
    x^2/a^2 + y^2/b^2 = 1
    area = pi*a*b
    central_point: (m,n)
    x = m + a*cos(sita)
    y = n + b*sin(sita)
    """
    a = random.randint(8, 70)  # 半长轴 semi-major axis
    b = cluster_area / math.pi / a
    # 0 2*pi*1/200 2*pi*2/200 2*pi*3/200 ... 2*pi*199/200
    print("b: ", b)
    oval_sets = []
    pxlist = []
    pylist = []
    m = central_point[0]
    n = central_point[1]
    print(cluster_angle)
    alpha = cluster_angle / 180.0 * math.pi
    print("alpha: ", alpha)
    for i in range(50):
        sita = 2 * math.pi * i / 50
        x = m + a * math.cos(sita)
        y = n + b * math.sin(sita)
        new_x = (x - m) * math.cos(alpha) - (y - n) * math.sin(alpha) + m
        new_y = (x - m) * math.sin(alpha) + (y - n) * math.cos(alpha) + n
        point = [new_x, new_y]
        pxlist.append(new_x)
        pylist.append(new_y)
        oval_sets.append(point)
    draw_skeleton(pxlist, pylist)
    pxlist, pylist = get_noise_effect(pxlist, pylist, noise_amount, noise_width, central_point)
    draw_skeleton(pxlist, pylist)
    pxlist, pylist = gen_distribution(cluster_area, cluster_density, cluster_distribution, pxlist, pylist, central_point, 2)
    draw_skeleton(pxlist, pylist)
    return oval_sets

def draw_triangle_skeleton(cluster_angle, cluster_area, cluster_density,
                           cluster_distribution, noise_amount, noise_width, central_point):
    """

    :rtype: list point sets skeleton
    :param cluster_area: float 缩放
    :param cluster_angle: float [-180,180]
    :param central_point: [float, float]
    """
    """
    Randomly generate 3 points and scale to the required area
    """
    vx = []
    vy = []
    cx = 0
    cy = 0
    a = central_point[0]
    b = central_point[1]
    while True:
        for i in range(3):
            vx.append(random.randint(0, 50))
            vy.append(random.randint(0, 50))
        # Sacle the triangle to required area
        old_S = gettrianglearea(vx[0], vy[0], vx[1], vy[1], vx[2], vy[2])
        if dcmp(old_S) != 0:
            break
    t = math.sqrt(float(cluster_area) / float(old_S))
    for i in range(3):
        vx[i] *= t
        vy[i] *= t
        cx += vx[i]
        cy += vy[i]
    cx /= 3.0
    cy /= 3.0
    for i in range(3):
        vx[i] -= (cx - a)
        vy[i] -= (cy - b)
    vx.append(vx[0])
    vy.append(vy[0])
    print("new_S: ", gettrianglearea(vx[0], vy[0], vx[1], vy[1], vx[2], vy[2]))
    C = gettriangleperimeter(vx[0], vy[0], vx[1], vy[1], vx[2], vy[2])
    triangle_sets = []
    pxlist = []
    pylist = []

    for j in range(3):
        step_num = int(float(getdist(vx[j], vy[j], vx[(j + 1) % 3], vy[(j + 1) % 3])) / float(C) * 200.0)
        for i in range(step_num):
            new_x = vx[j] + 1.0 * i / step_num * (vx[(j + 1) % 3] - vx[j])
            new_y = vy[j] + 1.0 * i / step_num * (vy[(j + 1) % 3] - vy[j])
            point = [new_x, new_y]
            pxlist.append(new_x)
            pylist.append(new_y)
            triangle_sets.append(point)
    print(len(pxlist), len(pylist))
    pxlist, pylist = get_noise_effect(pxlist, pylist, noise_amount, noise_width)
    draw_skeleton(pxlist, pylist)
    return triangle_sets
    # plt.plot(vx, vy)
    # plt.show()


def getpolygonperimeter(vx, vy):
    totvex = len(vx)
    sumlen = 0
    for i in range(totvex):
        sumlen += getdist(vx[i], vy[i], vx[(i + 1) % totvex], vy[(i + 1) % totvex])
    return sumlen


def getpolygonarea(vx, vy):
    """
    多边形坐落于一个外接圆上，利用叉积求面积
    :param vx: 顶点横坐标
    :param vy: 顶点纵坐标
    :return: 多边形的面积
    """
    tot = len(vx)
    sum = 0
    for i in range(tot):
        sum += vx[i]*vy[(i+1)%tot] - vx[(i+1)%tot]*vy[i]
    return sum*0.5


def draw_polygon_skeleton(num_vertex, cluster_angle, cluster_area, cluster_density,
                          cluster_distribution, noise_amount, noise_width, central_point):
    """

    :rtype: list point sets skeleton
    :param num_vertex: the number of vertexes
    :param cluster_area: float 缩放
    :param cluster_angle: float [-180,180]
    :param central_point: [float, float]
    """
    """
    Determine an external circle, then determine the vertex on the circle,
    connect and scale to the specified area
    r = 20
    x = rcos(sita)
    y = rsin(sita)
    """
    r = 20
    vx = []
    vy = []
    totangle = 2 * math.pi
    minangle = 20.0 / 360.0 * totangle
    pre = 0
    for i in range(num_vertex - 1):
        sita = random.uniform(min(minangle, 2.0 * totangle / 3.0), 2.0 * totangle / 3.0)
        totangle -= sita
        vx.append(r * math.cos(pre + sita))
        vy.append(r * math.sin(pre + sita))
        pre += sita
    vx.append(r * math.cos(pre + totangle))
    vy.append(r * math.sin(pre + totangle))
    alpha = cluster_angle/180.0*math.pi
    for i in range(num_vertex):
        newx = vx[i]*math.cos(alpha) - vy[i]*math.sin(alpha)
        newy = vx[i]*math.sin(alpha) + vy[i]*math.cos(alpha)
        vx[i] = newx
        vy[i] = newy
    # vx.append(vx[0])
    # vy.append(vy[0])
    S = getpolygonarea(vx, vy)
    t = math.sqrt(float(cluster_area)/float(S))
    cx = 0
    cy = 0
    a = central_point[0]
    b = central_point[1]
    for i in range(num_vertex):
        vx[i] *= t
        vy[i] *= t
        cx += vx[i]
        cy += vy[i]
    cx /= num_vertex
    cy /= num_vertex
    for i in range(num_vertex):
        vx[i] -= (cx - a)
        vy[i] -= (cy - b)
    print("area: ", getpolygonarea(vx, vy))
    pxlist = []
    pylist = []
    polygon_sets = []
    C = getpolygonperimeter(vx, vy)
    for i in range(num_vertex):
        step_num = int(
            float(getdist(vx[i], vy[i], vx[(i + 1) % num_vertex], vy[(i + 1) % num_vertex])) / float(C) * 300)
        print("i: ", i)
        print("dist: ", getdist(vx[i], vy[i], vx[(i + 1) % num_vertex], vy[(i + 1) % num_vertex]))
        print("step_num: ", step_num)
        for j in range(step_num):
            new_x = vx[i] + 1.0 * j * (vx[(i + 1) % num_vertex] - vx[i]) / float(step_num)
            new_y = vy[i] + 1.0 * j * (vy[(i + 1) % num_vertex] - vy[i]) / float(step_num)
            point = [new_x, new_y]
            pxlist.append(new_x)
            pylist.append(new_y)
            polygon_sets.append(point)
    print(len(pxlist), len(pylist))
    draw_skeleton(pxlist, pylist)
    pxlist, pylist = get_noise_effect(pxlist, pylist, noise_amount, noise_width, central_point)
    draw_skeleton(pxlist, pylist)
    pxlist, pylist = gen_distribution(cluster_area, cluster_density, cluster_distribution, pxlist, pylist,
                                      central_point, num_vertex)
    draw_skeleton(pxlist, pylist)
    return polygon_sets
    # plt.plot(vx, vy)
    # plt.show()


def draw_starlike_skeleton(cluster_angle, cluster_area, cluster_density,
                           cluster_distribution, noise_amount, noise_width, central_point):
    """

    :param cluster_angle: float
    :param cluster_area: float
    :param cluster_density: float
    :param cluster_distribution: int {1:uniform, 2:random, 3:Gaussian} 以区域中心点为中心
    :param noise_amount: int
    :param noise_width: float
    :param central_point: [float, float]
    :return: list, point sets skeleton
    """
    """
    (x-a)^2 + (y-b)^2 = r1^2
    (x-a)^2 + (y-b)^2 = r2^2
    pi*r*r = cluster_area --> r = sqrt(cluster_area/pi)
    x = a + r*sin(sita)
    y = b + r*cos(sita)
    """
    star_sets = []
    pxlist = []
    pylist = []
    a = central_point[0]
    b = central_point[1]
    r1 = math.sqrt(cluster_area/math.pi)
    r2 = r1*0.5
    alpha = cluster_angle/180.0*math.pi
    for i in range(0, 360, 36):
        if (i/36)%2 == 0:
            pass
        else:
            pass
    return star_sets

def gen_region_cluster_skeleton(cluster_type, cluster_shape, cluster_angle,
                                cluster_area, cluster_density, cluster_distribution,
                                noise_amount, noise_width, central_point):
    """

    :rtype: point sets skeleton
    :param cluster_type: int  1: linear  2: regional
    :param cluster_shape: int 1:圆形、2:椭圆形、3:三角形、4:四边形、5:五边形、6:六边形、7:七边形 8：星形
    :param cluster_area: float 缩放
    :param cluster_angle: float [-180,180]
    :param cluster_density: float
    :param cluster_distribution: int {1:uniform, 2:random, 3:Gaussian} 以区域中心点为中心
    :param noise_amount: int
    :param noise_width: float
    :param central_point: [float, float]
    """
    points_sets = []
    if cluster_type == 1:
        return points_sets
    if cluster_shape == 1:
        points_sets = draw_circle_skeleton(cluster_angle, cluster_area, cluster_density,
                                           cluster_distribution, noise_amount, noise_width, central_point, cluster_shape)
    elif cluster_shape == 2:
        points_sets = draw_oval_skeleton(cluster_angle, cluster_area, cluster_density,
                                         cluster_distribution, noise_amount, noise_width, central_point)
    elif cluster_shape == 3:
        points_sets = draw_polygon_skeleton(3, cluster_angle, cluster_area, cluster_density,
                                            cluster_distribution, noise_amount, noise_width, central_point)
    elif cluster_shape == 4:
        points_sets = draw_polygon_skeleton(4, cluster_angle, cluster_area, cluster_density,
                                            cluster_distribution, noise_amount, noise_width, central_point)
    elif cluster_shape == 5:
        points_sets = draw_polygon_skeleton(5, cluster_angle, cluster_area, cluster_density,
                                            cluster_distribution, noise_amount, noise_width, central_point)
    elif cluster_shape == 6:
        points_sets = draw_polygon_skeleton(6, cluster_angle, cluster_area, cluster_density,
                                            cluster_distribution, noise_amount, noise_width, central_point)
    elif cluster_shape == 7:
        points_sets = draw_polygon_skeleton(7, cluster_angle, cluster_area, cluster_density,
                                            cluster_distribution, noise_amount, noise_width, central_point)
    elif cluster_shape == 8:
        points_sets = draw_starlike_skeleton(cluster_angle, cluster_area, cluster_density,
                                             cluster_distribution, noise_amount, noise_width, central_point)
    return points_sets


def random_sample_in_high_dim():
    """
    return ramdomly sampled point(9 dim)
    cluster_type, cluster_shape, cluster_angle,
    cluster_area, cluster_density, cluster_distribution,
    noise_amount, noise_width, central_point
    """
    cluster_type = 2  # regional cluster
    cluster_shape = random.randint(1, 7)  # different shapes
    cluster_angle = random.randint(1, 7)
    cluster_area = random.uniform(50000, 100000)
    cluster_density = random.uniform(0.3, 0.7)
    cluster_distribution = random.randint(1, 3)
    noise_amount = random.randint(1, 10)
    noise_width = random.uniform(10, 70)
    central_point = [random.uniform(150, 300), random.uniform(150, 300)]
    return cluster_type, cluster_shape, cluster_angle, cluster_area, cluster_density, \
           cluster_distribution,noise_amount, noise_width, central_point


if __name__ == '__main__':
    # cluster_type, cluster_shape, cluster_angle,
    # cluster_area, cluster_density, cluster_distribution,
    # noise_amount, noise_width, central_point
    # circle
    for i in range(3):
        points_sets = gen_region_cluster_skeleton(2, 1, 10, 100000, 0.5, 3, 5, 30, [250, 250])
    for i in range(3):
        points_sets = gen_region_cluster_skeleton(2, 1, 10, 100000, 0.5, 1, 5, 30, [250, 250])
    for i in range(3):
        points_sets = gen_region_cluster_skeleton(2, 1, 10, 100000, 0.5, 2, 5, 30, [250, 250])
    # oval
    for i in range(3):
        points_sets = gen_region_cluster_skeleton(2, 2, 10, 100000, 0.5, 3, 5, 30, [250, 250])
    for i in range(3):
        points_sets = gen_region_cluster_skeleton(2, 2, 10, 100000, 0.5, 1, 5, 30, [250, 250])
    for i in range(3):
        points_sets = gen_region_cluster_skeleton(2, 2, 10, 100000, 0.5, 2, 5, 30, [250, 250])
    # triangle
    for i in range(3):
        points_sets = gen_region_cluster_skeleton(2, 3, 10, 100000, 0.5, 3, 5, 30, [250, 250])
    for i in range(3):
        points_sets = gen_region_cluster_skeleton(2, 3, 10, 100000, 0.5, 1, 5, 30, [250, 250])
    for i in range(3):
        points_sets = gen_region_cluster_skeleton(2, 3, 10, 100000, 0.5, 2, 5, 30, [250, 250])
    # quadrangle
    for i in range(3):
        points_sets = gen_region_cluster_skeleton(2, 4, 10, 100000, 0.5, 3, 5, 30, [250, 250])
    for i in range(3):
        points_sets = gen_region_cluster_skeleton(2, 4, 10, 100000, 0.5, 1, 5, 30, [250, 250])
    for i in range(3):
        points_sets = gen_region_cluster_skeleton(2, 4, 10, 100000, 0.5, 2, 5, 30, [250, 250])
    # pentagon
    # for i in range(3):
    #     points_sets = gen_region_cluster_skeleton(2, 5, 10, 100000, 0.5, 3, 5, 30, [250, 250])
    # for i in range(3):
    #     points_sets = gen_region_cluster_skeleton(2, 5, 10, 100000, 0.5, 1, 5, 30, [250, 250])
    # for i in range(3):
    #     points_sets = gen_region_cluster_skeleton(2, 5, 10, 100000, 0.5, 2, 5, 30, [250, 250])
    # hexagon
    # for i in range(3):
    #     points_sets = gen_region_cluster_skeleton(2, 6, 10, 100000, 0.5, 3, 5, 30, [250, 250])
    # for i in range(3):
    #     points_sets = gen_region_cluster_skeleton(2, 6, 10, 100000, 0.5, 1, 5, 30, [250, 250])
    # for i in range(3):
    #     points_sets = gen_region_cluster_skeleton(2, 6, 10, 100000, 0.5, 2, 5, 30, [250, 250])
    # heptagon
    # for i in range(3):
    #     points_sets = gen_region_cluster_skeleton(2, 7, 10, 100000, 0.5, 3, 5, 30, [250, 250])
    # for i in range(3):
    #     points_sets = gen_region_cluster_skeleton(2, 7, 10, 100000, 0.5, 1, 5, 30, [250, 250])
    # for i in range(3):
    #     points_sets = gen_region_cluster_skeleton(2, 7, 10, 100000, 0.5, 2, 5, 30, [250, 250])
    # for i in range(5):
    #     args = random_sample_in_high_dim()
    #     points_sets = gen_region_cluster_skeleton(args[0], args[1], args[2], args[3], args[4],
    #                                               args[5], args[6], args[7], args[8])
    #     pxlist = []
    #     pylist = []
    #     tot = len(points_sets)
    #     for i in range(tot):
    #         pxlist.append(points_sets[0])
    #         pylist.append(points_sets[1])
    #     draw_skeleton(pxlist, pylist)
