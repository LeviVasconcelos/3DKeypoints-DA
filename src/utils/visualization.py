#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:43:28 2019

@author: levi
"""

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
import cv2
import ref
chair_edges = [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 5], 
         [4, 8], [5, 9], [3, 7], [2, 6]]
human_edges = [[13,25],  #right arm connection 
               [13,17], #left arm connection
               [0,12], #spine1
               [12,13], #spine2
               [13, 14], [14, 15], #head
               [0,1], [1,2], [2,3], [3,4], [4,5], #right leg
               [0,6], [6,7], [7,8], [8,9], [9,10], #left leg
               [17,18],[18,19],[19,20],[20,21],[21,22],[22,23], #left arm
               [25,26],[26,27],[27,28],[28,29],[29,30],[30,31] #right arm
               ]
S = 224

def human_show3D(ax, points, c = (255, 0, 0), edges = human_edges, J = ref.J):
      show3D(ax, points, c, edges, J)

def chair_show3D(ax, points, c = (255, 0, 0), edges = chair_edges, J = ref.J):
      show3D(ax, points, c, edges, J)
    
def show3D(ax, points, c = (255, 0, 0), edges = chair_edges, J = ref.J):
    points = points.reshape(J, 3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.set_xlabel('')
    #ax.set_ylabel('x') 
    #ax.set_zlabel('y')
    oo = 0.5
    xmax, ymax, zmax, xmin, ymin, zmin = oo, oo, oo, -oo, -oo, -oo
    #print c, points
    x, y, z = np.zeros((3, J))
    for j in range(J):
        x[j] = points[j, 0] 
        y[j] = points[j, 1] 
        z[j] = points[j, 2] 
    ax.scatter(x, y, z, c = c)
    l = 0
    for e in edges:
        ax.plot(x[e], y[e], z[e], c =c)
        l += ((z[e[0]] - z[e[1]]) ** 2 + (x[e[0]] - x[e[1]]) ** 2 + (y[e[0]] - y[e[1]]) ** 2) ** 0.5
    max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([zb], [xb], [yb], 'w')


def chair_show2D(img, points, c, edges = chair_edges, J = ref.J):
  img2 = img.copy()
  points = ((points.reshape(J, 3) + 0.5) * S).astype(np.int32)
  points[:, 0], points[:, 1] = points[:, 1].copy(), points[:, 0].copy()  
  for j in range(J):
    cv2.circle(img2, (points[j, 0], points[j, 1]), 3, c, -1)
  #print points
  for e in edges:
    cv2.line(img2, (points[e[0], 0], points[e[0], 1]),
                  (points[e[1], 0], points[e[1], 1]), c, 2)
  return img2

def human_show2D(img, points, c):
  points = points.astype(np.int32)
  points[:, 0], points[:, 1] = points[:, 1].copy(), points[:, 0].copy()  
  for j in range(J):
    x = points[j, 0] if points[j, 0] > 0 else -points[j, 0]
    y = points[j, 1] if points[j, 1] > 0 else -points[j, 1]
    cv2.circle(img, (y, x), 3, c, -1)
  return img