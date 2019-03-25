import numpy as np
import cv2
import sys

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D

J = 10
edges = [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 5], 
         [4, 8], [5, 9], [3, 7], [2, 6]]
S = 224


def show2D(img, points, c):
  points = ((points.reshape(J, 3) + 0.5) * S).astype(np.int32)
  points[:, 0], points[:, 1] = points[:, 1].copy(), points[:, 0].copy()  
  for j in range(J):
    cv2.circle(img, (points[j, 0], points[j, 1]), 3, c, -1)
  print points
  for e in edges:
    cv2.line(img, (points[e[0], 0], points[e[0], 1]),
                  (points[e[1], 0], points[e[1], 1]), c, 2)
  return img


def show3D(ax, points, c = (255, 0, 0)):
    points = points.reshape(J, 3)
    print c, points
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

def load_priors(x):
	priors = np.load(x)
	mean = priors.mean(0).reshape(10,10,10,10)
	std = priors.std(0).reshape(10,10,10,10)
	norms = std/mean
	return mean, norms

def propagate_edge(means, norms, edge, edges, idx, vals, th=0.1):
	v = vals[idx]
	for i,e in enumerate(edges):
			if norms[edge[0],edge[1],e[0],e[1]]<th and (vals[i] is None):
				vals[i]=means[edge[0],edge[1],e[0],e[1]]*v
	return vals



def propagate_all(means, norms, edges,val_0 = 0.2):
	vals = {}
	for i,e in enumerate(edges):
		vals[i]=None
	vals[0]=val_0
	exist_none = True
	while exist_none:
		for i,e in enumerate(edges):
			if vals[i] is not None:
				vals = propagate_edge(means, norms, e, edges, i,vals)
		exist_none=False
		for i,e in enumerate(edges):
			if vals[i] is None:
				exist_none=True
				vals[i]=np.random.rand()
				break	
		
	return vals


def infer_points(edges, vals, point_0):
	coords = np.zeros((10,3))
	coords[0] = point_0
	coords[1] = point_0+np.array([vals[0],0, 0]) 
	coords[2] = point_0+np.array([0,vals[1], 0])
 
	coords[3] = coords[1]+np.array([0,vals[2], 0])
	coords[4] = coords[2]+np.array([0,0,vals[4]])

	coords[5] = coords[3]+np.array([0,0,vals[2]])
	coords[6] = coords[2]+np.array([0,vals[9],0])
	coords[7] = coords[3]+np.array([0,vals[8],0]) 

	coords[8] = coords[4]+np.array([0,vals[6],0])
	coords[9] = coords[5]+np.array([0,vals[7],0]) 

	return coords



def get_coords(x, point_0=[0,0,0]):
	means, norms = load_priors(x)
	vals = propagate_all(means, norms, edges)
	return infer_points(edges, vals, point_0)


path = sys.argv[1]
points=get_coords(path)
fig = plt.figure()
ax = fig.add_subplot((111),projection='3d')
show3D(ax, points, c = (255, 0, 0))
plt.show()
