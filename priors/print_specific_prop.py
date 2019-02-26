import numpy as np
import cv2
import sys

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D

J = 10
edges = [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 5], 
         [4, 8], [5, 9], [3, 7], [2, 6]]

def load_priors(x):
	priors = np.load(x)
	mean = priors.mean(0).reshape(10,10,10,10)
	std = priors.std(0).reshape(10,10,10,10)
	norms = std/mean*100
	return mean, norms


path = sys.argv[1]
edges = sys.argv[2].split(',')

edges =[int(e) for e in edges]

means,norms = load_priors(path)
print(means[edges[0],edges[1],edges[2],edges[3]],norm[edges[0],edges[1],edges[2],edges[3]])

