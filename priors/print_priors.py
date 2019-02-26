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
th = int(sys.argv[2])

means,norms = load_priors(path)

out=""
out+="---------- PRINTING CONFIDENT PROPS ----------\n"
caring_norms = []
caring_means = []
caring_edges = []

for i in range(J):
	for j in range(i+1,J):
		for k in range(J):
			for l in range(k+1,J):
				if (i!=k or j!=l):
					caring_norms.append(norms[i,j,k,l])
					caring_means.append(means[i,j,k,l])
					caring_edges.append("Props from edge [" + str(i)+","+str(j)+"] to edge [" + str(k)+","+str(l)+"]")

caring_norms=np.array(caring_norms)
indeces = np.argsort(caring_norms)

for i in indeces[:th]:
	out+=(caring_edges[i]+" has mean " + str(caring_means[i])+" and std at " + str(caring_norms[i])+"\n")


out+=("\n\n\n---------- PRINTING UNCONFIDENT PROPS ----------\n")
for i in indeces[-th:]:
	out+=(caring_edges[i]+" has mean " + str(caring_means[i])+" and std at " + str(caring_norms[i])+"\n")



with open(path.replace('.npy','-filtered.txt'), 'w') as f:
    f.write(out)
