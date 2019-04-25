import torch
import torch.nn as nn

# Computes the pairwise distances within x, where x has dimensions BxKxD, K=10 and D=3 in our case and B=batch-size

#### NOTE #####
# From the computations here defined, distance from a keypoint of another is just: dist[:,idx1, idx2]
# Extracting the proportion from 1 edge (idx1->idx2) to another (idx3->idx4) is just: prop[:,idx1*10+idx2, idx3*10+idx4], 
# where 10 is the number of keypoints 

def compute_distances(x, eps=10**(-6)):
    if len(x.shape)==2:
    	x=x.unsqueeze(0)
    if (torch.isnan(x).sum() > 0):
        print('X ALREADY WITH NANS TO COMPUTE DISTS')

    # Computes the squared norm of X
    x_squared=x.norm(p=2,dim=2).pow(2) # B x K

    
    x_squared_left = x_squared.unsqueeze(-1) # B x K x 1
    x_squared_right = x_squared.unsqueeze(1) # B x 1 x K

    x_transposed = x.permute(0,2,1) # B x K x D
    xxT = torch.bmm(x,x_transposed)
    dists = x_squared_left + x_squared_right - 2*xxT  +eps
    if (torch.isnan(dists).sum() > 0):
        print('Distances NaN')
    dists = (nn.functional.relu(dists)).pow(0.5)
    if (torch.isnan(dists).sum() > 0):
        print('Distances naN after Relu')
    if ((dists < 0).sum() > 0):
        print('NEGATIVE DISTANCES')
    return  dists # B x K x K


def replicate_mask(x): # x must be of dimension BxK
    # Computes the outer product of a vector by itself
    x_left = x.unsqueeze(-1) # B x K x 1
    x_right = x.unsqueeze(1) # B x 1 x K

    xxT = torch.bmm(x_left,x_right) # B x K x K

    return  xxT


def compute_proportions(x,eps=10**(-6)): # x has dimensions B x K x K
    x = x.view(x.shape[0],-1) # B x K^2
    numerator = x.unsqueeze(2) # B x K^2 x 1
    denominator = 1./(x.unsqueeze(1)+eps) # B x 1 x K^2
    mm = torch.bmm(numerator,denominator) # B x K^2 x K^2
    if (torch.isnan(mm).sum() > 0):
       print('NaN computing proportions...')
    assert(torch.isnan(mm).sum() < 1)
    return mm

def compute_masked_proportions(x,mask,eps=10**(-6)): # x has dimensions B x K x K
    x = x.view(x.shape[0],-1) # B x K^2
    numerator = x.unsqueeze(2)*mask # B x K^2 x 1
    denominator = 1./(x.unsqueeze(1)+eps) # B x 1 x K^2
    denominator = denominator + 10*mask
    mm = torch.bmm(numerator,denominator) # B x K^2 x K^2
    return mm


def check_dist(x,d):
	for i in range(10):
		for j in range(10):
			mdist = (x[0,i]-x[0,j]).norm()**2
			if (d[0,i,j]-mdist)>10**(-6):
				print(d[0:,i,j]-mdist)


def check_props(x,p, eps=10**(-6)):
	for i in range(100):
		for j in range(100):
			mp = x[0,i]/(x[0,j]+eps)
			if (p[0,i,j]-mp)>10**(-6):
					print(p[0,i,j]-mp)






