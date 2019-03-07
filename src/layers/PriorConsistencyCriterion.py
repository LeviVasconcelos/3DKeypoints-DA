import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from utils.horn87 import horn87, RotMat
import ref
from prior_generator import compute_distances, compute_proportions, replicate_mask 
from scipy.misc import toimage
from torchviz import make_dot

###########################
#### LOSS DEFINITIONS #####
###########################
'''First loss, create synth gt through 
the current predictions and the priors'''

class PriorToDistanceConsistencyCriterion(nn.Module):
  def __init__(self, Mean, Std, norm = 'l2', std_weight = False, J=10, eps = 10**(-4), cuda=True):
    super(PriorToDistanceConsistencyCriterion, self).__init__()

    self.J = J
    self.eps = eps

    self.eyeJ2 = torch.eye(J**2).unsqueeze(0).float()
    self.eyeJ2 = self.eyeJ2.view(1,self.J,self.J,self.J,self.J)

    self.eyeJ = torch.eye(J).unsqueeze(0).float()

    mask_props = torch.FloatTensor(1,self.J,self.J,self.J,self.J).zero_() 
    mask_props_self =  torch.FloatTensor(1,self.J,self.J,self.J,self.J).zero_()


    for i in range(self.J):
            mask_props_self[0,i,i,i,i]=1.
	    for j in range(self.J):
		    for l in range(self.J):
			    for m in range(self.J):
				if i==j or l==m or (i==l and j==m) or (i==m and j==l):
					mask_props[0,i,j,l,m]=1.0
					




    self.eyeJ2 = torch.autograd.Variable(self.eyeJ2)
    self.mask_props = torch.autograd.Variable(mask_props)
    self.mask_props_self = torch.autograd.Variable(mask_props_self)

    self.priorMean = torch.autograd.Variable(Mean,requires_grad=False)
    self.priorMean=self.priorMean.view(1,self.J, self.J, self.J,self.J)

    self.priorStd = torch.autograd.Variable(Std,requires_grad=False)
    self.priorStd=self.priorStd.view(1,self.J, self.J, self.J,self.J)

    if cuda:
	self.mask_props = self.mask_props.cuda()
	self.mask_props_self = self.mask_props_self.cuda()
	self.eyeJ2 = self.eyeJ2.cuda()
	self.eyeJ = self.eyeJ.cuda()

    # Init norm values
    if norm=='l2':
	self.norm=l2
    elif norm=='frobenius':
	self.norm=frobenius
    elif norm=='l1':
    	self.norm=l1
    else:
	print(norm + ' is not a valid norm value')
	exit(1)


  def forward(self, prediction, logger=None, n_iter=0, plot=False):
    prediction = prediction.view(prediction.shape[0],self.J,-1)
    dists = compute_distances(prediction, eps=self.eps)
    props = compute_proportions(dists, eps=self.eps).view(dists.shape[0],self.J,self.J,self.J,self.J)

    weights = self.compute_likelihood(props)*(1.-self.mask_props) + self.mask_props_self
    self.normalizer = weights/(weights.sum(-1).sum(-1).view(dists.shape[0],self.J,self.J,1,1))

    gt_dists = self.compute_gt(dists)

    diff = (dists-gt_dists)
    mse = self.norm(dists-gt_dists)

    return mse.mean()


  def compute_likelihood(self, x):
	return torch.exp(-torch.pow(x-self.priorMean,2)/(2*self.priorStd.pow(2)))


  def compute_gt(self,x):
	tiled = x.view(x.shape[0],-1).repeat(1,self.J**2).view(x.shape[0],self.J,self.J,self.J,self.J)
	return (self.priorMean*tiled*self.normalizer).sum(-1).sum(-1)






##############################
#### Weighted MDS, simple ####
##############################

class PriorToDistanceMDS(PriorToDistanceConsistencyCriterion):
  def __init__(self, Mean, Std, norm = 'l2', std_weight = False, J=10, eps = 10**(-4), cuda=True):
    super(PriorToDistanceMDS, self).__init__(Mean, Std, norm, std_weight, J, eps, cuda)

    self.normalizer = (1-self.mask_props) + self.mask_props_self
    self.normalizer = self.normalizer/(self.normalizer.sum(-1).sum(-1).view(1,self.J,self.J,1,1))

    if cuda:
	self.normalizer = self.normalizer.cuda()

  def forward(self, prediction, logger=None, n_iter=0, plot=False):
    prediction = prediction.view(prediction.shape[0],self.J,-1)
    dists = compute_distances(prediction, eps=self.eps)
    props = compute_proportions(dists, eps=self.eps).view(dists.shape[0],self.J,self.J,self.J,self.J)

    w = (torch.log(self.compute_likelihood(props)+self.eps)*self.mask_props).sum(-1).sum(-1)
    gt_dists = self.compute_gt(dists)

    w = w.view(w.shape[0],-1)
    w = w/w.sum(-1).view(-1,1)

    mse = self.norm(dists-gt_dists,w)

    #if logger is not None:
    #	g = make_dot(mse)
    #	with open('graph_mds_detached.txt', 'w') as f:
    #		f.write(str(g))
    #	exit(1)
    return mse.mean()




##############################
#### Weighted MDS, SMACOF ####
##############################
class PriorToDistanceSMACOF(PriorToDistanceConsistencyCriterion):
  def __init__(self, Mean, Std, norm = 'frobenius', std_weight = False, J=10, eps = 10**(-4), cuda=True):
    super(PriorToDistanceSMACOF, self).__init__(Mean, Std, norm, std_weight, J, eps, cuda)

    self.eyeK = torch.eye(3).unsqueeze(0).float()
    self.dists_mask = torch.FloatTensor(1,self.J,self.J).zero_() 

    self.normalizer = (1-self.mask_props) + self.mask_props_self				
    self.normalizer = self.normalizer/(self.normalizer.sum(-1).sum(-1).view(1,self.J,self.J,1,1))

    self.eyeK = torch.autograd.Variable(self.eyeK)
    self.dists_mask = torch.autograd.Variable(self.dists_mask)

    if cuda:
	self.normalizer = self.normalizer.cuda()
	self.eyeK = self.eyeK.cuda()
	self.dists_mask = self.dists_mask.cuda()


  def forward(self, prediction, logger=None, n_iter=0, plot=False):
    prediction = prediction.view(prediction.shape[0],self.J,-1)
    dists = compute_distances(prediction, eps=self.eps)
    props = compute_proportions(dists, eps=self.eps).view(dists.shape[0],self.J,self.J,self.J,self.J)

    w = (torch.log(self.compute_likelihood(props)+self.eps)*self.mask_props).sum(-1).sum(-1)
    gt_dists = self.compute_gt(dists)

    w = w.view(w.shape[0],-1)
    w = w/w.sum(-1).view(-1,1)		# TODO Compute in another way????

    error = self.compute_smacof(prediction, dists, w.view(w.shape[0],self.J,self.J), gt_dists)

    return error.mean()




  def compute_smacof(self,x,d,w,delta):

	#### Compute first term: \sum_{i<j} w_{ij} \delta_{ij}^2 ####
	delta = delta.view(delta.shape[0],self.J, self.J)
	first_term = (delta.pow(2)*w*self.dists_mask).sum(-1).sum(-1)	

	#### Compute the second term: \sum_{i<j} w_{ij} d^2_{ij}(X) = trace(X'VX)####
	V_ij = -w
	V_ii = w.sum(-1).view(w.shape[0],self.J, 1)
	V_ii = V_ii*self.eyeJ

	V = V_ij * (1.-self.eyeJ) + V_ii

        X = x.view(x.shape[0],self.J,-1)
	T = x.permute(0,2,1)
	TV = torch.bmm(T,V)
	TVX = torch.bmm(TV,X)

	second_term = self.trace(TVX)

	#### Compute the third term: -2 \sum_{i<j} w_{ij}\delta_{i,j} d_{ij}(X) = trace(X'B(Z)Z)####
	mask = (d.clone()==0).float()
	d_masked = d + mask	# Here the mask is applied to avoid divisions by zero
	b_0 = -(w*delta)/d_masked
	B_ij = b_0 * (1.-mask)
	B_ii = - B_ij.sum(-1).view(d.shape[0],self.J,1)
	B_ii = B_ii*self.eyeJ

	B = B_ij + B_ii
	
	Z = X			# TODO Change/Iterate???
	TB = torch.bmm(T,B)
	TBZ = torch.bmm(TB,Z)
	
	third_term = -2*self.trace(TBZ)
	
	return first_term + second_term + third_term
	
  def trace(self,x):
	return (x*self.eyeK).sum(-1).sum(-1)




######################
#### PRIOR LOADER ####
######################

def get_priors_from_file(path, device='cuda', eps=10**(-6)):
	priors = np.load(path)
	dists = np.load(path.replace('props','distances'))

	correlation = np.corrcoef(dists.reshape(-1,priors.shape[0]))
	mean = priors.mean(0)
	std = priors.std(0)

	mean = torch.from_numpy(mean).float()
	std = torch.from_numpy(std).float()

	correlation = torch.from_numpy(correlation).float()

	if device=='cuda':
		return mean.cuda(), std.cuda(), correlation#.cuda()

	return mean, std, correlation



###############
#### NORMS ####
###############

def l2(x,w=1.):
	x=x.view(x.shape[0],-1)
	return (w*(x.pow(2))).sum(-1)

def frobenius(x):
	assert len(x.shape)==2

	x_transposed = x.permute(0,2,1) # B x K x D
	xTx = torch.bmm(x_transposed,x)
	eye = torch.autograd.Variable(torch.eye(x.shape[1]).unsqueeze(0).float()).cuda()

	return (xTx*eye).sum(-1).sum(-1)


def l1(x,w=1):
	return torch.norm(x*w,p=1,dim=-1)




