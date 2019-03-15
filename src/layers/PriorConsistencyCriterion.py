import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from utils.horn87 import horn87, RotMat
import ref
from prior_generator import compute_distances, compute_proportions, replicate_mask 
from scipy.misc import toimage
from torchviz import make_dot
import math

###########################
#### LOSS DEFINITIONS #####
###########################
'''First loss, create synth gt through 
the current predictions and the priors'''

EDGES = [(0,1),(0,2),(1,3),(2,3),(2,4),(2,6),(3,5),(3,7),(4,5),(4,8),(5,9)]


class AbstractPriorLoss(nn.Module):
  def __init__(self, path, J=10, eps = 10**(-6), cuda=True, norm='l2', distances_refinement=None):
    super(AbstractPriorLoss, self).__init__()
    Mean,Std,DMean, DStd, Corr = get_priors_from_file(path)

    self.J = J
    self.eps = eps

    # Init priors holders
    self.priorMean = torch.autograd.Variable(Mean,requires_grad=False)
    self.priorMean=self.priorMean.view(1,self.J, self.J, self.J,self.J)

    self.priorStd = torch.autograd.Variable(Std,requires_grad=False)
    self.priorStd=self.priorStd.view(1,self.J, self.J, self.J,self.J)

    self.distMean = torch.autograd.Variable(DMean,requires_grad=False)
    self.distMean = self.distMean.view(1,self.J, self.J)

    self.distStd = torch.autograd.Variable(DStd,requires_grad=False)
    self.distStd=self.distStd.view(1,self.J, self.J)



    # Init possibly useful matrices
    self.upper_triangular = torch.FloatTensor(1,self.J,self.J).zero_() 


    self.eyeJ2 = torch.eye(J**2).unsqueeze(0).float()
    self.eyeJ2 = self.eyeJ2.view(1,self.J,self.J,self.J,self.J)

    self.eyeJ = torch.eye(J).unsqueeze(0).float()

    adjacency = torch.FloatTensor(self.J,self.J).zero_()
    mask_no_self_connections = torch.FloatTensor(1,self.J,self.J,self.J,self.J).zero_() + 1.
    self_keypoint_props =  torch.FloatTensor(1,self.J,self.J,self.J,self.J).zero_()

    for (i,j) in EDGES:
	adjacency[i,j]=1.

    adjacency = adjacency.view(1,1,1,self.J,self.J)

    for i in range(self.J):
            self_keypoint_props[0,i,i,i,i]=1.
	    for j in range(self.J):
		    if j>i:
			self.upper_triangular[0,i,j]=1.
		    for l in range(self.J):
			    for m in range(self.J):
				if i==j or l==m or (i==l and j==m) or (i==m and j==l):
					mask_no_self_connections[0,i,j,l,m]=0.0
					

    self.eyeJ = torch.autograd.Variable(self.eyeJ)
    self.eyeJ2 = torch.autograd.Variable(self.eyeJ2)
    
    self.upper_triangular = torch.autograd.Variable(self.upper_triangular,requires_grad=False)
    self.adjacency = torch.autograd.Variable(adjacency)
    self.mask_no_self_connections = torch.autograd.Variable(mask_no_self_connections)
    self.self_keypoint_props = torch.autograd.Variable(self_keypoint_props)

    if cuda:
	self.upper_triangular = self.upper_triangular.cuda()
	self.adjacency = self.adjacency.cuda()
	self.mask_no_self_connections = self.mask_no_self_connections.cuda()
	self.self_keypoint_props = self.self_keypoint_props.cuda()

	self.eyeJ2 = self.eyeJ2.cuda()
	self.eyeJ = self.eyeJ.cuda()

    if distances_refinement is None:
	print('Initializing a refiner as identity')
	self.refiner=(self.identity)
    else:
	print('Initializing a distances refiner')
	self.refiner=(self.refine_distances)
	factor = torch.exp((torch.abs(Corr))).view(1,self.J,self.J,self.J,self.J)
        factor = torch.autograd.Variable(factor.cuda())
	self.normalizer = factor*self.mask_no_self_connections + self.self_keypoint_props
	self.normalizer = self.normalizer/(self.normalizer.sum(-1).sum(-1).view(1,self.J,self.J,1,1))
	if cuda:
		self.normalizer = self.normalizer.cuda()

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
  def identity(self,x,props):
	return x

  def refine_distances(self, x,props):
	print('refining dists')
	tiled = x.view(x.shape[0],-1).repeat(1,self.J**2).view(x.shape[0],self.J,self.J,self.J,self.J)
	return (self.priorMean*tiled*self.normalizer).sum(-1).sum(-1)

  def compute_likelihood(self, x):
	return torch.exp(-torch.pow(x-self.priorMean,2)/(2*self.priorStd.pow(2)))

  def forward(self,x):
	pass







class PriorRegressionCriterion(AbstractPriorLoss):
  def __init__(self, path, J=10, eps = 10**(-6), cuda=True, norm='l2', distances_refinement=None, obj='props'):
    super(PriorRegressionCriterion, self).__init__(path, J, eps, cuda, norm, distances_refinement)

    if obj == 'props':
	self.forward=(self.forward_props)
    else:
	self.forward=(self.forward_dists)


  def forward_props(self, prediction, dt=None):
    prediction = prediction.view(prediction.shape[0],self.J,-1)
    dists = compute_distances(prediction, eps=self.eps)
    props = compute_proportions(dists, eps=self.eps).view(dists.shape[0],self.J,self.J,self.J,self.J)

    diff = (props-self.priorMean)
    mse = self.norm(diff)

    return mse


  def forward_dists(self, prediction, dt=None):
    prediction = prediction.view(prediction.shape[0],self.J,-1)
    dists = compute_distances(prediction, eps=self.eps)
    gt_dists=dt
    if gt_dists is None:
    	dists = compute_distances(prediction, eps=self.eps)
	props = compute_proportions(dists, eps=self.eps).view(dists.shape[0],self.J,self.J,self.J,self.J)
	gt_dists = self.refine_distances(props)*(1.-self.eyeJ)

    props = compute_proportions(dists, eps=self.eps).view(dists.shape[0],self.J,self.J,self.J,self.J)
    gt_dists = self.refiner(dists, props)

    diff = (dists-gt_dists)
    mse = self.norm(diff)

    return mse




	



##############################
#### Weighted MDS, simple ####
##############################

class PriorSMACOFCriterion(AbstractPriorLoss):
  def __init__(self, path, J=10, eps = 10**(-6), cuda=True, norm='l2', distances_refinement=None, iterate=False):
    super(PriorSMACOFCriterion, self).__init__(path, J, eps, cuda, norm, distances_refinement)

    self.eyeK = torch.eye(3).unsqueeze(0).float()
    self.dists_mask = torch.FloatTensor(1,self.J,self.J).zero_()


    for i in range(self.J):
	    for j in range(self.J):
		    if j>i:
			self.dists_mask[0,i,j]=1.

    self.eyeK = torch.autograd.Variable(self.eyeK)
    self.dists_mask = torch.autograd.Variable(self.dists_mask)

    if cuda:
	self.eyeK = self.eyeK.cuda()
	self.dists_mask = self.dists_mask.cuda()

    if iterate:
	self.forward = self.forward_iterative
    else:
	self.forward = self.forward_objective

  def forward_iterative(self, prediction, dt=None):
    prediction = prediction.view(prediction.shape[0],self.J,-1)
    dists=dt
    if dists is None:
    	dists = compute_distances(prediction, eps=self.eps)
 
    props = compute_proportions(dists, eps=self.eps).view(dists.shape[0],self.J,self.J,self.J,self.J)

    diff = (props-self.priorMean)
    mse = self.norm(diff)

    return mse.mean()



  def forward_objective(self, prediction, dt=None):

    prediction = prediction.view(prediction.shape[0],self.J,-1)
    gt_dists = dt

    if dt is None:
    	dists = compute_distances(prediction, eps=self.eps)
	props = compute_proportions(dists, eps=self.eps).view(dists.shape[0],self.J,self.J,self.J,self.J)
	gt_dists = self.refine_distances(dists,props)*(1.-self.eyeJ)

    w = torch.ones_like(gt_dists)					##### TODO ######

    error = self.compute_obj(prediction, gt_dists, w)/self.J

    return error



  def forward_iterative(self, prediction, dt=None):
    prediction = prediction.view(prediction.shape[0],self.J,-1)
    gt_dists = dt

    if dt is None:
    	dists = compute_distances(prediction, eps=self.eps)
	props = compute_proportions(dists, eps=self.eps).view(dists.shape[0],self.J,self.J,self.J,self.J)
	gt_dists = self.refine_distances(dists, props)*(1.-self.eyeJ)

    w = torch.ones_like(gt_dists)#None #torch.ones_like(gt_dists)					##### TODO ######

    X = self.iterate(prediction, gt_dists, w).detach()
    error = self.norm(X-prediction)/self.J

    return error


  def compute_obj(self,x,delta,w):
	d = compute_distances(x,self.eps)

	#### Compute first term: \sum_{i<j} w_{ij} \delta_{ij}^2 ####
	delta = delta.view(delta.shape[0],self.J, self.J)
	first_term = (delta.pow(2)*self.dists_mask).sum(-1).sum(-1)	

	#### Compute the second term: \sum_{i<j} w_{ij} d^2_{ij}(X) = trace(X'VX)####

	V = self.compute_V(x,w)

        X = x.view(x.shape[0],self.J,-1)
	T = x.permute(0,2,1)
	TV = torch.bmm(T,V)
	TVX = torch.bmm(TV,X)

	second_term = self.trace(TVX)

	#### Compute the third term: -2 \sum_{i<j} w_{ij}\delta_{i,j} d_{ij}(X) = trace(X'B(Z)Z)####
	
	B = self.compute_B(x,d,w,delta)
	Z = X			# TODO Change/Iterate???
	TB = torch.bmm(T,B)
	TBZ = torch.bmm(TB,Z)
	
	third_term = -2*self.trace(TBZ)
	
	return first_term + second_term + third_term


  def iterate(self,x,delta, w=None, iters=10, use_w=False):
	delta = delta.view(delta.shape[0],self.J, self.J)

	#### Compute the second term: \sum_{i<j} w_{ij} d^2_{ij}(X) = trace(X'VX)####
        X = x.view(x.shape[0],self.J,-1)
	T = x.permute(0,2,1)

	for i in range(iters):

		d = compute_distances(X, eps=self.eps)
		B = self.compute_B(x,d,w,delta)
		if use_w:
			V = self.compute_V(x,w) ##### TODO
			X = 1./self.J*torch.bmm(B,X) ##### TODO
		else:
			X = 1./self.J*torch.bmm(B,X)  
			
	
	return X


  def compute_V(self, x,w):
	V_ij = -w*(1.-self.eyeJ)
	V_ii = -V_ij.sum(-1).unsqueeze(-1)*(self.eyeJ)

	V = V_ij + V_ii

	return V

  def compute_B(self, x,d,w,delta):
	mask = (d.clone()==0).float()
	d_masked = d + mask	# Here the mask is applied to avoid divisions by zero
	b_0 = -(w*delta)/d_masked
	B_ij = b_0 * (1.-mask)
	B_ii = - B_ij.sum(-1).view(d.shape[0],self.J,1)
	B_ii = B_ii*self.eyeJ

	B = B_ij + B_ii

	return B

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

	mean_d = dists.mean(0)
	std_d = dists.std(0)

	mean_d = torch.from_numpy(mean_d).float()
	std_d = torch.from_numpy(std_d).float()

	correlation = torch.from_numpy(correlation).float()

	if device=='cuda':
		return mean.cuda(), std.cuda(), mean_d.cuda(), std_d.cuda(),correlation#.cuda()

	return mean, std, mean_d, std_d, correlation



###############
#### NORMS ####
###############


def l2(x,w=1.):
	x=x.view(x.shape[0],-1)
	return (w*(x.pow(2))).sum(-1)

def frobenius(x):
	assert len(x.shape)==3

	x_transposed = x.permute(0,2,1) # B x K x D
	xTx = torch.bmm(x_transposed,x)
	eye = torch.autograd.Variable(torch.eye(x.shape[1]).unsqueeze(0).float()).cuda()

	return (xTx*eye).sum(-1).sum(-1)


def l1(x,w=1):
	x=x.view(x.shape[0],-1)
	return torch.norm(x*w,p=1,dim=-1)



