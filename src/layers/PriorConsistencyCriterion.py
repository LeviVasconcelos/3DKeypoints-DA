import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from utils.horn87 import horn87, RotMat
import ref
from prior_generator import compute_distances, compute_proportions, replicate_mask 
from scipy.misc import toimage



class PriorConsistencyCriterion(nn.Module):
  def __init__(self, Mean, Std, norm = 'frobenius', std_weight = False, J=10, eps = 10**(-4)):
    super(PriorConsistencyCriterion, self).__init__()

    self.J = J

    self.eps = eps

    self.eyeJ = Mean[:,:J,:J]*0.+1.
    self.eyeJ2 = Mean*0.

    for i in range(J**2):
	self.eyeJ2[0,i,i] = 1.

    for i in range(J):
		self.eyeJ[0,i,i]=0.

    self.eyeJ2 = torch.autograd.Variable(self.eyeJ2)

    self.mask = replicate_mask(self.eyeJ.view(1,-1))
    self.weights = (1./(Std+self.eps))*self.mask

    self.Var=torch.autograd.Variable((2*Std.pow(2)+self.eps).view(1,-1),requires_grad=False)
    self.priorMean = torch.autograd.Variable(Mean,requires_grad=False)
    self.priorMean=self.priorMean.view(1,-1)

    # Init normalizer
    if std_weight:
	self.normalizer = torch.autograd.Variable(self.weights/(self.weights.sum()))
	self.normalizer = self.normalizer.view(1,-1)
    else:
	self.normalizer = 1./self.mask.sum()

    self.mask = torch.autograd.Variable(self.mask.view(1,-1),requires_grad=False)


    # Init norm values
    if norm=='l2':
	if std_weight:
    		self.norm=self.weighted_l2
	else:
		self.norm=self.l2
    elif norm=='frobenius':
	if std_weight:
    		self.norm=self.weighted_frobenius
	else:
		self.norm=self.frobenius
    elif norm=='l1':
    	self.norm=self.l1
    elif norm=='likelihood':
    	self.norm=self.likelihood
    else:
	print(norm + ' is not a valid norm value')
	exit(1)


  ######################
  #### FORWARD PASS ####
  ######################

  def forward(self, prediction, logger=None, n_iter=0, plot=False):
    prediction = prediction.view(prediction.shape[0],self.J,-1)
    dists = compute_distances(prediction, eps=self.eps)
    props_unmasked = compute_proportions(dists, eps=self.eps)
    props = props_unmasked
    props = props.view(props.shape[0],-1)

    diff = (props-self.priorMean).pow(2)/self.Var
    mse = self.norm(props-self.priorMean)

    if logger is not None and plot:
	vis=diff.mean(0).view(1,self.J**2, -1)
	vis.detach()
	matrix = vis.data.abs().cpu().numpy()
	logger.add_image('Difference matrix', ((matrix-matrix.min())/matrix.max()*255).astype('uint8'), n_iter)
	
    return mse.mean()


  ##############################
  #### DEFINITION OF NORMS #####
  ##############################


  def frobenius(self,x):
	x = x.view(x.shape[0],self.J**2,-1)*self.mask

	x_transposed = x.permute(0,2,1) # B x K x D
	xTx = torch.bmm(x_transposed,x)
	return (xTx*self.eyeJ2).sum(-1).sum(-1)

  def l1(self,x):
	return torch.norm(x*self.normalizer,p=1,dim=-1)


  def l2(self,x):
	return torch.norm(x*self.mask,p=2,dim=-1)*self.normalizer

  def likelihood(self,x,eps=10**-8):
	likelihood = torch.exp(-x.pow(2)/self.Var)
	log_likelihood = -torch.log(likelihood+eps)*self.mask
	return log_likelihood.sum(-1)*self.normalizer

  def weighted_l2(self,x):
   	diff = x.pow(2)*self.normalizer*self.mask
   	mse = diff.sum(-1).pow(0.5)
	return mse

  def weighted_frobenius(self,x):
	normalizer = (self.normalizer+self.eps).pow(0.5)*self.eyeJ2.view(1,-1)
	x = x*normalizer
	x = x.view(x.shape[0],self.J**2,-1)

	x_transposed = x.permute(0,2,1) # B x K x D
	xTx = torch.bmm(x_transposed,x)
	return (xTx*self.eyeJ2).sum(-1).sum(-1)





class DistanceConsistencyCriterion(nn.Module):
  def __init__(self, Mean, Std, norm = 'frobenius', std_weight = False, J=10, eps = 10**(-4)):
    super(DistanceConsistencyCriterion, self).__init__()
    self.J = J



    self.eps = eps
    self.eyeJ = Mean*0.
    self.distMean = torch.autograd.Variable(Mean,requires_grad=False)
    for i in range(J):
	self.eyeJ[0,i,i] = 1.

    # Init norm values
    if norm=='l2':
	self.norm=self.l2

    elif norm=='frobenius':
	self.norm=self.frobenius

    elif norm=='l1':
    	self.norm=self.l1
    else:
	print(norm + ' is not a valid norm value')
	exit(1)
    self.eyeJ = torch.autograd.Variable(self.eyeJ)


    self.mask = torch.autograd.Variable(self.mask,requires_grad=False)



  ######################
  #### FORWARD PASS ####
  ######################

  def forward(self, prediction, logger=None, n_iter=0, plot=False):
    prediction = prediction.view(prediction.shape[0],self.J,-1)
    dists = compute_distances(prediction, eps=self.eps)

    diff = torch.abs(dists-self.distMean)
    mse = self.norm(dists-self.distMean)

    if logger is not None and plot:
	vis=diff.mean(0).view(1,self.J, -1)
	vis.detach()
	matrix = vis.data.abs().cpu().numpy()
	logger.add_image('Difference matrix', ((matrix-matrix.min())/matrix.max()*255).astype('uint8'), n_iter)
	
    return mse.mean()





  ##############################
  #### DEFINITION OF NORMS #####
  ##############################


  def frobenius(self,x):
	x_transposed = x.permute(0,2,1) # B x K x D
	xTx = torch.bmm(x_transposed,x)
	return (xTx*self.eyeJ).sum(-1).sum(-1)

  def l1(self,x):
	return torch.norm(x,p=1,dim=-1)


  def l2(self,x):
	return torch.norm(x,p=2,dim=-1)





class PriorToDistanceConsistencyCriterion(nn.Module):
  def __init__(self, Mean, Std, norm = 'frobenius', std_weight = False, J=10, eps = 10**(-4)):
    super(PriorToDistanceConsistencyCriterion, self).__init__()

    self.J = J
    self.weights = Std.view(1,self.J,self.J,self.J,self.J)
    self.eps = eps
    self.eyeJ2 = Mean*0.

    for i in range(J**2):
	self.eyeJ2[0,i,i] = 1.

    self.eyeJ2 = torch.autograd.Variable(self.eyeJ2)

    self.priorMean = torch.autograd.Variable(Mean,requires_grad=False)
    self.priorMean=self.priorMean.view(1,self.J, self.J, self.J,self.J)

    # Init normalizer
    if std_weight:
	self.normalizer = torch.autograd.Variable(self.weights/(self.weights.sum(-1).sum(-1).view(1,self.J,self.J,1,1)))
    else:
	self.normalizer = 1./self.J**2


    # Init norm values
    if norm=='l2':
	self.norm=self.l2
    elif norm=='frobenius':
	self.norm=self.frobenius
    elif norm=='l1':
    	self.norm=self.l1
    else:
	print(norm + ' is not a valid norm value')
	exit(1)


  ######################
  #### FORWARD PASS ####
  ######################

  def forward(self, prediction, logger=None, n_iter=0, plot=False):
    prediction = prediction.view(prediction.shape[0],self.J,-1)
    dists = compute_distances(prediction, eps=self.eps)
    gt_dists = self.compute_gt(dists)

    diff = dists-gt_dists
    mse = self.norm(dists-gt_dists)

    if logger is not None and plot:
	vis=diff.mean(0).view(1,self.J, -1)
	vis.detach()
	matrix = vis.data.abs().cpu().numpy()
	logger.add_image('Difference matrix', ((matrix-matrix.min())/matrix.max()*255).astype('uint8'), n_iter)
	
    return mse.mean()


  def compute_gt(self,x):
	tiled = x.view(x.shape[0],-1).repeat(1,self.J**2).view(x.shape[0],self.J,self.J,self.J,self.J)
	tiled.detach()
	return (self.priorMean*tiled*self.normalizer).sum(-1).sum(-1)

  ##############################
  #### DEFINITION OF NORMS #####
  ##############################


  def frobenius(self,x):
	x = x.view(x.shape[0],self.J**2,-1)

	x_transposed = x.permute(0,2,1) # B x K x D
	xTx = torch.bmm(x_transposed,x)
	return (xTx*self.eyeJ2).sum(-1).sum(-1)

  def l1(self,x):
	return torch.norm(x,p=1,dim=-1)/self.J**2


  def l2(self,x):
	return torch.norm(x,p=2,dim=-1)/self.J**2





######################
#### PRIOR LOADER ####
######################

def get_priors_from_file(path, device='cuda', eps=10**(-6)):
	priors = np.load(path)
	mean = priors.mean(0)
	std = priors.std(0)
	norms = mean/(std+eps)
	if device=='cuda':
		return torch.from_numpy(mean).cuda(), torch.from_numpy(norms).cuda()

	return torch.from_numpy(mean), torch.from_numpy(std)





