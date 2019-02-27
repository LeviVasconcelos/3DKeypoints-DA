import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from utils.horn87 import horn87, RotMat
import ref
from prior_generator import compute_distances, compute_proportions, replicate_mask 

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
    self.priorStd = Std

    self.mask = replicate_mask(self.eyeJ.view(1,-1))
    self.weights = (1./(Std+self.eps))*self.mask

    self.priorMean = torch.autograd.Variable(Mean*self.mask,requires_grad=False)
    self.priorMean=self.priorMean.view(1,-1)

    # Init normalizer
    if std_weight:
	self.normalizer = torch.autograd.Variable(self.weights/((self.weights).sum()))
	self.normalizer=self.normalizer.view(1,-1)
    else:
	self.normalizer = 1./self.mask.sum()

    self.mask = torch.autograd.Variable(self.mask,requires_grad=False)


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
    else:
	print(norm + ' is not a valid norm value')
	exit(1)


  ######################
  #### FORWARD PASS ####
  ######################

  def forward(self, prediction):
    prediction = prediction.view(prediction.shape[0],self.J,-1)
    dists = compute_distances(prediction, eps=self.eps)
    props_unmasked = compute_proportions(dists, eps=10**(-6))
    props = props_unmasked*self.mask
    props = props.view(props.shape[0],-1)

    mse = self.norm(props-self.priorMean)

    return mse.mean()


  ##############################
  #### DEFINITION OF NORMS #####
  ##############################


  def frobenius(self,x):
	x = x.view(x.shape[0],self.J**2,-1)

	x_transposed = x.permute(0,2,1) # B x K x D
	xTx = torch.bmm(x_transposed,x)
	return (xTx*self.eyeJ2).sum(-1)*self.normalizer


  def l1(self,x):
	return torch.norm(x*self.normalizer,p=1,dim=-1)


  def l2(self,x):
	return torch.norm(x,p=2,dim=-1)*self.normalizer

  def weighted_l2(self,x):
   	diff = x.pow(2)*self.normalizer
   	mse = diff.sum(-1).pow(0.5)
	return mse

  def weighted_frobenius(self,x):
	x = x*self.normalizer
	x = x.view(x.shape[0],self.J**2,-1)

	x_transposed = x.permute(0,2,1) # B x K x D
	xTx = torch.bmm(x_transposed,x)
	return (xTx*self.eyeJ2).sum(-1)



######################
#### PRIOR LOADER ####
######################

def get_priors_from_file(path, device='cuda'):
	priors = np.load(path)
	mean = priors.mean(0)
	std = priors.std(0)
	norms = std/mean
	if device=='cuda':
		return torch.from_numpy(mean).cuda(), torch.from_numpy(norms).cuda()

	return torch.from_numpy(mean), torch.from_numpy(norms)




