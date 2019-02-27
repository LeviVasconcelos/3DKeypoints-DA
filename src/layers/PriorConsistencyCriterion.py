import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from utils.horn87 import horn87, RotMat
import ref
from prior_generator import compute_distances, compute_proportions 

class PriorConsistencyCriterion(nn.Module):
  def __init__(self, Mean, Std, logger=None, J=10, eps = 10**(-4)):
    super(PriorConsistencyCriterion, self).__init__()

    self.J = J

    self.eps = eps

    self.priorMask = Mean*0.
    self.eye = Mean[:,:self.J,:self.J]*0.
    for i in range(J**2):
	for j in range(i+1,J**2):
		self.priorMask[0,i,j]=1.

    for i in range(J):
		self.eye[0,i,i]=1.

    self.normalizer = self.priorMask.sum()
    self.eye = torch.autograd.Variable(self.eye)
    self.priorMean = torch.autograd.Variable(Mean*self.priorMask,requires_grad=False)
    self.priorMask = torch.autograd.Variable(self.priorMask, requires_grad=False)
    self.priorStd = Std
    self.priorMean=self.priorMean.view(1,-1)


  def forward(self, prediction):
    prediction = prediction.view(prediction.shape[0],self.J,-1)
    dists = compute_distances(prediction, eps=self.eps)
    props = compute_proportions(dists, eps=10**(-6))
    props = props*self.priorMask
    props = props.view(props.shape[0],-1)


    mse = torch.norm(props-self.priorMean,p=2,dim=-1)/self.normalizer

    ls = mse.mean()
    return ls


def get_priors_from_file(path, device='cuda'):
	priors = np.load(path)
	mean = priors.mean(0)
	std = priors.std(0)
	norms = std/mean
	if device=='cuda':
		return torch.from_numpy(mean).cuda(), torch.from_numpy(norms).cuda()

	return torch.from_numpy(mean), torch.from_numpy(norms)
