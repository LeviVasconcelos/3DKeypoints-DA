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

    self.mask = torch.autograd.Variable(self.mask.view(1,-1)*0.+1,requires_grad=False)


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
	log_likelihhod=log_likelihood.view(log_likelihood.shape[0],-1)
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




class SelectedDistanceConsistencyCriterion(nn.Module):
  def __init__(self, Mean, Std, norm = 'frobenius', std_weight = False, J=10, eps = 10**(-4)):
    super(SelectedDistanceConsistencyCriterion, self).__init__()
    self.J = J

    self.eps = eps

    self.mask = [[0,0,1,0,0,0,1,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,1,0,1,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,1,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0]]



    self.mask = torch.autograd.Variable(torch.FloatTensor(self.mask).view(1,self.J,self.J).cuda(),requires_grad=False)

    index = torch.arange(1,self.J+1).long()
    index[-1]=0
    self.index = torch.autograd.Variable(index.cuda(),requires_grad=False)


  ######################
  #### FORWARD PASS ####
  ######################

  def forward(self, prediction, logger=None, n_iter=0, plot=False):

    prediction = prediction.view(prediction.shape[0],self.J,-1)
    dists = compute_distances(prediction, eps=self.eps)

	    
    transformed = torch.index_select(torch.index_select(dists,1,self.index),2,self.index)
    
    diff = torch.abs(dists-transformed)
    mse = ((dists-transformed).pow(2)*self.mask).view(dists.shape[0],-1).sum(-1)
	
    return mse.mean()


class FullSelectedDistanceConsistencyCriterion(nn.Module):
  def __init__(self, Mean, Std, norm = 'frobenius', std_weight = False, J=10, eps = 10**(-4)):
    super(FullSelectedDistanceConsistencyCriterion, self).__init__()
    self.J = J

    self.eps = eps

    self.mask_parallel = [[0,0,1,0,0,0,1,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,1,0,1,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,1,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0]]

    self.mask_parallel = torch.autograd.Variable(torch.FloatTensor(self.mask_parallel).view(1,self.J,self.J).cuda(),requires_grad=False)

    self.mask_diagonal = [[0,0,0,1,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,1,0,1,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0]]

    self.mask_diagonal = torch.autograd.Variable(torch.FloatTensor(self.mask_diagonal).view(1,self.J,self.J).cuda(),requires_grad=False)


    self.mask_2diagonal = [[0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,1,0],
                 [0,0,0,0,0,0,0,0,0,1],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0]]

    self.mask_2diagonal = torch.autograd.Variable(torch.FloatTensor(self.mask_2diagonal).view(1,self.J,self.J).cuda(),requires_grad=False)

    index = torch.arange(1,self.J+1).long()
    index[-1]=0
    self.index_n1 = torch.autograd.Variable(index.cuda(),requires_grad=False)

    index = torch.arange(-1,self.J-1).long()
    index[0]=self.J-1
    self.index_p1 = torch.autograd.Variable(index.cuda(),requires_grad=False)


    index = torch.arange(2,self.J+2).long()
    index[-1]=1
    index[-2]=0
    self.index_n2 = torch.autograd.Variable(index.cuda(),requires_grad=False)

    index = torch.arange(-2,self.J-2).long()
    index[0]=self.J-2
    index[1]=self.J-1
    self.index_p2 = torch.autograd.Variable(index.cuda(),requires_grad=False)

  ######################
  #### FORWARD PASS ####
  ######################

  def forward(self, prediction, logger=None, n_iter=0, plot=False):

    prediction = prediction.view(prediction.shape[0],self.J,-1)
    dists = compute_distances(prediction, eps=self.eps)

	    
    transformed_parallel = torch.index_select(torch.index_select(dists,1,self.index_n1),2,self.index_n1)
    transformed_diagonal = torch.index_select(torch.index_select(dists,1,self.index_n1),2,self.index_p1)
    transformed_2diagonal = torch.index_select(torch.index_select(dists,1,self.index_n2),2,self.index_p2)
    
    mse_parallel = ((dists-transformed_parallel).pow(2)*self.mask_parallel).view(dists.shape[0],-1).sum(-1)
    mse_diagonal = ((dists-transformed_diagonal).pow(2)*self.mask_diagonal).view(dists.shape[0],-1).sum(-1)
    mse_2diagonal = ((dists-transformed_2diagonal).pow(2)*self.mask_2diagonal).view(dists.shape[0],-1).sum(-1)

    mse = mse_parallel#+mse_diagonal#+mse_2diagonal

    return mse.mean()







class PriorToDistanceConsistencyCriterion(nn.Module):
  def __init__(self, Mean, Std, norm = 'frobenius', std_weight = False, J=10, eps = 10**(-4), cuda=True):
    super(PriorToDistanceConsistencyCriterion, self).__init__()

    self.J = J
    self.eps = eps

    self.eyeJ2 = torch.FloatTensor(J**2,J**2).zero_() 
    

    self.eyeJ2 = self.eyeJ2.view(1,self.J,self.J,self.J,self.J)

    self.weight_masked =  torch.FloatTensor(1,self.J,self.J,self.J,self.J).zero_()
    if cuda:
	self.eyeJ2 = self.eyeJ2.cuda()
	self.weight_masked = self.weight_masked.cuda()

    for i in range(self.J):
            self.weight_masked[0,i,i,i,i]=1.
	    for j in range(self.J):
		    for l in range(self.J):
			    for m in range(self.J):
				if i==j or l==m or (i==l and j==m) or (i==m and j==l):
					self.eyeJ2[0,i,j,l,m]=1.0
					



    self.eyeJ2 = torch.autograd.Variable(self.eyeJ2)
    self.weight_masked = torch.autograd.Variable(self.weight_masked)

    self.priorMean = torch.autograd.Variable(Mean,requires_grad=False)
    self.priorMean=self.priorMean.view(1,self.J, self.J, self.J,self.J)

    self.priorStd = torch.autograd.Variable(Std,requires_grad=False)
    self.priorStd=self.priorStd.view(1,self.J, self.J, self.J,self.J)

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
    props = compute_proportions(dists, eps=self.eps).view(dists.shape[0],self.J,self.J,self.J,self.J)

    self.weights = self.compute_likelihood(props)*(1.-self.eyeJ2) + self.weight_masked
    self.normalizer = self.weights/(self.weights.sum(-1).sum(-1).view(dists.shape[0],self.J,self.J,1,1))

    gt_dists = self.compute_gt(dists)

    diff = (dists-gt_dists)
    mse = self.norm(dists-gt_dists)

    if logger is not None and plot:
	vis=diff.mean(0).view(1,self.J, -1)
	vis.detach()
	matrix = vis.data.abs().cpu().numpy()
	logger.add_image('Difference matrix', ((matrix-matrix.min())/matrix.max()*255).astype('uint8'), n_iter)
	
    return mse.mean()



  def compute_likelihood(self, x):
	return torch.exp(-torch.pow(x-self.priorMean,2)/self.priorStd.pow(2))

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
	x=x.view(x.shape[0],-1)
	return torch.norm(x,p=1,dim=-1)/self.J**2


  def l2(self,x):
	x=x.view(x.shape[0],-1)
	return x.pow(2).sum(-1)








class PriorToDistanceMDS(nn.Module):
  def __init__(self, Mean, Std, norm = 'frobenius', std_weight = False, J=10, eps = 10**(-4), cuda=True):
    super(PriorToDistanceMDS, self).__init__()

    self.J = J
    self.eps = eps

    mask_props = torch.FloatTensor(1,self.J,self.J,self.J,self.J).zero_() 
    mask_props_self =  torch.FloatTensor(1,self.J,self.J,self.J,self.J).zero_()


    for i in range(self.J):
            mask_props_self[0,i,i,i,i]=1.
	    for j in range(self.J):
		    for l in range(self.J):
			    for m in range(self.J):
				if i==j or l==m or (i==l and j==m) or (i==m and j==l):
					mask_props[0,i,j,l,m]=1.0
					


    self.normalizer = (1-mask_props) + mask_props_self

    self.normalizer = self.normalizer/(self.normalizer.sum(-1).sum(-1).view(1,self.J,self.J,1,1))

    self.normalizer = torch.autograd.Variable(self.normalizer)
    self.mask_props = torch.autograd.Variable(mask_props)
    self.mask_props_self = torch.autograd.Variable(mask_props_self)

    if cuda:
	self.normalizer = self.normalizer.cuda()
	self.mask_props = self.mask_props.cuda()
	self.mask_props_self = self.mask_props_self.cuda()


    self.priorMean = torch.autograd.Variable(Mean,requires_grad=False)
    self.priorMean=self.priorMean.view(1,self.J, self.J, self.J,self.J)

    self.priorStd = torch.autograd.Variable(Std,requires_grad=False)
    self.priorStd=self.priorStd.view(1,self.J, self.J, self.J,self.J)

    # Init norm values
    self.norm=self.l2


  ######################
  #### FORWARD PASS ####
  ######################

  def forward(self, prediction, logger=None, n_iter=0, plot=False):
    prediction = prediction.view(prediction.shape[0],self.J,-1)
    dists = compute_distances(prediction, eps=self.eps)
    props = compute_proportions(dists, eps=self.eps).view(dists.shape[0],self.J,self.J,self.J,self.J)

    w = (torch.log(self.compute_likelihood(props)+self.eps)*self.mask_props).sum(-1).sum(-1)
    gt_dists = self.compute_gt(dists)

    w = w.view(w.shape[0],-1)
    w = w/w.sum(-1).view(-1,1)

    mse = self.norm(dists-gt_dists,w)

    return mse.mean()



  def compute_likelihood(self, x):
	return torch.exp(-torch.pow(x-self.priorMean,2)/(2*self.priorStd.pow(2)))

  def compute_gt(self,x):
	tiled = x.view(x.shape[0],-1).repeat(1,self.J**2).view(x.shape[0],self.J,self.J,self.J,self.J)
	tiled.detach()
	return (self.priorMean*tiled*self.normalizer).sum(-1).sum(-1)

  ##############################
  #### DEFINITION OF NORMS #####
  ##############################

  def l2(self,x,w):
	x=x.view(x.shape[0],-1)
	return (w*(x.pow(2))).sum(-1)




class PriorToDistanceSMACOF(nn.Module):
  def __init__(self, Mean, Std, norm = 'frobenius', std_weight = False, J=10, eps = 10**(-4), cuda=True):
    super(PriorToDistanceSMACOF, self).__init__()

    self.J = J
    self.eps = eps

    self.eyeJ = torch.eye(self.J).unsqueeze(0).float()
    self.eyeK = torch.eye(3).unsqueeze(0).float()

    mask_props = torch.FloatTensor(1,self.J,self.J,self.J,self.J).zero_() 
    mask_props_self =  torch.FloatTensor(1,self.J,self.J,self.J,self.J).zero_()
    self.dists_mask = torch.FloatTensor(1,self.J,self.J).zero_() 

    for i in range(self.J):
	    mask_props_self[0,i,i,i,i]=1.
	    for j in range(self.J):
		    if j>i:
			self.dists_mask[0,i,j]=1.
		    for l in range(self.J):
			    for m in range(self.J):
				if i==j or l==m or (i==l and j==m) or (i==m and j==l):
					mask_props[0,i,j,l,m]=1.0


    self.normalizer = (1-mask_props) + mask_props_self				
    self.normalizer = self.normalizer/(self.normalizer.sum(-1).sum(-1).view(1,self.J,self.J,1,1))

    self.normalizer = torch.autograd.Variable(self.normalizer)
    self.mask_props = torch.autograd.Variable(mask_props)
    self.eyeJ = torch.autograd.Variable(self.eyeJ)
    self.eyeK = torch.autograd.Variable(self.eyeK)
    self.dists_mask = torch.autograd.Variable(self.dists_mask)

    if cuda:
	self.normalizer = self.normalizer.cuda()
	self.mask_props = self.mask_props.cuda()
	self.eyeJ = self.eyeJ.cuda()
	self.eyeK = self.eyeK.cuda()
	self.dists_mask = self.dists_mask.cuda()

    self.priorMean = torch.autograd.Variable(Mean,requires_grad=False)
    self.priorMean=self.priorMean.view(1,self.J, self.J, self.J,self.J)

    self.priorStd = torch.autograd.Variable(Std,requires_grad=False)
    self.priorStd=self.priorStd.view(1,self.J, self.J, self.J,self.J)



  ######################
  #### FORWARD PASS ####
  ######################

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
	mask = (d==0).float().detach()
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

  def compute_likelihood(self, x):
	return torch.exp(-torch.pow(x-self.priorMean,2)/(2*self.priorStd.pow(2)))

  def compute_gt(self,x):
	tiled = x.view(x.shape[0],-1).repeat(1,self.J**2).view(x.shape[0],self.J,self.J,self.J,self.J)
	tiled.detach()
	return (self.priorMean*tiled*self.normalizer).sum(-1).sum(-1)




######################
#### PRIOR LOADER ####
######################

def get_priors_from_file(path, device='cuda', eps=10**(-6)):
	priors = np.load(path)

	correlation = np.corrcoef(priors.reshape(-1,priors.shape[0]))
	mean = priors.mean(0)
	std = priors.std(0)

	mean = torch.from_numpy(mean).float()
	std = torch.from_numpy(std).float()

	correlation = torch.from_numpy(correlation).float()

	if device=='cuda':
		return mean.cuda(), std.cuda(), correlation.cuda()

	return mean, std, correlation





