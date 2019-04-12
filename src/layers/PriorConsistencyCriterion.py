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
import os

###########################
#### LOSS DEFINITIONS #####
###########################
'''First loss, create synth gt through 
the current predictions and the priors'''

EDGES = [(0,1),(0,2),(1,3),(2,3),(2,4),(2,6),(3,5),(3,7),(4,5),(4,8),(5,9)]


class AbstractPriorLoss(nn.Module):
  def __init__(self, path, J=10, eps = 10**(-6), device='cuda', norm='l2', distances_refinement=None):
    super(AbstractPriorLoss, self).__init__()
    #Mean,Std,DMean, DStd, Corr = get_priors_from_file(path)
    Mean, Std, DMean, DStd = load_priors_from_file(path)

    self.J = J
    self.eps = eps

    # Init priors holders
    self.priorMean = Mean.view(1,self.J, self.J, self.J,self.J)
    self.priorStd = Std.view(1,self.J, self.J, self.J,self.J)
    self.distMean = DMean.view(1,self.J, self.J)
    self.distStd = DStd.view(1,self.J, self.J)



    # Init possibly useful matrices
    self.upper_triangular = torch.FloatTensor(1,self.J**2,self.J**2).zero_() 


    self.eyeJ2 = torch.eye(J**2).unsqueeze(0).float()
    self.eyeJ2 = self.eyeJ2.view(1,self.J,self.J,self.J,self.J)

    self.eyeJ = torch.eye(J).unsqueeze(0).float()

    adjacency = torch.FloatTensor(self.J,self.J).zero_()
    mask_no_self_connections = torch.FloatTensor(1,self.J,self.J,self.J,self.J).zero_() + 1.
    self_keypoint_props =  torch.FloatTensor(1,self.J,self.J,self.J,self.J).zero_()

    for (i,j) in EDGES:
      adjacency[i,j]=1.

    adjacency = adjacency.view(1,1,1,self.J,self.J)
    for i in range(self.upper_triangular.shape[1]):
          for j in range(i+1,self.upper_triangular.shape[2]):
                self.upper_triangular[0,i,j]=1.

    for i in range(self.J):
            self_keypoint_props[0,i,i,i,i]=1.
            for j in range(self.J):
                  for l in range(self.J):
                        for m in range(self.J):
                              if i==j or l==m or (i==l and j==m) or (i==m and j==l):
                                    mask_no_self_connections[0,i,j,l,m]=0.0

    self.upper_triangular = self.upper_triangular.view(1,self.J,self.J,self.J,self.J).to(device)
    self.adjacency = adjacency
    self.mask_no_self_connections = mask_no_self_connections.to(device)
    self.self_keypoint_props = self_keypoint_props.to(device)

    self.eyeJ2 = self.eyeJ2.to(device)
    self.eyeJ = self.eyeJ.to(device)

    if distances_refinement is None:
          print('Initializing a refiner as identity')
          self.refiner=(self.identity)
    else:
          print('Initializing a distances refiner')
          self.refiner=(self.refine_distances)
          factor = 1.#self.adjacency.to(device) #torch.exp((torch.abs(Corr))).view(1,self.J,self.J,self.J,self.J)*self.adjacency
          #factor = factor.to(device)
          self.normalizer = factor*self.mask_no_self_connections + self.self_keypoint_props
          self.normalizer = self.normalizer/(self.normalizer.sum(-1).sum(-1).view(1,self.J,self.J,1,1))
          self.normalizer = self.normalizer.to(device)
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
        tiled = x.view(x.shape[0],-1).repeat(1,self.J**2).view(x.shape[0],self.J,self.J,self.J,self.J)
        return (self.priorMean*tiled*self.normalizer).sum(-1).sum(-1)

  def compute_likelihood(self, x):
        return -(torch.pow(x-self.priorMean,2)/(2*self.priorStd.pow(2))).view(x.shape[0],-1).mean(-1)

  def forward(self,x):
        pass

class PriorRegressionCriterion(AbstractPriorLoss):
  def __init__(self, path, J=10, eps = 10**(-6), device='cuda', norm='l2', distances_refinement=None, obj='props'):
    super(PriorRegressionCriterion, self).__init__(path, J, eps, device, norm, distances_refinement)
    self.obj = obj
    if obj == 'props':
          self.forward=(self.forward_props)
    elif obj == 'dists':
          self.forward=(self.forward_dists)
    else:
          self.forward=(self.forward_synth)
  
  def forward_props(self, prediction, dt=None):
    eps = 1e-6
    prediction = prediction.view(prediction.shape[0],self.J,-1)
    dists = compute_distances(prediction, eps=self.eps)
    props = compute_proportions(dists, eps=self.eps).view(dists.shape[0],self.J,self.J,self.J,self.J)
    diff = (props-self.priorMean)#/(self.priorStd + eps)
    mse = self.norm(diff)
    assert(torch.isnan(mse).sum() < 1)
    return mse


  def forward_dists(self, prediction, dt=None):
    eps = 1e-6
    prediction = prediction.view(prediction.shape[0],self.J,-1)
    dists = compute_distances(prediction, eps=self.eps)

    diff = (dists-self.distMean)/(self.distStd + eps)
    mse = self.norm(diff)

    return mse

  def forward_synth(self, prediction, dt=None):
    prediction = prediction.view(prediction.shape[0],self.J,-1)
    dists = compute_distances(prediction, eps=self.eps)
    props = compute_proportions(dists, eps=self.eps).view(dists.shape[0],self.J,self.J,self.J,self.J)
    gt_dists = self.refiner(props)*(1.-self.eyeJ)

    props = compute_proportions(dists, eps=self.eps).view(dists.shape[0],self.J,self.J,self.J,self.J)
    gt_dists = self.refiner(dists, props)

    diff = (dists-gt_dists)
    mse = self.norm(diff)

    return mse


##############################
#### Weighted MDS, simple ####
##############################

class PriorSMACOFCriterion(AbstractPriorLoss):
  def __init__(self, path, J=10, eps = 10**(-6), device='cuda', norm='l2', distances_refinement=None, iterate=False, rotation_weight=0, scale_weight=0):
    super(PriorSMACOFCriterion, self).__init__(path, J, eps, device, norm, distances_refinement)

    self.eyeK = torch.eye(3).unsqueeze(0).float()
    self.rotation_weight = rotation_weight
    self.scale_weight = scale_weight
    self.dists_mask = torch.FloatTensor(1,self.J,self.J).zero_()
    for i in range(self.J):
          for j in range(self.J):
                if j>i:
                      self.dists_mask[0,i,j]=1.
                      
    self.eyeK = self.eyeK.to(device)
    self.dists_mask = self.dists_mask.to(device)

    if rotation_weight > 0 or scale_weight > 0:
          self.forward = self.forward_regularized    
    elif iterate:
          self.forward = self.forward_iterative
    else:
          self.forward = self.forward_objective


  def forward_regularized(self, prediction, dt=None):

    prediction = prediction.view(prediction.shape[0],self.J,-1)

    dists = compute_distances(prediction, eps=self.eps)
    props = compute_proportions(dists, eps=self.eps).view(dists.shape[0],self.J,self.J,self.J,self.J)
    gt_dists = self.refiner(dists,props)*(1.-self.eyeJ)

    w = torch.ones_like(gt_dists)

    regression_term = self.compute_obj(prediction, gt_dists, w)/self.J

    rotation_loss=0.
    if self.rotation_weight > 0:
        smacof_x = self.iterate(prediction, gt_dists, w).detach()
        rotation_loss = compute_rotation_loss(prediction, smacof_x)

    scale_loss=0.
    if self.scale_weight > 0:
        if self.rotation_weight <= 0:
            smacof_x = self.iterate(prediction, gt_dists, w).detach()
        smacof_dists = compute_distances(smacof_x, eps=self.eps)
        scale_loss = compute_scale_loss(dists, smacof_dists)

    return regression_term + self.scale_weight * scale_loss + self.rotation_weight * rotation_loss


  def forward_objective(self, prediction, dt=None):

    prediction = prediction.view(prediction.shape[0],self.J,-1)

    dists = compute_distances(prediction, eps=self.eps)
    props = compute_proportions(dists, eps=self.eps).view(dists.shape[0],self.J,self.J,self.J,self.J)
    gt_dists = self.refiner(dists,props)*(1.-self.eyeJ)

    w = torch.ones_like(gt_dists)

    error = self.compute_obj(prediction, gt_dists, w)/self.J

    return error



  def forward_iterative(self, prediction, dt=None):
    prediction = prediction.view(prediction.shape[0],self.J,-1)

    dists = compute_distances(prediction, eps=self.eps)*(1.-self.eyeJ)
    props = compute_proportions(dists, eps=self.eps).view(dists.shape[0],self.J,self.J,self.J,self.J)
    gt_dists = self.refiner(dists, props)*(1.-self.eyeJ)

    w = torch.ones_like(gt_dists)

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
        Z = X
        TB = torch.bmm(T,B)
        TBZ = torch.bmm(TB,Z)
        third_term = -2*self.trace(TBZ)
        return first_term + second_term + third_term


  def iterate(self,x,delta, w=None, iters=10, use_w=False):
        delta = delta.view(delta.shape[0],self.J, self.J)
        #### Compute the second term: \sum_{i<j} w_{ij} d^2_{ij}(X) = trace(X'VX)####
        X = x.view(x.shape[0],self.J,-1)
        x.permute(0,2,1)
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

      correlation = np.corrcoef(dists.reshape(priors.shape[0],-1).transpose())
      mean = priors.mean(0)
      std = priors.std(0)

      mean = torch.from_numpy(mean).float()
      std = torch.from_numpy(std).float()

      mean_d = dists.mean(0)
      std_d = dists.std(0)

      mean_d = torch.from_numpy(mean_d).float()
      std_d = torch.from_numpy(std_d).float()

      correlation = torch.from_numpy(correlation).float()

      return mean.to(device), std.to(device), mean_d.to(device), std_d.to(device),correlation#.to(device)



def load_priors_from_file(root_folder, device='cuda', eps=10**(-6)):
      #ModelNet_MeanDists.npy  ModelNet_MeanProp.npy  ModelNet_StdDists.npy  ModelNet_StdProp.npy
      kMeanDistsFilename = 'ModelNet_MeanDists.npy'
      kStdDistsFilename = 'ModelNet_StdDists.npy'
      kMeanFilename = 'ModelNet_MeanProp.npy'
      kStdFilename = 'ModelNet_StdProp.npy'
      dist_mean = torch.from_numpy(np.load(os.path.join(root_folder, kMeanDistsFilename))).float()
      dist_std = torch.from_numpy(np.load(os.path.join(root_folder, kStdDistsFilename))).float()
      prop_mean = torch.from_numpy(np.load(os.path.join(root_folder, kMeanFilename))).float()
      prop_std = torch.from_numpy(np.load(os.path.join(root_folder, kStdFilename))).float()
      if (torch.isnan(prop_std).sum() > 0 or 
         torch.isnan(prop_mean).sum() > 0 or
         torch.isnan(dist_mean).sum() > 0 or
         torch.isnan(dist_std).sum() > 0):
               print('LOADED FILES CONTAIN NaNs')
      return prop_mean.to(device), prop_std.to(device), dist_mean.to(device), dist_std.to(device)

###############
#### NORMS ####
###############


def l2(x,w=1.):
      x=x.view(x.shape[0],-1)
      #return (w*(x.pow(2))).sum(-1)
      return (w*(x.pow(2))).sum(-1).pow(0.5)/x.shape[1]

def frobenius(x):
      assert len(x.shape)==3
      x_transposed = x.permute(0,2,1) # B x K x D
      xTx = torch.bmm(x_transposed,x)
      eye = torch.eye(x.shape[1]).unsqueeze(0).float().to(x.device)
      return (xTx*eye).sum(-1).sum(-1)


def l1(x,w=1):
      x=x.view(x.shape[0],-1)
      return torch.norm(x*w,p=1,dim=-1)


def compute_rotation_loss(x,y):
      rot_loss = 0.
      target = torch.eye(x.shape[-1]).unsqueeze(0).to(x.device)
      diag_0 = torch.diag(torch.Tensor([1.,1.,0])).to(x.device)
      diag_1 = torch.diag(torch.Tensor([0.,0.,1.])).to(x.device)
      yTx = torch.bmm(y.permute(0,2,1),x)

      for i in range(x.shape[0]):
            U,S,Vt = yTx[i].svd()
            xTy = yTx[i].permute(1,0)
            s = torch.sign(xTy.det())
            diag_S = diag_0 + (diag_1*s)
            R = torch.mm(U, torch.mm(diag_S, Vt.t()))
            I = torch.mm(R, R.t())
            assert((I - torch.eye(3).to(x.device)).sum().item() < 1e-4) #DEBUG PURPOSE
            rot_loss = rot_loss + torch.norm(R-target)
      return rot_loss/x.shape[0]


def compute_scale_loss(dist_x,dist_y):
      max_x,_ = torch.max(dist_x.view(dist_x.shape[0],-1),1)
      max_y,_ = torch.max(dist_y.view(dist_y.shape[0],-1),1)

      return l2(max_x-max_y)

'''
def compute_rotation(x,y):
      diag_0 = torch.diag(torch.Tensor([1.,1.,0])).to(x.device)
      diag_1 = torch.diag(torch.Tensor([0.,0.,1.])).to(x.device)
      yTx = torch.mm(y.permute(1,0),x)

      U,S,Vt = yTx.svd()
      xTy = yTx.permute(1,0)
      s = torch.sign(xTy.det())
      diag_S = diag_0 + (diag_1*s)
      print(diag_S)
      #R = U*(diag_S.double())*Vt
      R = torch.mm(U,torch.mm((diag_S.double()),Vt.t()))
      return R
'''
