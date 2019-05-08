import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from utils.horn87 import horn87, RotMat
import ref
from prior_generator import compute_distances, compute_proportions, replicate_mask, get_shape_index
from scipy.misc import toimage
from torchviz import make_dot
import math
import os

###########################
#### LOSS DEFINITIONS #####
###########################
'''First loss, create synth gt through 
the current predictions and the priors'''

EDGES = ref.human_edges 

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
                  if i==j: 
			mask_no_self_connections[0,i,j,:,:]=0.0
			#self_keypoint_props[0,i,j,i,j] = 1.
			continue
                  for l in range(self.J):
                        for m in range(self.J):
                              if l==m or (i==l and j==m) or (i==m and j==l):
                                    mask_no_self_connections[0,i,j,l,m]=0.0

    self.upper_triangular = self.upper_triangular.view(1,self.J,self.J,self.J,self.J).to(device)
    self.adjacency = adjacency.to('cuda')
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
          factor = 1.
          if distances_refinement == 'adjacency':
              factor = self.adjacency.to(device)
          self.normalizer = factor*self.mask_no_self_connections + self.self_keypoint_props
          self.normalizer = self.normalizer/(self.normalizer.sum(-1).sum(-1).view(1,self.J,self.J,1,1))
          self.normalizer = self.normalizer.to(device)
          print('PERFORMING NORMALIZER CHECKS')
          if (torch.isnan(self.normalizer).sum() > 0):
                print('normalizer has NaNs')
                return
          if (abs(torch.abs(self.normalizer.sum(-1).sum(-1).mean()) - 1.0) > 10e-5):
                print('normalizer not 1') 
                return

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
        #print('x_shape: ', x.shape)
        #print('pros_shape: ', props.shape)
        #print('normalizer shape:', self.normalizer.shape)
        tiled = x.view(x.shape[0],-1).repeat(1,self.J**2).view(x.shape[0],self.J,self.J,self.J,self.J)
        #return (props*tiled*self.normalizer).sum(-1).sum(-1)
        #before_reconstruction = (props*tiled*self.normalizer)
        gt_dists = (props*tiled*self.normalizer).sum(-1).sum(-1)
        #print(torch.abs(gt_dists - x).sum())
        #print('gt_shape:', gt_dists.shape)
        if (torch.abs(gt_dists) > 500.).sum() > 0:
            tiled_max = torch.max(tiled)
            gt_dists_max = torch.max(gt_dists)
            normalizer_max = torch.max(self.normalizer)
            props_max_idx = torch.argmax(props).item() 
            print('norm_weight of props_max:', self.normalizer.view(-1)[props_max_idx % self.normalizer.numel()])
            print('tiled max: ',tiled_max)
            print('gt_dist max: ',gt_dists_max)
            print('normalizer max: ',normalizer_max)
            idx = torch.argmax(gt_dists).item()
            gt_dist_idx = get_shape_index(idx, gt_dists.shape)
            #np.save('props_of_max.npy', props[gt_dist_idx].cpu().numpy())
            #np.save('tiled_of_max.npy', tiled[gt_dist_idx].cpu().numpy())
            #np.save('norm_of_max.npy', self.normalizer[0][gt_dist_idx[1:]].cpu().numpy())
            #print('idx: ', gt_dist_idx)
            print('gt_dist_max from idx: ', gt_dists[gt_dist_idx])
            print('max before reconstruction: ', torch.max(before_reconstruction[gt_dist_idx])) # 17 x 17
            br_max_idx = get_shape_index(torch.argmax(before_reconstruction[gt_dist_idx]), before_reconstruction[gt_dist_idx].shape)
            #print('max br from idx: ', before_reconstruction[gt_dist_idx][br_max_idx])
            print('proportions of max br: ', props[gt_dist_idx][br_max_idx]) 
            print('tiled of max br: ', tiled[gt_dist_idx][br_max_idx])
            print('normalizer of max br: ', self.normalizer[0][gt_dist_idx[1:]][br_max_idx])
            print('product: ', props[gt_dist_idx][br_max_idx]*tiled[gt_dist_idx][br_max_idx]* self.normalizer[0][gt_dist_idx[1:]][br_max_idx])
            print('gt_dists too high:')
            if (torch.abs(x) > 5.).sum() > 0:
                print('predictions weird')
        return gt_dists

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
  def __init__(self, path, J=10, eps = 10**(-6), device='cuda', norm='l2', distances_refinement=None, iterate=False, rotation_weight=0, scale_weight=0, debug=0, debug_folder=''):
    super(PriorSMACOFCriterion, self).__init__(path, J, eps, device, norm, distances_refinement)

    self.eyeK = torch.eye(3).unsqueeze(0).float()
    self.rotation_weight = rotation_weight
    self.scale_weight = scale_weight
    self.dists_mask = torch.FloatTensor(1,self.J,self.J).zero_()
    self.debug_folder=debug_folder
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
          self.forward = self.forward_objective if (debug == 0) else self.forward_debug
    self.epoch = 0
    self.last_epoch = None

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

  def forward_debug(self, prediction, dt=None, iters=100):
    if self.last_epoch != self.epoch and prediction.shape[0] == 4:
        prediction = prediction.view(prediction.shape[0], self.J, -1)
        dt = dt.view(dt.shape[0], self.J, -1)
        dists_predictions = compute_distances(prediction, eps=self.eps)
        gt_dists = compute_distances(dt, eps=self.eps)
        gt_props, _ = compute_proportions(gt_dists, eps=self.eps)
        gt_props = gt_props.view(gt_dists.shape[0],self.J,self.J,self.J,self.J)
        reconstructed_gt_dists = (self.refiner(dists_predictions, gt_props)*(1.-self.eyeJ))
        reconstructed_mean_dists = (self.refiner(dists_predictions, self.priorMean)*(1.-self.eyeJ))
        reconstructed_kps = self.iterate(prediction, reconstructed_gt_dists, torch.ones_like(gt_dists), iters=iters)
        reconstructed_mean_kps = self.iterate(prediction, reconstructed_mean_dists, torch.ones_like(gt_dists), iters=iters)
        
        np.save(os.path.join(self.debug_folder,'reconstructed_%d_iters_%d' % (self.epoch, iters)), reconstructed_kps.detach().cpu().numpy())
        np.save(os.path.join(self.debug_folder,'reconstructed_mean_%d_iters_%d' % (self.epoch, iters)), reconstructed_mean_kps.detach().cpu().numpy())
        np.save(os.path.join(self.debug_folder,'predictions_%d' % (self.epoch)), prediction.detach().cpu().numpy())
        np.save(os.path.join(self.debug_folder,'ground_truth_%d' % (self.epoch)), dt.detach().cpu().numpy())
        self.last_epoch = self.epoch
    return self.forward_objective(prediction, dt)

  def forward_objective(self, prediction, dt=None):
    prediction = prediction.view(prediction.shape[0], self.J, -1)
    dt = dt.view(dt.shape[0], self.J, -1)
    #print('Predictions distances computation')
    dists_predictions = compute_distances(prediction, eps=self.eps)
    #print('GT distance computation')
    #dists = compute_distances(dt, eps=self.eps)
    #props, idx = compute_proportions(dists, eps=self.eps)
    #props = props.view(dists.shape[0],self.J,self.J,self.J,self.J)
    #if idx is not None:
    #   issue_dt = dt[idx[0]].cpu().numpy()
    #   np.save('issue_batch_%d.npy' % (idx[0]), issue_dt)
    #   print('***** FILE SAVED ****')
    #print('props sum: ', props.sum())
    #print('prior Mean sum: ', self.priorMean.sum())
    #gt_dists = (self.refiner(dists_predictions,props)*(1.-self.eyeJ))
    gt_dists = (self.refiner(dists_predictions,self.priorMean)*(1.-self.eyeJ))
    #f = open('diff_dists.txt', 'a+')
    #f.write('%lf\n' % torch.abs((gt_dists - dists_predictions)).mean().item())
    #f.close()
    ##print('gt_dist sum: ', gt_dists.sum())
    #if (torch.isnan(dists_predictions).sum() > 0):
    #    print('prediction_dists with NAN')
    #    return
    #if (torch.isnan(dists).sum() > 0):
    #    print('GT distances with NaN')
    #    return
    #if (torch.isnan(props).sum() > 0):
    #    print('GT props with NaN')
    #    return
    #if (torch.isnan(gt_dists).sum() > 0):
    #    print('refiner distances with NaN')
    #    return
    w = torch.ones_like(gt_dists)
    #print()
    #print()
    #if (gt_dists[0].mean()>10):
    #	print(dists[0],gt_dists[0],dists_predictions[0])
    # 	print(dt[0],prediction[0])
    #	print(props[0])
    # 	print()
    #	exit(1)
    error = self.compute_obj(prediction, gt_dists, w)/self.J
    #mask_error = (gt_dists.mean(-1).mean(-1)<2.0).float()
    return error#*torch.exp(-error)#mask_error



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
        #print('COMPUTING OBJECTIVE DISTANCE COMPUTATION')
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
      kMeanDistsFilename = 'HumansRGB_MeanDists.npy'
      kStdDistsFilename = 'HumansRGB_StdDists.npy'
      kMeanFilename = 'HumansRGB_MeanProp.npy'
      kStdFilename = 'HumansRGB_StdProp.npy'
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
