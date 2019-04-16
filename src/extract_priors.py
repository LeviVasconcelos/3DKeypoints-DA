import torch
import numpy as np
from utils.utils import AverageMeter, show3D
from utils.eval import accuracy, shapeConsistency, accuracy_dis
import cv2
import ref
from progress.bar import Bar
import layers.prior_generator as prior
from tqdm import tqdm

def extract(args, loader, model, nViews=ref.nViews):
  model.eval()
  nViews = loader.dataset.nViews
  dist = []
  props = []
  for i, (input, target, meta) in enumerate(loader):  
    input_var = torch.autograd.Variable(input.cuda())
    target_var = torch.autograd.Variable(target)
    output = model(input_var)

    for j in range(input.numpy().shape[0]):
        img = (input.numpy()[j] * 255).transpose(1, 2, 0).astype(np.uint8)
        gt = target.cpu()[j]#.numpy()[j]
        pred = (output.data).cpu()[j].view(10,3)#.numpy()[j]
	cdist = prior.compute_distances(pred)
	dist.append(cdist.numpy())
	props.append(prior.compute_proportions(cdist).numpy())

  dist_arrays = np.zeros((len(dist),10,10))
  props_arrays = np.zeros((len(dist),100,100))
  for i in range(len(dist)):
	dist_arrays[i]=dist[i]
	props_arrays[i]=props[i]
  return dist, props#np.asarray(props)

def compute_proportions_np(x, eps=1e-6):
    numerator = x.flatten()[:,np.newaxis] # B x K^2 x 1
    denominator = (1./(numerator+eps)).T # B x 1 x K^2
    mm = np.matmul(numerator,denominator) # B x K^2 x K^2
    #if (np.isnan(mm).sum() > 0):
    #   print('NaN computing proportions...')
    #assert(np.isnan(mm).sum() < 1)
    return mm

def extract_props_from_dists(dists):
  dd = dists.reshape((dists.shape[0], ref.J, ref.J))
  mean = np.zeros((ref.J**2, ref.J**2))
  std = np.zeros((ref.J**2, ref.J**2))
  #for i in range(0,dists.shape[0], b):
  for d in tqdm(dd):
      #x = (prior.compute_proportions(dd[i:i+b]).to('cuda'))
      x = compute_proportions_np(d)
      #print(x.shape)
      mean += x
  mean /= float(dd.shape[0])
  np.save('proportions_mean.npy', mean)
  for d in tqdm(dd):
      x = compute_proportions_np(d)
      std += (x - mean)**2 
  std /= float(dd.shape[0])
  std = np.sqrt(std)
  np.save('proportions_std.npy', std)
  return mean, std 
      

def extract_dists_gt(loader, nViews=ref.nViews):
  nViews = loader.dataset.nViews
  dist = []
  print('Starting prior computation')
  pbar = tqdm(len(loader))
  for i, (_, target, _) in enumerate(loader):  
    target_var = torch.autograd.Variable(target)
    #print(target_var.shape)
    for j in range(target_var.shape[0]):
        gt = target.cpu()[j]#.numpy()[j]
        cdist = prior.compute_distances(gt)
        dist.append(cdist.numpy())
        pbar.update(1)
  return np.asarray(dist)


