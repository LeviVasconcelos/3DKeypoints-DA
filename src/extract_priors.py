import torch
import numpy as np
from utils.utils import AverageMeter, show3D
from utils.eval import accuracy, shapeConsistency, accuracy_dis
import cv2
import ref
from progress.bar import Bar
import layers.prior_generator as prior

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


