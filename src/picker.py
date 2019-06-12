#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:15:53 2019

@author: levi
"""

import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import numpy as np
import math
from tqdm import tqdm
import os
from layers.PriorConsistencyCriterion import PriorRegressionCriterion, PriorSMACOFCriterion
import ref
from tensorboardX import SummaryWriter
from datasets.humans36m import Humans36mRGBSourceDataset, Humans36mRGBTargetDataset
from datasets.Fusion import Fusion
from datasets.humans36m import Humans36mDataset
from tqdm import tqdm
from utils.visualization import chair_show3D, chair_show2D, human_show2D, human_show3D, human_from_3D 
import torchvision.models as models
from model import getModel 
from extract_priors import extract_dists_gt,extract_props_from_dists
from opts import opts
from utils.utils import collate_fn_cat
from train_with_priors import  train_priors, validate_priors
import models.DIALResNet as dial
import cv2

args = opts().parse()
if args.targetDataset == 'Redwood':
  from datasets.chairs_Redwood import ChairsRedwood as TargetDataset
elif args.targetDataset == 'ShapeNet':
  from datasets.chairs_Annotatedshapenet import ChairsShapeNet as TargetDataset
elif args.targetDataset == 'RedwoodRGB':
  from datasets.chairs_RedwoodRGB import ChairsRedwood as TargetDataset
elif args.targetDataset == 'HumansRGB':
  from datasets.humans36m import Humans36mRGBTargetDataset as TargetDataset
elif args.targetDataset == 'HumansDepth':
  from datasets.humans36m import Humans36mDepthTargetDataset as TargetDataset
if args.sourceDataset =='ModelNet':
  from datasets.chairs_modelnet import ChairsModelNet as SourceDataset
elif args.sourceDataset == 'HumansRGB':
  from datasets.humans36m import Humans36mRGBSourceDataset as SourceDataset
elif args.sourceDataset == 'HumansDepth':
  from datasets.humans36m import Humans36mDepthSourceDataset as SourceDataset


def load_data_parallel(x):
	# original saved file with DataParallel
	state_dict = x
	# create new OrderedDict that does not contain `module.`
	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
	    name = k[7:] # remove `module.`
	    new_state_dict[name] = v
	# load params
	return new_state_dict

def loadModel_chair(filepath, dial_model=False):
    if not dial_model:
        print('creating normal resnet50')
        model = models.resnet50(num_classes = ref.J * 3).cuda()
    else:
        print('creating dial resnet50')
        model = dial.resnet50(num_classes = ref.J * 3).cuda()
        
    if os.path.isfile(filepath):
      checkpoint = torch.load(filepath)
      if 'pth' not in filepath:
              print('data_parallel')
              model.load_state_dict(load_data_parallel(checkpoint['state_dict']))
      else:
              print('only cuda')
              if (dial_model):
                  model.load_state_dict(checkpoint['state_dict'])
              else:
                  model.load_state_dict(checkpoint['state_dict'])
    else:
        raise Exception("=> no model found at '{}'".format(filepath))
  
    return model.cuda()

def loadModel(filepath):
    #dict_models = {'resnet50': models.resnet50}
    model = models.resnet50(num_classes = ref.J * 3)
    model.cuda()
    if os.path.isfile(filepath):
          print("=> loading model '{}'".format(filepath))
          checkpoint = torch.load(filepath)
          if 'pth' not in filepath:
                model.load_state_dict(load_data_parallel(checkpoint['state_dict']))
          else:
                model.load_state_dict(checkpoint['state_dict'])
    else:
          raise Exception("=> no model found at '{}'".format(filepath))
    return model.cuda()

device = 'cuda'

def ApplyModel(model, input_var, unnorm_net):
    output = model(input_var).detach()
    unnormed_prediction = unnorm_net(output.view(input_var.shape[0], ref.J, 3))
    return unnormed_prediction

def RotMat(axis, ang):
  s = np.sin(ang)
  c = np.cos(ang)
  res = np.zeros((3, 3))
  if axis == 'Z':
    res[0, 0] = c
    res[0, 1] = -s
    res[1, 0] = s
    res[1, 1] = c
    res[2, 2] = 1
  elif axis == 'Y':
    res[0, 0] = c
    res[0, 2] = s
    res[1, 1] = 1
    res[2, 0] = -s
    res[2, 2] = c
  elif axis == 'X':
    res[0, 0] = 1
    res[1, 1] = c
    res[1, 2] = -s
    res[2, 1] = s
    res[2, 2] = c
  return res

def horn87(pointss, pointst):
  centers = pointss.mean(axis = 1)
  centert = pointst.mean(axis = 1)
  pointss = (pointss.transpose() - centers).transpose()
  pointst = (pointst.transpose() - centert).transpose()
  m = np.dot(pointss, pointst.transpose(1, 0))
  n = np.array([[m[0, 0] + m[1, 1] + m[2, 2], m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]], 
                [m[1, 2] - m[2, 1], m[0, 0] - m[1, 1] - m[2, 2], m[0, 1] + m[1, 0], m[0, 2] + m[2, 0]], 
                [m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], m[1, 1] - m[0, 0] - m[2, 2], m[1, 2] + m[2, 1]], 
                [m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], m[2, 2] - m[0, 0] - m[1, 1]]])
  v, u = np.linalg.eig(n)
  id = v.argmax()

  q = u[:, id]
  r = np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])], 
                [2*(q[2]*q[1]+q[0]*q[3]), q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2*(q[2]*q[3]-q[0]*q[1])], 
                [2*(q[3]*q[1]-q[0]*q[2]), 2*(q[3]*q[2]+q[0]*q[1]), q[0]**2-q[1]**2-q[2]**2+q[3]**2]])
  t = centert - np.dot(r, centers)

  return r.astype(np.float32), t.astype(np.float32) 

def rotate_Y(points, gt):
    #print(points.shape)
    #print(gt.shape)
    R,t = horn87(points.transpose(), gt.transpose())
    #R = RotMat('Y', math.pi)
    ret = np.matmul(R,points.transpose()).transpose()
    return ret


def SavePose(img, prediction, gt_, uncentred, intrinsics_, filename):
    np.savez(filename + '.npz', img=img, pred=prediction, 
              gt=gt_, gt_uncentred=uncentred, intrinsics=np.array(intrinsics_))

def DrawImage(img, prediction, gt_, uncentred, intrinsics, epoch, nViews=4, tag=''):
    for i in range(1):
        numpy_img = img.copy()
        camera = i 
        pred = prediction.data[camera].cpu().numpy().copy()
        gt = gt_.data[camera].cpu().numpy().copy()
        pred = rotate_Y(pred, gt)
        gt_uncentred = uncentred.data[camera].cpu().numpy().copy()
        filename_2d = os.path.join(args.debug_folder, tag + ('_img2d_%s_%d_%d.png' % (args.expID, i, epoch)))
        SavePose(img, pred, gt, gt_uncentred, intrinsics[camera], filename_2d)
        numpy_img = human_from_3D(numpy_img, gt_uncentred, intrinsics[camera],
                                    (180,0,0), 224./1000.)
        numpy_img = human_from_3D(numpy_img, pred + gt_uncentred[0], intrinsics[camera], 
                                    (0,0,180), 224./1000., flip=False)
        
        cv2.imwrite(filename_2d, numpy_img)

   

def main_humans():
    source_model = loadModel(args.sourceModel)
    middle_model = loadModel(args.middleModel)
    final_model = loadModel(args.finalModel)
    huang_model = loadModel(args.huangModel)
    source_dataset = SourceDataset('train', args.sourceNViews)
    target_dataset = TargetDataset('test', args.targetNViews)
    target_loader =  torch.utils.data.DataLoader(target_dataset, batch_size = 1, 
                                                  shuffle=False, num_workers=1, pin_memory=True, 
                                                  collate_fn=collate_fn_cat)
    unnorm_net = source_dataset._unnormalize_pose
    unnorm_tgt = target_dataset._unnormalize_pose
    for i, (input, target, meta, uncentred, intrinsics) in enumerate(tqdm(target_loader)):
        input_var = input.to(device).detach()
        target_var = target.to(device)
        unnormed_gt = unnorm_tgt(target_var)
        source_prediction = ApplyModel(source_model, input_var, unnorm_net)
        middle_prediction = ApplyModel(middle_model, input_var, unnorm_net)
        final_prediction = ApplyModel(final_model, input_var, unnorm_net) 
        huang_prediction = ApplyModel(huang_model, input_var, unnorm_net) 


        numpy_img = (input.numpy()[0] * 255).transpose(1, 2, 0).astype(np.uint8)
        DrawImage(numpy_img, source_prediction, unnormed_gt, uncentred, intrinsics, i, tag='source')
        DrawImage(numpy_img, middle_prediction, unnormed_gt, uncentred,  intrinsics, i, tag='middle')
        DrawImage(numpy_img, final_prediction, unnormed_gt, uncentred, intrinsics, i, tag='final')
        DrawImage(numpy_img, huang_prediction, unnormed_gt, uncentred, intrinsics, i, tag='huang')


def ApplyModel_chair(model, input_var):
    output = model(input_var).detach()
    return output.view(input_var.shape[0], ref.J, 3) 

def SavePose_chair(img, prediction, gt_, filename):
    np.savez(filename + 'npz', img=img, pred=prediction, 
              gt=gt_)

def DrawImage_chair(img, prediction, gt_, epoch, nViews=4, tag=''):
    for i in range(nViews):
        numpy_img = img.copy()
        camera = i 
        pred = prediction.data[camera].cpu().numpy().copy()
        gt = gt_.data[camera].cpu().numpy().copy()
        #pred = rotate_Y(pred, gt)
        filename_2d = os.path.join(args.debug_folder, tag + ('_img2d_%s_%d_%d.png' % (args.expID, i, epoch)))
        #def chair_show2D(img, points, c, edges = chair_edges, J = ref.J):

        SavePose_chair(img, pred, gt, filename_2d)
        numpy_img = chair_show2D(numpy_img, gt, (180,0,0))
        numpy_img = chair_show2D(numpy_img, pred, (0,0,180))
        cv2.imwrite(filename_2d, numpy_img)


def main_chairs():
    source_model = loadModel_chair(args.sourceModel)
    #middle_model = loadModel(args.middleModel)
    final_model = loadModel_chair(args.finalModel, dial_model=True)
    huang_model = loadModel_chair(args.huangModel)
    target_dataset = TargetDataset('test', args.targetNViews)
    target_loader =  torch.utils.data.DataLoader(target_dataset, batch_size = 1, 
                                                  shuffle=False, num_workers=1, pin_memory=True, 
                                                  collate_fn=collate_fn_cat)
    for i, (input, target, meta) in enumerate(tqdm(target_loader)):
        input_var = input.to(device).detach()
        target_var = target.to(device)
        source_prediction = ApplyModel_chair(source_model, input_var)
        #middle_prediction = ApplyModel_chair(middle_model, input_var)
        final_prediction = ApplyModel_chair(final_model, input_var) 
        huang_prediction = ApplyModel_chair(huang_model, input_var) 

        #print('gt', target_var)
        #print('final', final_prediction)

        numpy_img = (input.numpy()[0] * 255).transpose(1, 2, 0).astype(np.uint8)
        DrawImage_chair(numpy_img, source_prediction, target_var, i, tag='source', nViews=args.targetNViews)
        #DrawImage_chair(numpy_img, middle_prediction, i, tag='middle')
        DrawImage_chair(numpy_img, final_prediction, target_var, i, tag='final', nViews=args.targetNViews)
        DrawImage_chair(numpy_img, huang_prediction, target_var, i, tag='huang', nViews=args.targetNViews)



if __name__ == '__main__':
    main_chairs()
