#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:15:53 2019

@author: levi
"""

import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
from extract_priors import extract_dists_gt,extract_props_from_dists
from opts import opts
from utils.utils import collate_fn_cat
args = opts().parse()
if args.targetDataset == 'Redwood':
  from datasets.chairs_Redwood import ChairsRedwood as TargetDataset
elif args.targetDataset == 'ShapeNet':
  from datasets.chairs_Annotatedshapenet import ChairsShapeNet as TargetDataset
elif args.targetDataset == 'RedwoodRGB':
  from datasets.chairs_RedwoodRGB import ChairsRedwood as TargetDataset
elif args.targetDataset == '3DCNN':
  from datasets.chairs_3DCNN import Chairs3DCNN as TargetDataset
elif args.targetDataset == 'HumansRGB':
  from datasets.humans36m import Humans36mRGBTargetDataset as TargetDataset
elif args.targetDataset == 'HumansDepth':
  from datasets.humans36m import Humans36mDepthTargetDataset as TargetDataset

def main():
      print('celled....')  
      #if os.path.exists(args.propsFile+'-source-distances.npy'):
      print('data loaded')
      if not os.path.exists(args.propsFile+'-source-distances.npy'):
          target_dataset = TargetDataset('train', args.nViews)
          dataset_loader = torch.utils.data.DataLoader(
               target_dataset, batch_size=args.batchSize, shuffle=not args.test,
               num_workers=4 if not args.test else 1, pin_memory=False, collate_fn=collate_fn_cat)
          dist = extract_dists_gt(dataset_loader)
          dist = dist.reshape((dist.shape[0], dist.shape[2], dist.shape[3]))
          mean = dist.mean(0)
          std = dist.std(0)
          np.save(args.propsFile+'-source-distances.npy', dist)
          np.save(args.propsFile+'-source-distances-mean.npy', mean)
          np.save(args.propsFile+'-source-distances-std.npy', std)
      else:
          print('loading distances')
          dist = np.load(args.propsFile+'-source-distances.npy')
      props,_ = extract_props_from_dists(dist)
main()      
