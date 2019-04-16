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

import ref
from extract_priors import extract_gt
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
      target_dataset = TargetDataset('train', ref.nViews)
      dataset_loader = torch.utils.data.DataLoader(
              target_dataset, batch_size=args.batchSize, shuffle=not args.test,
              num_workers=args.workers if not args.test else 1, pin_memory=False, collate_fn=collate_fn_cat)
      dist, props = extract_gt(dataset_loader)
      np.save(args.propsFile+'-source-distances.npy', dist)
      np.save(args.propsFile+'-source-props.npy', props)
      
