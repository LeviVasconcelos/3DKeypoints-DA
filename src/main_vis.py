import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets

from layers.PriorConsistencyCriterion import PriorConsistencyCriterion,DistanceConsistencyCriterion, get_priors_from_file

from tensorboardX import SummaryWriter

import ref
import cv2
import numpy as np
from datasets.Fusion import Fusion

from utils.logger import Logger
from opts import opts
from train_with_priors import train, validate, test
from optim_latent import initLatent, stepLatent, getY
from model import getModel
from utils.utils import collate_fn_cat

from datasets.chairs_modelnet import ChairsModelNet as SourceDataset
args = opts().parse()
if args.targetDataset == 'Redwood':
  from datasets.chairs_Redwood import ChairsRedwood as TargetDataset
elif args.targetDataset == 'ShapeNet':
  from datasets.chairs_Annotatedshapenet import ChairsShapeNet as TargetDataset
elif args.targetDataset == 'RedwoodRGB':
  from datasets.chairs_RedwoodRGB import ChairsRedwood as TargetDataset
elif args.targetDataset == '3DCNN':
  from datasets.chairs_3DCNN import Chairs3DCNN as TargetDataset
else:
  raise Exception("No target dataset {}".format(args.targetDataset))

splits = ['train', 'valSource', 'valTarget']


logger = SummaryWriter(args.logDir)

def main():

  # Init model
  model = getModel(args)
  cudnn.benchmark = True

  # Init optimizer
  optimizer = torch.optim.SGD(model.parameters(), args.LR,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)


  # Init priors:
  if 'distances' in args.propsFile:
	  Mean,Std = get_priors_from_file(args.propsFile)
	  prior_loss = DistanceConsistencyCriterion(Mean,Std,norm = args.lossNorm, std_weight = args.weightedNorm, eps=args.eps)
  else:
	  Mean,Std = get_priors_from_file(args.propsFile)
	  prior_loss = PriorConsistencyCriterion(Mean,Std,norm = args.lossNorm, std_weight = args.weightedNorm, eps=args.eps)

  # Init loaders
  valSource_dataset = SourceDataset('test', ref.nValViews)
  valTarget_dataset = TargetDataset('test', ref.nValViews)
  
  valSource_loader = torch.utils.data.DataLoader(valSource_dataset, batch_size = 1, 
                        shuffle=False, num_workers=1, pin_memory=True, collate_fn=collate_fn_cat)
  valTarget_loader = torch.utils.data.DataLoader(valTarget_dataset, batch_size = 1, 
                        shuffle=False, num_workers=1, pin_memory=True, collate_fn=collate_fn_cat)
  

  if args.test:
    test(args, valSource_loader, model, prior_loss, None)
    test(args, valTarget_loader, model, prior_loss, None)
    return

  train_dataset = Fusion(SourceDataset, TargetDataset, nViews = args.nViews, targetRatio = args.targetRatio, totalTargetIm = args.totalTargetIm)
  trainTarget_dataset = train_dataset.targetDataset
  
  trainTarget_loader = torch.utils.data.DataLoader(
      trainTarget_dataset, batch_size=args.batchSize, shuffle=True,
      num_workers=args.workers if not args.test else 1, pin_memory=True, collate_fn=collate_fn_cat)



  valSource_mpjpe, valSource_loss, valSource_unSuploss = validate(args, 'Source', valSource_loader, model, prior_loss, 0)
  valTarget_mpjpe, valTarget_loss, valTarget_unSuploss = validate(args, 'Target', valTarget_loader, model, prior_loss, 0)
  logger.add_scalar('val/source-accuracy', valSource_mpjpe, 0)
  logger.add_scalar('val/target-accuracy', valTarget_mpjpe, 0)

  logger.add_scalar('val/target-accuracy', valTarget_mpjpe, 0)
    
  logger.add_scalar('val/source-regr-loss', valSource_loss, 0)
  logger.add_scalar('val/target-regr-loss', valTarget_loss, 0)
    
  logger.add_scalar('val/source-prior-loss', valSource_unSuploss, 0)
  logger.add_scalar('val/target-prior-loss', valTarget_unSuploss, 0)


  print 'Start training...'
  for epoch in range(1, args.epochs + 1):
    adjust_learning_rate(optimizer, epoch, args.dropLR)
    train_mpjpe, train_loss, train_unSuploss = train(args, trainTarget_loader, model, prior_loss, logger, optimizer, epoch-1)
    valSource_mpjpe, valSource_loss, valSource_unSuploss = validate(args, 'Source', valSource_loader, model, prior_loss, epoch)
    valTarget_mpjpe, valTarget_loss, valTarget_unSuploss = validate(args, 'Target', valTarget_loader, model, prior_loss, epoch)

    logger.add_scalar('val/source-accuracy', valSource_mpjpe, epoch)
    logger.add_scalar('val/target-accuracy', valTarget_mpjpe, epoch)

    logger.add_scalar('val/source-regr-loss', valSource_loss, epoch)
    logger.add_scalar('val/target-regr-loss', valTarget_loss, epoch)
    
    logger.add_scalar('val/source-prior-loss', valSource_unSuploss, epoch)
    logger.add_scalar('val/target-prior-loss', valTarget_unSuploss, epoch)
    
    if epoch % 10 == 0:
      torch.save({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
      }, args.save_path + '/checkpoint_{}.pth.tar'.format(epoch))
    
  print('Training endend')
  logger.close()

def adjust_learning_rate(optimizer, epoch, dropLR):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = args.LR * (0.1 ** (epoch // dropLR))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


if __name__ == '__main__':
  main()

