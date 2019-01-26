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

import ref
import cv2
import numpy as np
from datasets.Fusion import Fusion

from utils.logger import Logger
from opts import opts
from train import train, validate, test, dial_train
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

DIAL = args.approx_dial 

def main():
  call_count = 0
  now = datetime.datetime.now()
  logger = Logger(args.save_path + '/logs_{}'.format(now.isoformat()))

  model = getModel(args)
  cudnn.benchmark = True
  params_to_optim = list(filter(lambda p: p.requires_grad, model.parameters()))
  optimizer = torch.optim.SGD(params_to_optim, args.LR,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

  valSource_dataset = SourceDataset('test', ref.nValViews)
  valTarget_dataset = TargetDataset('test', ref.nValViews)
  
  valSource_loader = torch.utils.data.DataLoader(valSource_dataset, batch_size = 1, 
                        shuffle=False, num_workers=1, pin_memory=True, collate_fn=collate_fn_cat)
  valTarget_loader = torch.utils.data.DataLoader(valTarget_dataset, batch_size = 1, 
                        shuffle=False, num_workers=1, pin_memory=True, collate_fn=collate_fn_cat)
  
  if args.test:
    f = {}
    for split in splits:
      f['{}'.format(split)] = open('{}/{}.txt'.format(args.save_path, split), 'w')
    test(args, valSource_loader, model, None, f['valSource'], 'valSource')
    test(args, valTarget_loader, model, None, f['valTarget'], 'valTarget')
    return
  
  fusion_dataset = Fusion(SourceDataset, TargetDataset, nViews = args.nViews, targetRatio = args.targetRatio, totalTargetIm = args.totalTargetIm)
  trainTarget_dataset = TargetDataset
  trainSource_dataset = SourceDataset
  
  fusion_loader = torch.utils.data.DataLoader(
      fusion_dataset, batch_size=args.batchSize, shuffle=not args.test,
      num_workers=args.workers if not args.test else 1, pin_memory=True, collate_fn=collate_fn_cat)
  trainSource_loader = torch.utils.data.DataLoader(
      trainSource_dataset, batch_size=args.batchSize, shuffle=True,
      num_workers=args.workers if not args.test else 1, pin_memory=True, collate_fn=collate_fn_cat)
  trainTarget_loader = torch.utils.data.DataLoader(
      trainTarget_dataset, batch_size=args.batchSize, shuffle=True,
      num_workers=args.workers if not args.test else 1, pin_memory=True, collate_fn=collate_fn_cat)

  M = None
  if args.shapeWeight > ref.eps:
    print 'getY...'
    model.set_domain(source=False)
    Y = getY(SourceDataset)
    M = initLatent(trainTarget_loader, model, Y, nViews = args.nViews, S = args.sampleSource, AVG = args.AVG, dial=DIAL)
  
  print 'Start training...'
  
  for epoch in range(1, args.epochs + 1):
    adjust_learning_rate(optimizer, epoch, args.dropLR)
    if args.shapeWeight > ref.eps and args.dialModel:
          train_loader = zip(trainSource_loader, trainTarget_loader)
          train_mpjpe, train_loss, train_unSuploss = dial_train(args, train_loader, model, optimizer, M, epoch, dial=DIAL, nViews=args.nViews)
    else:
          train_loader = fusion_loader
          train_mpjpe, train_loss, train_unSuploss = train(args, train_loader, model, optimizer, M, epoch, dial=DIAL, nViews=args.nViews)
    if args.dialModel:
          model.set_domain(source=True)
    valSource_mpjpe, valSource_loss, valSource_unSuploss = validate(args, 'Source', valSource_loader, model, None, epoch)
    if args.dialModel:
          model.set_domain(source=False)
    valTarget_mpjpe, valTarget_loss, valTarget_unSuploss = validate(args, 'Target', valTarget_loader, model, None, epoch)

    train_loader.dataset.targetDataset.shuffle()
    if args.shapeWeight > ref.eps and epoch % args.intervalUpdateM == 0:
      if args.dialModel:
            model.set_domain(source=False)
      M = stepLatent(trainTarget_loader, model, M, Y, nViews = args.nViews, lamb = args.lamb, mu = args.mu, S = args.sampleSource, call_count=call_count, dial=DIAL)
      call_count += 1

    logger.write('{} {} {}\n'.format(train_mpjpe, valSource_mpjpe, valTarget_mpjpe))
    
    logger.scalar_summary('train_mpjpe', train_mpjpe, epoch)
    logger.scalar_summary('valSource_mpjpe', valSource_mpjpe, epoch)
    logger.scalar_summary('valTarget_mpjpe', valTarget_mpjpe, epoch)
    
    logger.scalar_summary('train_loss', train_loss, epoch)
    logger.scalar_summary('valSource_loss', valSource_loss, epoch)
    logger.scalar_summary('valTatget_loss', valTarget_loss, epoch)
    
    logger.scalar_summary('train_unSuploss', train_unSuploss, epoch)
    logger.scalar_summary('valSource_unSuploss', valSource_unSuploss, epoch)
    logger.scalar_summary('valTarget_unSuploss', valTarget_unSuploss, epoch)
    
    if epoch % 10 == 0:
      torch.save({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
      }, args.save_path + '/checkpoint_{}.pth.tar'.format(epoch))
  logger.close()

def adjust_learning_rate(optimizer, epoch, dropLR):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = args.LR * (0.1 ** (epoch // dropLR))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


if __name__ == '__main__':
  main()