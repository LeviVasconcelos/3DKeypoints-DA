import os
import time
import datetime
import itertools

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
from train import train, validate, test, train_priors, validate_priors
from optim_latent import initLatent, stepLatent, getY
from model import getModel
from utils.utils import collate_fn_cat
from dial_train import train_statistics
from adda_train import train_discriminator
from extract_priors import extract

from datasets.chairs_modelnet import ChairsModelNet as SourceDataset
args = opts().parse()

if args.sourceDataset =='ModelNet':
  from datasets.chairs_modelnet import ChairsModelNet as SourceDataset
elif args.sourceDataset == 'HumansRGB':
  from datasets.humans36m import Humans36mRGBDataset as SourceDataset
elif args.sourceDataset == 'HumansDepth':
  from datasets.humans36m import Humans36mDepthDataset as SourceDataset
else:
  raise Exception("No source dataset: " + args.sourceDataset)

if args.targetDataset == 'Redwood':
  from datasets.chairs_Redwood import ChairsRedwood as TargetDataset
elif args.targetDataset == 'ShapeNet':
  from datasets.chairs_Annotatedshapenet import ChairsShapeNet as TargetDataset
elif args.targetDataset == 'RedwoodRGB':
  from datasets.chairs_RedwoodRGB import ChairsRedwood as TargetDataset
elif args.targetDataset == '3DCNN':
  from datasets.chairs_3DCNN import Chairs3DCNN as TargetDataset
elif args.targetDataset == 'HumansRGB':
  from datasets.humans36m import Humans36mRGBDataset as TargetDataset
elif args.targetDataset == 'HumansDepth':
  from datasets.humans36m import Humans36mDepthDataset as TargetDataset
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

          # Couple sanity checks
  kHumansDataset = ['HumansRGB', 'HumansDepth']
  if args.targetDataset in kHumansDataset or args.sourceDataset in kHumansDataset:
      assert(ref.nViews <= 4)
      assert(args.nViews <= 4)
      assert(ref.J == 32)
      assert(ref.category == 'Human')
      assert(ref.nValViews <= 4)
  source_valViews = ref.nValViews if args.sourceDataset != 'HumansDepth' else 1
  target_valViews = ref.nValViews if args.targetDataset != 'HumansDepth' else 1
  valSource_dataset = SourceDataset('test', ref.nValViews)
  valTarget_dataset = TargetDataset('test', ref.nValViews)
  
  valSource_loader = torch.utils.data.DataLoader(valSource_dataset, batch_size = 1, 
                        shuffle=False, num_workers=1, pin_memory=False, collate_fn=collate_fn_cat)
  valTarget_loader = torch.utils.data.DataLoader(valTarget_dataset, batch_size = 1, 
                        shuffle=False, num_workers=1, pin_memory=False, collate_fn=collate_fn_cat)
  
  if args.dialModel:
        print 'dial model on'

  if args.extractProps:
    #sourceDist, sourceProps = extract(args, valSource_loader, model)
    targetDist, targetProps = extract(args, valTarget_loader, model)
    #np.save(args.propsFile+'-source-distances.npy', sourceDist)	
    #np.save(args.propsFile+'-source-props.npy', sourceProps)	
    np.save(args.propsFile+'-distances.npy', targetDist)	
    np.save(args.propsFile+'-props.npy', targetProps)	
    return

  if args.test:
    f = {}
    for split in splits:
      f['{}'.format(split)] = open('{}/{}.txt'.format(args.save_path, split), 'w')
    if args.dialModel:
          model.set_domain(source=True)
    test(args, valSource_loader, model, None, f['valSource'], 'valSource')
    if args.dialModel:
          model.set_domain(source=False)
    test(args, valTarget_loader, model, None, f['valTarget'], 'valTarget')
    return
  if args.shapeWeight > ref.eps and args.dialModel:
        model.init_target_weights()
  if args.dial_copy_source:
        model.init_target_weights()
  fusion_dataset = Fusion(SourceDataset, TargetDataset, nViews = args.nViews, targetRatio = args.targetRatio, totalTargetIm = args.totalTargetIm)
  trainTarget_dataset = fusion_dataset.targetDataset
  trainSource_dataset = fusion_dataset.sourceDataset
  
  fusion_loader = torch.utils.data.DataLoader(
      fusion_dataset, batch_size=args.batchSize, shuffle=not args.test,
      num_workers=args.workers if not args.test else 1, pin_memory=False, collate_fn=collate_fn_cat)
  trainSource_loader = torch.utils.data.DataLoader(
      trainSource_dataset, batch_size=args.batchSize, shuffle=True,
      num_workers=args.workers if not args.test else 1, pin_memory=False, collate_fn=collate_fn_cat)
  trainTarget_loader = torch.utils.data.DataLoader(
      trainTarget_dataset, batch_size=args.batchSize, shuffle=True,
      num_workers=args.workers if not args.test else 1, pin_memory=False, collate_fn=collate_fn_cat)


  if args.adda:
        tgt_model = train_discriminator(model, trainSource_loader, trainTarget_loader, args.epochs)
        torch.save({'epochs': args.epochs, 
                    'arch': args.arch, 
                    'state_dict': tgt_model.state_dict(), }, 
        args.save_path + '/adda_fitted{}.pth.tar'.format(args.epochs))
        return

  if args.dial_fit:
        loss_history = train_statistics(model, trainTarget_loader, args.epochs)
        torch.save({'epochs': args.epochs, 
                    'arch': args.arch, 
                    'state_dict': model.state_dict(), }, 
        args.save_path + '/dial_fitted{}.pth.tar'.format(args.epochs))
        np.save(args.save_path + '/dial_fitted{}.loss.txt'.format(args.epochs), 
                np.asarray([x.avg for x in loss_history]))
        return

  M = None
  if args.shapeWeight > ref.eps:
    print 'getY...'
    if args.dialModel:
      model.set_domain(source=False)
    Y = getY(SourceDataset('train', args.nViews))
    M = initLatent(trainTarget_loader, model, Y, nViews = args.nViews, S = args.sampleSource, AVG = args.AVG, dial=DIAL)
  
  print 'Start training...'
  
  for epoch in range(1, args.epochs + 1):
    adjust_learning_rate(optimizer, epoch, args.dropLR)
    if args.shapeWeight > ref.eps and args.dialModel:
          train_loader = fusion_loader
          len_loader = len(train_loader)
          train_mpjpe, train_loss, train_unSuploss = dial_train(args, (train_loader, len_loader), model, optimizer, M, epoch, dial=DIAL, nViews=args.nViews)
    else:
          train_loader = fusion_loader
          train_mpjpe, train_loss, train_unSuploss = train(args, train_loader, model, optimizer, M, epoch, dial=DIAL, nViews=args.nViews)
    if args.dialModel:
          model.set_domain(source=True)
    valSource_mpjpe, valSource_loss, valSource_unSuploss = validate(args, 'Source', valSource_loader, model, None, epoch)
    if args.dialModel:
          model.set_domain(source=False)
    valTarget_mpjpe, valTarget_loss, valTarget_unSuploss = validate(args, 'Target', valTarget_loader, model, None, epoch)

    if args.shapeWeight > ref.eps:
          trainTarget_loader.dataset.shuffle()
    else:
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
    
    if epoch % 20 == 0:
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
