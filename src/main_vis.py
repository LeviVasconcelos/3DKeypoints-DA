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

from layers.PriorConsistencyCriterion import PriorRegressionCriterion, PriorSMACOFCriterion

from tensorboardX import SummaryWriter

import ref
import cv2
import numpy as np
from datasets.Fusion import Fusion
from datasets.humans36m import Humans36mDataset

from utils.utils import createDirIfNonExistent
from utils.logger import Logger

from opts import opts
from train import train, validate, test
from train_with_priors import  train_priors, validate_priors
from optim_latent import initLatent, stepLatent, getY
from model import getModel
from utils.utils import collate_fn_cat
from extract_priors import extract
from layers.prior_generator import compute_distances

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
kHumansDataset = ['HumansRGB', 'HumansDepth']
DIAL = args.approx_dial
def main():
  total_runs = args.runs if not args.test else 1
  for run in range(total_runs):
      logger = SummaryWriter(args.logDir+'-run'+str(run))

      # Init model
      model = getModel(args)
      cudnn.benchmark = True

      # Init optimizer
      optimizer = torch.optim.SGD(model.parameters(), args.LR,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)


      # Init loaders
      # Couple sanity checks
      if args.targetDataset in kHumansDataset or args.sourceDataset in kHumansDataset:
            assert(ref.nViews <= 4)
            assert(args.nViews <= 4)
            assert(ref.J == 32)
            assert(ref.category == 'Human')
            assert(ref.nValViews <= 4)

      source_valViews = ref.nValViews if args.sourceDataset != 'HumansDepth' else 1
      target_valViews = ref.nValViews if args.targetDataset != 'HumansDepth' else 1

      valSource_dataset = SourceDataset('test', source_valViews)
      valTarget_dataset = TargetDataset('test', target_valViews)
      valSource_loader = torch.utils.data.DataLoader(valSource_dataset, batch_size = 1, 
                        shuffle=False, num_workers=1, pin_memory=True, collate_fn=collate_fn_cat)
      valTarget_loader = torch.utils.data.DataLoader(valTarget_dataset, batch_size = 1, 
                        shuffle=False, num_workers=1, pin_memory=True, collate_fn=collate_fn_cat)
      
      
      if not args.shapeConsistency:
            if args.propsOnly:
                  prior_loss = PriorRegressionCriterion(args.propsFile, norm = args.lossNorm, distances_refinement=args.distsRefiner, obj='props')
            else:
                  prior_loss = PriorSMACOFCriterion(args.propsFile, norm = args.lossNorm, distances_refinement=args.distsRefiner, iterate=False, J=ref.J)

      if args.test:
            if not args.shapeConsistency:
                  test(args, valTarget_loader, model, prior_loss, None)
            else:
                  f = {}
                  for split in splits:
                        f['{}'.format(split)] = open('{}/{}.txt'.format(args.save_path, split), 'w')
                  test(args, valTarget_loader, model, None, f['valTarget'], 'valTarget')
            return
      
      if args.extractProps:
            sourceDist, sourceProps = extract(args, valSource_loader, model)
            #targetDist, targetProps = extract(args, valTarget_loader, model)
            np.save(args.propsFile+'-source-distances.npy', sourceDist)
            np.save(args.propsFile+'-source-props.npy', sourceProps)
            #np.save(args.propsFile+'-distances.npy', targetDist)
            #np.save(args.propsFile+'-props.npy', targetProps)
            return

      train_dataset = Fusion(SourceDataset, TargetDataset, nViews = args.nViews, targetRatio = args.targetRatio, totalTargetIm = args.totalTargetIm)
      trainTarget_dataset = train_dataset.targetDataset

      fusion_loader = torch.utils.data.DataLoader(
              train_dataset, batch_size=args.batchSize, shuffle=not args.test,
              num_workers=args.workers if not args.test else 1, pin_memory=False, collate_fn=collate_fn_cat)      
      trainTarget_loader = torch.utils.data.DataLoader(
              trainTarget_dataset, batch_size=args.batchSize, shuffle=True,
              num_workers=args.workers if not args.test else 1, pin_memory=True, collate_fn=collate_fn_cat)

      trainSource_dataset = train_dataset.sourceDataset
      trainSource_loader = torch.utils.data.DataLoader(
          trainSource_dataset, batch_size=args.batchSize, shuffle=True,
          num_workers=args.workers if not args.test else 1, pin_memory=True, collate_fn=collate_fn_cat)
      if not args.shapeConsistency:
            valSource_mpjpe, valSource_shape,  valSource_loss, valSource_unSuploss = validate_priors(args, 'Source', valSource_loader, model, prior_loss, 0)
            valTarget_mpjpe, valTarget_shape, valTarget_loss, valTarget_unSuploss = validate_priors(args, 'Target', valTarget_loader, model, prior_loss, 0, plot_img=True, logger=logger)
            logger.add_scalar('val/source-accuracy', valSource_mpjpe, 0)
            logger.add_scalar('val/target-accuracy', valTarget_mpjpe, 0)
            
            logger.add_scalar('val/target-accuracy', valTarget_mpjpe, 0)
            logger.add_scalar('val/target-accuracy-shape', valTarget_shape, 0)
            
            logger.add_scalar('val/source-regr-loss', valSource_loss, 0)
            logger.add_scalar('val/target-regr-loss', valTarget_loss, 0)
            
            logger.add_scalar('val/source-prior-loss', valSource_unSuploss, 0)
            logger.add_scalar('val/target-prior-loss', valTarget_unSuploss, 0)
      else:
            valTarget_mpjpe, valTarget_loss, valTarget_unSuploss = validate(args, 'Target', valTarget_loader, model, None, epoch, visualize=True)
            logger.add_scalar('val/target-accuracy', valTarget_mpjpe, epoch)
            logger.add_scalar('val/target-regr-loss', valTarget_loss, epoch)
            logger.add_scalar('val/target-prior-loss', valTarget_unSuploss, epoch)
            valSource_mpjpe, valSource_loss, valSource_unSuploss = validate(args, 'Source', valSource_loader, model, None, epoch)
            logger.add_scalar('val/source-accuracy', valTarget_mpjpe, epoch)
            logger.add_scalar('val/source-regr-loss', valTarget_loss, epoch)
            logger.add_scalar('val/source-prior-loss', valTarget_unSuploss, epoch)



      M = None
      if args.shapeWeight > ref.eps and args.shapeConsistency:
            print 'getY...'
            if args.dialModel:
                  model.set_domain(source=False)
            Y, Y_raw = getY(SourceDataset('train', args.nViews))
            np.save('RotatedY-' + args.sourceDataset + '.npy', Y)
            np.save('RotatedYRaw-' + args.sourceDataset + '.npy', Y_raw)
            print 'RotatedY-' + args.sourceDataset + '.npy' + ' Was saved...'
            M = initLatent(trainTarget_loader, model, Y, nViews = args.nViews, S = args.sampleSource, AVG = args.AVG, dial=DIAL)
      print 'Start training...'
      for epoch in range(1, args.epochs + 1):
            adjust_learning_rate(optimizer, epoch, args.dropLR)
            if args.shapeConsistency:
                  if args.shapeWeight > ref.eps and args.dialModel:
                        train_loader = fusion_loader
                        len_loader = len(train_loader)
                        train_mpjpe, train_loss, train_unSuploss = dial_train(args, (train_loader, len_loader), model, optimizer, M, epoch, dial=DIAL, nViews=args.nViews)
                  else:
                        train_loader = fusion_loader
                        train_mpjpe, train_loss, train_unSuploss = train(args, train_loader, model, optimizer, M, epoch, dial=DIAL, nViews=args.nViews)
                  if args.shapeWeight > ref.eps:
                        trainTarget_loader.dataset.shuffle()
                  else:
                        train_loader.dataset.targetDataset.shuffle()
                  if args.shapeWeight > ref.eps and epoch % args.intervalUpdateM == 0:
                        M = stepLatent(trainTarget_loader, model, M, Y, nViews = args.nViews, lamb = args.lamb, mu = args.mu, S = args.sampleSource, call_count=call_count, dial=DIAL)
                        call_count += 1
                  if epoch % 2 == 0:
                        valTarget_mpjpe, valTarget_loss, valTarget_unSuploss = validate(args, 'Target', valTarget_loader, model, None, epoch, visualize=True)
                        logger.add_scalar('val/target-accuracy', valTarget_mpjpe, epoch)
                        logger.add_scalar('val/target-regr-loss', valTarget_loss, epoch)
                        logger.add_scalar('val/target-prior-loss', valTarget_unSuploss, epoch)
                  if epoch % 5 == 0:
                        valSource_mpjpe, valSource_loss, valSource_unSuploss = validate(args, 'Source', valSource_loader, model, None, epoch)
                        logger.add_scalar('val/source-accuracy', valTarget_mpjpe, epoch)
                        logger.add_scalar('val/source-regr-loss', valTarget_loss, epoch)
                        logger.add_scalar('val/source-prior-loss', valTarget_unSuploss, epoch)
                        
            else:
                  train_priors(args, [trainTarget_loader], model, prior_loss, args.batch_norm, logger, optimizer, epoch-1, threshold = args.threshold)
                  
                  if epoch % 2 == 0:
                        valTarget_mpjpe, valTarget_shape, valTarget_loss, valTarget_unSuploss = validate_priors(args, 'Target', valTarget_loader, model, prior_loss, epoch, plot_img=True, logger=logger)
                        logger.add_scalar('val/target-accuracy', valTarget_mpjpe, epoch)
                        logger.add_scalar('val/target-accuracy-shape', valTarget_shape, epoch)
                        logger.add_scalar('val/target-regr-loss', valTarget_loss, epoch)
                        logger.add_scalar('val/target-prior-loss', valTarget_unSuploss, epoch)
                  
                  if epoch % 5 == 0:
                        valSource_mpjpe, valSource_shape, valSource_loss, valSource_unSuploss = validate_priors(args, 'Source', valSource_loader, model, prior_loss, epoch)
                        logger.add_scalar('val/source-accuracy', valSource_mpjpe, epoch)
                        logger.add_scalar('val/source-prior-loss', valSource_unSuploss, epoch)
                        logger.add_scalar('val/source-regr-loss', valSource_loss, epoch)
                        
            if epoch % 10 == 0:
                  torch.save({
                              'epoch': epoch + 1,
                              'arch': args.arch,
                              'state_dict': model.state_dict(),
                              'optimizer' : optimizer.state_dict(),
                              }, args.save_path+ '/checkpoint_{}.pth.tar'.format(epoch))
      print('Training endend')
      logger.close()


def cast_dists(x,J=10,cuda=True):
    r = torch.FloatTensor(x.shape[0],J*(J-1)/2).zero_() 
    if cuda:
        r = r.cuda()

    c = 0
    for i in range(J):
        for j in range(i+1,J):
            r[:,c] = x.data[:,i,j]
            c+=1
    return torch.autograd.Variable(r)
    
def adjust_learning_rate(optimizer, epoch, dropLR):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = args.LR * (0.1 ** (epoch // dropLR))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


if __name__ == '__main__':
  main()
