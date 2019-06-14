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
import time
import ref
import cv2
import numpy as np
from datasets.Fusion import Fusion
from datasets.humans36m import Humans36mDataset

from utils.utils import createDirIfNonExistent
from utils.logger import Logger, log_parameters

from opts import opts
from train import train, validate, test, train_source_only, eval_source_only
from train_with_priors import  train_priors, validate_priors
from optim_latent import initLatent, stepLatent, getY, getYHumans
from model import getModel
from utils.utils import collate_fn_cat
from extract_priors import extract
from layers.prior_generator import compute_distances

from datasets.chairs_modelnet import ChairsModelNet as SourceDataset
from datasets.humans36m import Humans36mDepthSourceDataset, Humans36mDepthTargetDataset, Humans36mRGBTargetDataset, Humans36mRGBSourceDataset

args = opts().parse()

if args.sourceDataset =='ModelNet':
  from datasets.chairs_modelnet import ChairsModelNet as SourceDataset
elif args.sourceDataset == 'HumansRGB':
  from datasets.humans36m import Humans36mRGBSourceDataset as SourceDataset
elif args.sourceDataset == 'HumansDepth':
  from datasets.humans36m import Humans36mDepthSourceDataset as SourceDataset
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
  from datasets.humans36m import Humans36mRGBTargetDataset as TargetDataset
elif args.targetDataset == 'HumansDepth':
  from datasets.humans36m import Humans36mDepthTargetDataset as TargetDataset
else:
  raise Exception("No target dataset {}".format(args.targetDataset))

splits = ['train', 'valSource', 'valTarget']
kHumansDataset = ['HumansRGB', 'HumansDepth']
DIAL = args.approx_dial
refiners = ['adjacency', 'all']
def main():
  if args.distsRefiner is not None and args.distsRefiner not in refiners:
     print('ERROR: refiner %s does not match any refiner: ' % args.distsRefiner,
             refiners)
     return
  log_parameters(args)  
  call_count = 0
  total_runs = args.runs if not args.test else 1
  for run in range(total_runs):
      logger = SummaryWriter(args.logDir+'-run'+str(run))

      # Init model
      model = getModel(args)
      cudnn.benchmark = False

      # Init optimizer
      optimizer = torch.optim.SGD(model.parameters(), args.LR,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)


      # Init loaders
      # Couple sanity checks
      if args.targetDataset in kHumansDataset or args.sourceDataset in kHumansDataset:
            assert(ref.nViews <= 4)
            assert(args.nViews <= 4)
            assert(ref.J == 15)
            assert(ref.category == 'Human')
            assert(ref.nValViews <= 4)

      source_valViews = ref.nValViews if args.sourceDataset != 'HumansDepth' else 1
      target_valViews = ref.nValViews if args.targetDataset != 'HumansDepth' else 1

      valSource_dataset = SourceDataset('test', source_valViews)#, nImages=375) #subjects 5,6 Depth
      valSource_loader = torch.utils.data.DataLoader(valSource_dataset, batch_size = 1, 
                        shuffle=False, num_workers=1, pin_memory=True, collate_fn=collate_fn_cat)
      #valTarget_dataset = TargetDataset('test', target_valViews, nImages=375)
      #valTarget_dataset = Humans36mDepthSourceDataset('train', 1)#, nImages=375, meta=-5) #subjects 0,1,2 RGB
      #valTarget_loader = torch.utils.data.DataLoader(valTarget_dataset, batch_size = 1, 
      #                  shuffle=False, num_workers=1, pin_memory=True, collate_fn=collate_fn_cat)
      testTarget_dataset = TargetDataset('test', 1)#, nImages=375, meta=-5) #subject 5,6 RGB
      testTarget_loader = torch.utils.data.DataLoader(testTarget_dataset, batch_size = 1, 
                        shuffle=False, num_workers=1, pin_memory=True, collate_fn=collate_fn_cat)
      #valTrainTarget_dataset = TargetDataset('train', 1, meta=1) #subjects 3,4 RGB
      #valTrainTarget_loader = torch.utils.data.DataLoader(valTrainTarget_dataset, batch_size = 1, 
      #                  shuffle=False, num_workers=1, pin_memory=True, collate_fn=collate_fn_cat)
 

      if not args.shapeConsistency and not args.sourceOnly and not args.test:
            if args.propsOnly:
                  prior_loss = PriorRegressionCriterion(args.propsFile, norm = args.lossNorm, 
                                                         distances_refinement=args.distsRefiner, 
                                                         obj='props')
            elif args.distsOnly:
                  prior_loss = PriorRegressionCriterion(args.propsFile, norm = args.lossNorm, 
                                                         distances_refinement=args.distsRefiner, 
                                                         obj='dists')
            else:
                  prior_loss = PriorSMACOFCriterion(args.propsFile, norm = args.lossNorm, 
                                                     distances_refinement=args.distsRefiner, 
                                                     iterate=False, J=ref.J, rotation_weight=0, 
                                                     scale_weight=0, debug=args.DEBUG,  
                                                     debug_folder=args.debug_folder)
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

      train_dataset = Fusion(SourceDataset, TargetDataset, nViews = args.nViews, 
                              targetRatio = args.targetRatio, totalTargetIm = args.totalTargetIm)
      trainTarget_dataset = train_dataset.targetDataset

      fusion_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, 
                                                   shuffle=not args.test, 
                                                   num_workers=args.workers if not args.test else 1, 
                                                   pin_memory=False, collate_fn=collate_fn_cat)
      trainTarget_loader = torch.utils.data.DataLoader(trainTarget_dataset, batch_size=args.batchSize, 
                                                       shuffle=True,
                                                       num_workers=args.workers if not args.test else 1,
                                                       pin_memory=True, collate_fn=collate_fn_cat)

      trainSource_dataset = train_dataset.sourceDataset
      trainSource_loader = torch.utils.data.DataLoader(trainSource_dataset, batch_size=args.batchSize, 
                                                        shuffle=True,
                                                        num_workers=args.workers if not args.test else 1, 
                                                        pin_memory=True, collate_fn=collate_fn_cat)

      lambda_identity = lambda a: a 
      unnorm_net = lambda_identity
      unnorm_val_src = lambda_identity
      unnorm_val_tgt = lambda_identity
      unnorm_train_net = lambda_identity 
      unnorm_train_tgt = lambda_identity
      if ref.category == 'Human':
          unnorm_net = trainSource_dataset._unnormalize_pose
          #unnorm_net = valTarget_dataset._unnormalize_pose
          unnorm_val_src = valTarget_dataset._unnormalize_pose
          unnorm_val_tgt = valSource_dataset._unnormalize_pose
          if args.unnormalized:
              print('unnormalized')
              unnorm_train_net = trainSource_dataset._unnormalize_pose
              unnorm_train_tgt = trainTarget_dataset._unnormalize_pose 
      print('unnorm_train_net is identity: ', unnorm_train_net == lambda_identity)
      print('unnorm_train_tgt is identity: ', unnorm_train_tgt == lambda_identity)

      if not args.shapeConsistency and not args.sourceOnly:
            print('Initial validation on source')
            #validate_priors(args, 'val/Target_test', testTarget_loader, 
            #                 model, prior_loss, 0,
            #                 logger=logger, 
            #                 unnorm_net=unnorm_net, 
            #                 unnorm_tgt=unnorm_val_tgt)
            #print('initial validation on target')
            #validate_priors(args, 'val/Target_source', valTarget_loader, 
            #                 model, prior_loss, 0, plot_img=True, 
            #                 logger=logger, 
            #                 unnorm_net=unnorm_net, 
            #                 unnorm_tgt=unnorm_val_tgt)
            #print('initial validation on train-target')
            #validate_priors(args, 'val/Target_train', valTrainTarget_loader, 
            #                 model, prior_loss, 0, plot_img=False, 
            #                 logger=logger,
            #                 unnorm_net=unnorm_net, 
            #                 unnorm_tgt=unnorm_val_tgt)


      elif not args.sourceOnly:
            print('starting validation on target_012')
            #validate(args, 'val/Target_012', valTarget_loader, 
            #          model, None, 0, visualize=False, 
            #          logger=logger, 
            #          unnorm_net=unnorm_net, 
            #          unnorm_tgt=unnorm_val_tgt)
            #print('starting validation on target_56')
            #validate(args, 'val/Target_56', testTarget_loader, 
            #          model, None, 0, visualize=True, 
            #          logger=logger, 
            #          unnorm_net=unnorm_net, 
            #          unnorm_tgt=unnorm_val_tgt)
            #print('starting validation on source')
            #validate(args, 'val/Source', valSource_loader, 
            #          model, None, 0, 
            #          logger=logger,
            #          unnorm_net=unnorm_net, 
            #          unnorm_tgt=unnorm_val_src)


      M = None
      start = time.time()
      if args.shapeWeight > ref.eps and args.shapeConsistency:
            print 'getY...'
            if args.dialModel:
                  model.set_domain(source=False)
            Y, Y_raw = [None, None]
            if ref.category == 'Chair':
                  Y, Y_raw = getY(SourceDataset('train', args.nViews))
            else: 
                  Y = getYHumans(trainSource_dataset)
            #np.save('RotatedY-' + args.sourceDataset + '.npy', Y)
            #np.save('RotatedYRaw-' + args.sourceDataset + '.npy', Y_raw)
            #print 'RotatedY-' + args.sourceDataset + '.npy' + ' Was saved...'
            M = initLatent(trainTarget_loader, model, Y, 
                            nViews = args.nViews, S = args.sampleSource, 
                            AVG = args.AVG, dial=DIAL)
      print 'Start training...'
      for epoch in range(1, args.epochs + 1):
            adjust_learning_rate(optimizer, epoch, args.dropLR)
            if args.sourceOnly:
                  train_source_only(args, trainSource_loader, model, optimizer, epoch)
                  if epoch % 4 == 0:
                        mean, std = valTarget_dataset._get_normalization_statistics()
                        net_mean, net_std = train_dataset.sourceDataset._get_normalization_statistics()
                        eval_source_only(args, 'val/Target', valTarget_loader, model, 
                                          epoch, plot_img=True, logger=logger, 
                                          statistics=(mean, std), 
                                          net_statistics=(net_mean, net_std))
                        
            elif args.shapeConsistency:
                  if args.shapeWeight > ref.eps and args.dialModel:
                        train_loader = fusion_loader
                        len_loader = len(train_loader)
                        dial_train(args, (train_loader, len_loader), model, 
                                    optimizer, M, epoch, dial=DIAL, nViews=args.nViews)
                  else:
                        train_loader = fusion_loader
                        train(args, train_loader, model, optimizer, M, epoch, 
                               dial=DIAL, nViews=args.nViews)
                  if args.shapeWeight > ref.eps:
                        trainTarget_loader.dataset.shuffle()
                  else:
                        train_loader.dataset.targetDataset.shuffle()
                  if args.shapeWeight > ref.eps and epoch % args.intervalUpdateM == 0:
                        M = stepLatent(trainTarget_loader, model, M, Y, 
                                        nViews = args.nViews, lamb = args.lamb, 
                                        mu = args.mu, S = args.sampleSource, 
                                        call_count=call_count, dial=DIAL)
                        call_count += 1
                  #if epoch % 2 == 0:
                  #      validate(args, 'val/Target_012', valTarget_loader, model, 
                  #                None, epoch, visualize=False, 
                  #                logger=logger, 
                  #                unnorm_net=unnorm_net,
                  #                unnorm_tgt=unnorm_val_tgt)
                  #      validate(args, 'val/Target_56', testTarget_loader, 
                  #                model, None, epoch, visualize=True, 
                  #                logger=logger, 
                  #                unnorm_net=unnorm_net, 
                  #                unnorm_tgt=unnorm_val_tgt)

                  #if epoch % 5 == 0:
                  #      validate(args, 'val/Source', valSource_loader, model, 
                  #                None, epoch, 
                  #                logger=logger,
                  #                unnorm_net=unnorm_net,
                  #                unnorm_tgt=unnorm_val_src)
                        
            elif not args.sourceOnly:
                  train_priors(args, [trainTarget_loader], model, 
                                prior_loss, args.batch_norm, logger, 
                                optimizer, epoch-1, threshold = args.threshold,
                                unnorm_net=unnorm_train_net, 
                                unnorm_tgt=unnorm_train_tgt)
                  
                  #if epoch % 2 == 0:
                        #validate_priors(args, 'val/Target_source', valTarget_loader, 
                        #                 model, prior_loss, epoch, plot_img=True, 
                        #                 logger=logger,
                        #                 unnorm_net=unnorm_net, 
                        #                 unnorm_tgt=unnorm_val_tgt)

                        #validate_priors(args, 'val/Target_train', valTrainTarget_loader, 
                        #                 model, prior_loss, epoch, plot_img=False, 
                        #                 logger=logger,
                        #                 unnorm_net=unnorm_net, 
                        #                 unnorm_tgt=unnorm_val_tgt)

                  #if epoch % 5 == 0:
                        #validate_priors(args, 'val/Target_test', testTarget_loader, 
                        #                 model, prior_loss, epoch, 
                        #                 logger=logger,
                        #                 unnorm_net=unnorm_net, 
                        #                 unnorm_tgt=unnorm_val_src)
                        
            if epoch % 10 == 0:
                  torch.save({
                              'epoch': epoch + 1,
                              'arch': args.arch,
                              'state_dict': model.state_dict(),
                              'optimizer' : optimizer.state_dict(),
                              }, args.save_path+ '/checkpoint_{}.pth.tar'.format(epoch))
      end = time.time()
      print('Training endend')
      print('elapsed time: ', end - start)
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
