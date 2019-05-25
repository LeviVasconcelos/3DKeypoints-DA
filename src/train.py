
import torch
import numpy as np
from utils.utils import AverageMeter
from utils.visualization import chair_show3D, chair_show2D, human_show2D, human_show3D, human_from_3D 
from utils.eval import accuracy, shapeConsistency, accuracy_dis
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import ref
from progress.bar import Bar
from layers.ShapeConsistencyCriterion import ShapeConsistencyCriterion
from datasets.Fusion import unpack_splitted

'''
input shape:torch.Size([64, 3, 224, 224])
target shape:torch.Size([64, 10, 3])
meta shape:torch.Size([64, 15])
'''

__DEBUG = False

def source_only_train_step(args, epoch, loader, model, optimizer = None, device = 'cuda'):
      model.train()
      regression_loss = []
      bar = Bar('{}'.format(ref.category), max=len(loader))
      accumulate_loss = 0.
      count_loss = 0.
      L1_crit = torch.nn.L1Loss()
      for i, (input, target, meta, _, _) in enumerate(loader):
            input_var = input.to(device)
            target_var = target.to(device)
            output = model(input_var)
            
            #loss = ((output - target_var.view(target_var.shape[0],-1)) ** 2).sum() / ref.J / 3 / input.shape[0]
            loss = torch.abs(output - target_var.view(target_var.shape[0],-1)).sum() / ref.J / 3 / input.shape[0]
            regression_loss.append(loss.item())
            accumulate_loss += loss.item()
            count_loss += 1.
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            Bar.suffix = 'Epoch [%d] (%d/%d) Loss: %.5f' % (epoch, i, len(loader), accumulate_loss/count_loss)
            bar.next()
            
      bar.finish()
      return np.array(regression_loss).mean()


def source_only_eval(args, ds_split, epoch, loader, model, plot_img = False, logger = None, device='cuda', statistics=None, net_statistics=None):

      regr_loss = []
      accuracy_this = []
      accuracy_shape = []
      
      device = 'cuda'
      model.eval()
      mean, std = statistics
      mean = torch.from_numpy(mean).float().to(device)
      std = torch.from_numpy(std).float().to(device)
      net_mean, net_std = net_statistics
      net_mean = torch.from_numpy(net_mean).float().to(device)
      net_std = torch.from_numpy(net_std).float().to(device)

      for i, (input, target, meta, _, _) in enumerate(loader):
            input_var = input.to(device)
            target_var = target.to(device)
            output = model(input_var)
            
            #loss = ((output - target_var.view(target_var.shape[0],-1)) ** 2).sum() / ref.J / 3 / input.shape[0]
            loss = torch.abs(output - target_var.view(target_var.shape[0],-1)).sum() / ref.J / 3 / input.shape[0]
            unormed_output = (output.view(input.shape[0], ref.J, 3) * net_std) + net_mean
            unormed_target = (target_var * std) + mean
            current_acc = accuracy(unormed_output.view(target_var.shape[0], -1).data, unormed_target.data, meta)
            current_acc_shape = accuracy_dis(unormed_output.view(target_var.shape[0], -1).data, unormed_target.data, meta)
            
            accuracy_this.append(current_acc.item())
            accuracy_shape.append(current_acc_shape.item())
            regr_loss.append(loss.item())
            
            if plot_img:
                  draw_2d = chair_show2D if ref.category == 'Chair' else human_show2D
                  draw_3d = chair_show3D if ref.category == 'Chair' else human_show3D
                  #numpy_img = (input.numpy()[0] * 255).transpose(1, 2, 0).astype(np.uint8)
                  #filename_2d = os.path.join(args.save_path, 'img2d_%s_%d_%d.png' % (args.expID, i, epoch))
                  #cv2.imwrite(filename_2d, numpy_img)
                  if i < 10:
                        pred = unormed_output.data.cpu().numpy()[0].copy()
                        gt = unormed_target.data.cpu().numpy()[0].copy()
                        #numpy_img = draw_2d(numpy_img, pred, (255,0,0))
                        #numpy_img = draw_2d(numpy_img, gt, (0,0,255))
                        #filename_2d = os.path.join(args.save_path, 'img2d_%s_%d_%d.png' % (args.expID, i, epoch))
                        #cv2.imwrite(filename_2d, numpy_img)
                        fig = plt.figure()
                        ax = fig.add_subplot((111), projection='3d')
                        draw_3d(ax, pred, 'r')
                        draw_3d(ax, gt, 'b')
                        #TODO: make it directly to numpy to avoid disk IO
                        filename_3d = os.path.join(args.save_path, 'img3d_%s_%d_%d.png' % (args.expID, i, epoch))
                        plt.savefig(filename_3d)
                        logger.add_image('Image 3D ' + str(i), (np.asarray(Image.open(filename_3d))).transpose(2,0,1), epoch)
                        plt.close()
                        #logger.add_image('Image 2D ' + str(i), (np.asarray(Image.open(filename_2d))).transpose(2,0,1), epoch)
      regr_loss_mean = np.array(regr_loss).mean()
      accuracy_mean = np.array(accuracy_this).mean()
      accuracy_shape_mean =  np.array(accuracy_shape).mean()
      if 'val' in ds_split:
          tag = ds_split.split('/')[-1]   
          logger.add_scalar('val/' + tag + '-accuracy', accuracy_mean, epoch)
          logger.add_scalar('val/' + tag + '-regr-loss', regr_loss_mean, epoch)
          logger.add_scalar('val/' + tag + '-unsup-loss', accuracy_shape_mean, epoch)
                
      return regr_loss_mean, accuracy_mean, accuracy_shape_mean

def step(args, ds_split, epoch, loader, model, optimizer = None, M = None, f = None, tag = None, dial=False, nViews=ref.nViews, visualize=False, logger=None, unnorm_net=(lambda pose:pose), unnorm_tgt=(lambda pose:pose)):
  losses, mpjpe, mpjpe_r = AverageMeter(), AverageMeter(), AverageMeter()
  viewLosses, shapeLosses, supLosses = AverageMeter(), AverageMeter(), AverageMeter()
  
  if ds_split == 'train':
    model.train()
  else:
    model.eval()
  bar = Bar('{}'.format(ref.category), max=len(loader))
  
  nViews = loader.dataset.nViews
  if dial:
    print 'dial activated (from train function)'
    model.eval()
  numpy_imgs = None
  for i, (input, target, meta, uncentred, intrinsics) in enumerate(loader):
    if __DEBUG:
      print "input shape:" + str(input.size())
      print "target shape:" + str(target.size())
      print "meta shape:" + str(meta.size())
    
    ((sourceInput, sourceLabel, sourceMeta), (targetInput, targetLabel, targetMeta)) = unpack_splitted((input, target, meta))
    #input_var = input.cuda()
    #target_var = target
    source_mpjpe_this, source_mpjpe_r_this, source_shapeLoss = 0, 0, 0
    target_mpjpe_this, target_mpjpe_r_this, target_shapeLoss = 0, 0, 0
    loss_src, loss_tgt = 0, 0
    if sourceInput.nelement() > 0:
        sourceOutput = model(sourceInput)
        sourceOutput = unnorm_net(sourceOutput.view(sourceOutput.shape[0], 
                                  ref.J, 3)).view(sourceOutput.shape[0], -1)
        sourceLabel = unnorm_tgt(sourceLabel.cuda()).cpu()
        loss_src = ShapeConsistencyCriterion(1, supWeight = 1, unSupWeight = args.shapeWeight, 
                                              M = M)(sourceOutput.cpu(), sourceLabel, sourceMeta)
        source_mpjpe_this = accuracy(sourceOutput.data, sourceLabel.data, sourceMeta)
        source_mpjpe_r_this = accuracy_dis(sourceOutput.data, sourceLabel.data, sourceMeta)
        source_shapeLoss = shapeConsistency(sourceOutput.data, sourceMeta, 1, M, split = ds_split)

    if targetInput.nelement() > 0:
        targetOutput = model(targetInput)
        targetOutput = unnorm_net(targetOutput.view(targetOutput.shape[0], 
                                  ref.J, 3)).view(targetOutput.shape[0], -1)
        targetLabel = unnorm_tgt(targetLabel.cuda()).cpu()
        loss_tgt = ShapeConsistencyCriterion(nViews, supWeight = 1, unSupWeight = args.shapeWeight, 
                                               M = M)(targetOutput.cpu(), targetLabel, targetMeta)
        target_mpjpe_this = accuracy(targetOutput.data, targetLabel.data, targetMeta)
        target_mpjpe_r_this = accuracy_dis(targetOutput.data, targetLabel.data, targetMeta)
        target_shapeLoss = shapeConsistency(targetOutput.data, targetMeta, nViews, M, split = ds_split)

   
    loss = loss_tgt + loss_src
    shapeLoss = target_shapeLoss + source_shapeLoss 
    mpjpe_this = target_mpjpe_this + source_mpjpe_this
    mpjpe_r_this = target_mpjpe_r_this + source_mpjpe_r_this
    
    '''if ds_split == 'test':
      for j in range(input.numpy().shape[0]):
        img = (input.numpy()[j] * 255).transpose(1, 2, 0).astype(np.uint8)
        cv2.imwrite('{}/img_{}/{}.png'.format(args.save_path, tag, i * input.numpy().shape[0] + j), img)
        gt = target.cpu().numpy()[j]
        pred = (output.data).cpu().numpy()[j]
        vis = meta.cpu().numpy()[j][5:]
        for t in range(ref.J):
          f.write('{} {} {} '.format(pred[t * 3], pred[t * 3 + 1], pred[t * 3 + 2]))
        f.write('\n')
        for t in range(ref.J):
          f.write('{} {} {} '.format(gt[t, 0], gt[t, 1], gt[t, 2]))
        f.write('\n')
        if args.saveVis:
          for t in range(ref.J):
            f.write('{} 0 0 '.format(vis[t]))
          f.write('\n')'''
    if visualize:
          draw_2d = chair_show2D if ref.category == 'Chair' else human_show2D
          draw_3d = chair_show3D if ref.category == 'Chair' else human_show3D
          numpy_img = (input.numpy()[0] * 255).transpose(1, 2, 0).astype(np.uint8)
          filename_2d = os.path.join(args.save_path, 'img2d_%s_%d_%d.png' % (args.expID, i, epoch))
          #cv2.imwrite(filename_2d, numpy_img)
          if i < 10:
                camera = 0 
                pred = targetOutput.data[camera].cpu().view(ref.J, 3).numpy().copy()
                gt = targetLabel.data[camera].cpu().numpy().copy()
                gt_uncentred = uncentred.data[camera].cpu().numpy().copy()
                #pred = targetOutput.data.cpu().numpy()[0].copy()
                #gt = targetLabel.data.cpu().numpy()[0].copy()
                #numpy_img = draw_2d(numpy_img, pred, (255,0,0))
                #numpy_img = draw_2d(numpy_img, gt, (0,0,255))
                numpy_img = human_from_3D(numpy_img, gt_uncentred, intrinsics[camera],
                                    (180,0,0), 224./1000.)
                numpy_img = human_from_3D(numpy_img, pred - gt_uncentred[0], intrinsics[camera], 
                                    (0,0,180), 224./1000., flip=True)
                filename_2d = os.path.join(args.save_path, 'img2d_%s_%d_%d.png' % (args.expID, i, epoch))
                cv2.imwrite(filename_2d, numpy_img)
                fig = plt.figure()
                ax = fig.add_subplot((111), projection='3d')
                draw_3d(ax, pred, 'r')
                draw_3d(ax, gt, 'b')
                #TODO: make it directly to numpy to avoid disk IO
                filename_3d = os.path.join(args.save_path, 'img3d_%s_%d_%d.png' % (args.expID, i, epoch))
                plt.savefig(filename_3d)
                logger.add_image('Image 3D ' + str(i), (np.asarray(Image.open(filename_3d))).transpose(2,0,1), epoch)
                logger.add_image('Image 2D ' + str(i), (np.asarray(Image.open(filename_2d))).transpose(2,0,1), epoch)
                plt.close()

    losses.update(loss.item(), input.size(0))
    shapeLosses.update(shapeLoss, input.size(0))
    mpjpe.update(mpjpe_this, input.size(0))
    mpjpe_r.update(mpjpe_r_this, input.size(0))
    
    if ds_split == 'train':
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    if 'val' in ds_split:
      tag = ds_split.split('/')[-1]
      logger.add_scalar('val/' + tag + '-accuracy', mpjpe_this, epoch)
      logger.add_scalar('val/' + tag + '-regr-loss', losses.avg, epoch)
      logger.add_scalar('val/' + tag + '-unsup-loss', mpjpe_r_this, epoch)
 
    
    Bar.suffix = '{split:10}: [{0:2}][{1:3}/{2:3}] | Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | shapeLoss {shapeLoss.avg:.6f} | AE {mpjpe.avg:.6f} | ShapeDis {mpjpe_r.avg:.6f}'.format(epoch, i, len(loader), total=bar.elapsed_td, eta=bar.eta_td, loss=losses, mpjpe=mpjpe, split = ds_split, shapeLoss = shapeLosses, mpjpe_r = mpjpe_r)
    bar.next()
      
  bar.finish()
  return mpjpe.avg, losses.avg, shapeLosses.avg

#TODO: Refactor using unpack_splitted
def dial_step(args, split, epoch, (loader, len_loader), model, optimizer = None, M = None, f = None, tag = None, dial=False, nViews=ref.nViews):
  losses, mpjpe, mpjpe_r = AverageMeter(), AverageMeter(), AverageMeter()
  viewLosses, shapeLosses, supLosses = AverageMeter(), AverageMeter(), AverageMeter()
  
  if split == 'train':
    model.train()
  else:
    model.eval()
  bar = Bar('{}'.format(ref.category), max=len_loader)
  
  for i, (input, target, meta) in enumerate(loader):
    if split == 'train':
          optimizer.zero_grad()
    sourceInput, sourceLabel, sourceMeta = torch.Tensor(), torch.Tensor(), torch.Tensor()
    targetInput, targetLabel, targetMeta = torch.Tensor(), torch.Tensor(), torch.Tensor()
    source_output, target_output = torch.Tensor(), torch.Tensor()
    source_loss_value, target_loss_value = 0, 0
    inputDomains = [meta[i,0] for i in range(input.shape[0])]
    source_input_list = [x for i,x in enumerate(input) if abs(meta[i,0]) == 1]
    if (len(source_input_list) > 0): 
          sourceInput = torch.stack(source_input_list)
          sourceLabel = torch.stack([x for i,x in enumerate(target) if abs(meta[i,0]) == 1])
          sourceMeta = torch.stack([x for i,x in enumerate(meta) if abs(meta[i,0]) == 1])
          sourceDomains = [sourceMeta[i,0] for i in range(sourceMeta.shape[0])]
          if __DEBUG:
            print "source input shape: " + str(sourceInput.size())
            print "source label shape: " + str(sourceLabel.size())
            print "source meta shape: " + str(sourceMeta.size())
            print "source domain: " + str(sourceDomains)
    
    target_input_list = [x for i,x in enumerate(input) if abs(meta[i,0]) > 1]
    if (len(target_input_list) > 0):
          targetInput = torch.stack(target_input_list)
          targetLabel = torch.stack([x for i,x in enumerate(target) if abs(meta[i,0]) > 1])
          targetMeta = torch.stack([x for i,x in enumerate(meta) if abs(meta[i,0]) > 1])
          targetDomains = [targetMeta[i,0] for i in range(targetMeta.shape[0])]
          if __DEBUG:
            print "targe input shape:" + str(targetInput.size())
            print "target label shape:" + str(targetLabel.size())
            print "target meta shape:" + str(targetMeta.size())
            print "target Domain: " + str(targetDomains)
    
    if (len(source_input_list) > 0):
          source_input_var = sourceInput.cuda()
          source_label_var = sourceLabel
          model.set_domain(source=True)
          source_output = model(source_input_var)
          source_loss = ShapeConsistencyCriterion(nViews, supWeight = 1, unSupWeight = args.shapeWeight, M = M)(source_output.cpu(), source_label_var, sourceMeta)
          if split == 'train':
                source_loss.backward()
          source_loss_value = source_loss.data[0]
          #del source_loss
    if (len(target_input_list) > 0):
          target_input_var = targetInput.cuda()
          target_label_var = targetLabel
          model.set_domain(source=False)
          target_output = model(target_input_var)
          target_loss = ShapeConsistencyCriterion(nViews, supWeight = 1, unSupWeight = args.shapeWeight, M = M)(target_output.cpu(), target_label_var, targetMeta)
          if split == 'train':
                target_loss.backward()
          target_loss_value = target_loss.data[0]
          #del target_loss
    if split == 'train':
          optimizer.step()
          
    
    input_ = torch.cat((sourceInput, targetInput), 0)
    target_ = torch.cat((sourceLabel, targetLabel), 0)
    if (source_output.nelement() > 0 and target_output.nelement() > 0):
          output_ = torch.cat((source_output, target_output), 0)
    else:
          output_ = source_output if source_output.nelement() > 0 else target_output
    if (sourceMeta.nelement() > 0 and targetMeta.nelement() > 0):
          meta_ = torch.cat((sourceMeta, targetMeta), 0)
    else:
          meta_ = sourceMeta if sourceMeta.nelement() > 0 else targetMeta

    mpjpe_this = accuracy(output_.data, target_, meta_)
    mpjpe_r_this = accuracy_dis(output_.data, target_, meta_)
    shapeLoss = shapeConsistency(output_.data, meta_, nViews, M, split = split)

    losses.update(source_loss_value + target_loss_value, input_.size(0))
    shapeLosses.update(shapeLoss, input_.size(0))
    mpjpe.update(mpjpe_this, input_.size(0))
    mpjpe_r.update(mpjpe_r_this, input_.size(0))
    
    
    Bar.suffix = '{split:10}: [{0:2}][{1:3}/{2:3}] | Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | shapeLoss {shapeLoss.avg:.6f} | AE {mpjpe.avg:.6f} | ShapeDis {mpjpe_r.avg:.6f}'.format(epoch, i, len_loader, total=bar.elapsed_td, eta=bar.eta_td, loss=losses, mpjpe=mpjpe, split = split, shapeLoss = shapeLosses, mpjpe_r = mpjpe_r)
    bar.next()
      
  bar.finish()
  return mpjpe.avg, losses.avg, shapeLosses.avg

def train_source_only(args, train_loader, model, optimizer, epoch):
      return source_only_train_step(args, epoch, train_loader, model, optimizer, device = 'cuda')

def eval_source_only(args, ds_split, val_loader, model, epoch, plot_img=False, logger=None, statistics=None, net_statistics=None):
      return source_only_eval(args, ds_split, epoch, val_loader, model, plot_img = plot_img, logger = logger, statistics=statistics, net_statistics=net_statistics)

def train(args, train_loader, model, optimizer, M, epoch, dial=False, nViews=ref.nViews, unnorm_net=(lambda pose:pose), unnorm_tgt=(lambda pose:pose)):
  return step(args, 'train', epoch, train_loader, model, optimizer, M = M, dial=dial, unnorm_net=unnorm_net, unnorm_tgt=unnorm_tgt)

def validate(args, supTag, val_loader, model, M, epoch, visualize=False, logger=None, unnorm_net=(lambda pose:pose), unnorm_tgt=(lambda pose:pose)):
  return step(args, 'val' + supTag, epoch, val_loader, model, M = M, visualize=visualize, logger=logger, unnorm_net=unnorm_net, unnorm_tgt=unnorm_tgt)

def test(args, loader, model, M, f, tag):
  return step(args, 'test', 0, loader, model, M = M, f = f, tag = tag)
