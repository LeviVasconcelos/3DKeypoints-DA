
import torch
import numpy as np
from utils.utils import AverageMeter, show3D
from utils.eval import accuracy, shapeConsistency, accuracy_dis
import cv2
import ref
from progress.bar import Bar
from layers.ShapeConsistencyCriterion import ShapeConsistencyCriterion

'''
input shape:torch.Size([64, 3, 224, 224])
target shape:torch.Size([64, 10, 3])
meta shape:torch.Size([64, 15])
'''

__DEBUG = False

def step(args, split, epoch, loader, model, optimizer = None, M = None, f = None, tag = None, dial=False, nViews=ref.nViews):
  losses, mpjpe, mpjpe_r = AverageMeter(), AverageMeter(), AverageMeter()
  viewLosses, shapeLosses, supLosses = AverageMeter(), AverageMeter(), AverageMeter()
  
  if split == 'train':
    model.train()
  else:
    model.eval()
  bar = Bar('{}'.format(ref.category), max=len(loader))
  
  nViews = loader.dataset.nViews
  if dial:
    print 'dial activated (from train function)'
    model.eval()
  for i, (input, target, meta) in enumerate(loader):
    if __DEBUG:
      print "input shape:" + str(input.size())
      print "target shape:" + str(target.size())
      print "meta shape:" + str(meta.size())
    
    input_var = torch.autograd.Variable(input.cuda())
    target_var = torch.autograd.Variable(target)
    output = model(input_var)
    loss = ShapeConsistencyCriterion(nViews, supWeight = 1, unSupWeight = args.shapeWeight, M = M)(output.cpu(), target_var, torch.autograd.Variable(meta))

    if split == 'test':
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
          f.write('\n')

    mpjpe_this = accuracy(output.data, target, meta)
    mpjpe_r_this = accuracy_dis(output.data, target, meta)
    shapeLoss = shapeConsistency(output.data, meta, nViews, M, split = split)

    losses.update(loss.data[0], input.size(0))
    shapeLosses.update(shapeLoss, input.size(0))
    mpjpe.update(mpjpe_this, input.size(0))
    mpjpe_r.update(mpjpe_r_this, input.size(0))
    
    if split == 'train':
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    Bar.suffix = '{split:10}: [{0:2}][{1:3}/{2:3}] | Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | shapeLoss {shapeLoss.avg:.6f} | AE {mpjpe.avg:.6f} | ShapeDis {mpjpe_r.avg:.6f}'.format(epoch, i, len(loader), total=bar.elapsed_td, eta=bar.eta_td, loss=losses, mpjpe=mpjpe, split = split, shapeLoss = shapeLosses, mpjpe_r = mpjpe_r)
    bar.next()
      
  bar.finish()
  return mpjpe.avg, losses.avg, shapeLosses.avg

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
          source_input_var = torch.autograd.Variable(sourceInput.cuda())
          source_label_var = torch.autograd.Variable(sourceLabel)
          model.set_domain(source=True)
          source_output = model(source_input_var)
          source_loss = ShapeConsistencyCriterion(nViews, supWeight = 1, unSupWeight = args.shapeWeight, M = M)(source_output.cpu(), source_label_var, torch.autograd.Variable(sourceMeta))
          if split == 'train':
                source_loss.backward()
          source_loss_value = source_loss.data[0]
          del source_loss
    if (len(target_input_list) > 0):
          target_input_var = torch.autograd.Variable(targetInput.cuda())
          target_label_var = torch.autograd.Variable(targetLabel)
          model.set_domain(source=False)
          target_output = model(target_input_var)
          target_loss = ShapeConsistencyCriterion(nViews, supWeight = 1, unSupWeight = args.shapeWeight, M = M)(target_output.cpu(), target_label_var, torch.autograd.Variable(targetMeta))
          if split == 'train':
                target_loss.backward()
          target_loss_value = target_loss.data[0]
          del target_loss
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
    if split == 'test':
      for j in range(input_.numpy().shape[0]):
        img = (input_.numpy()[j] * 255).transpose(1, 2, 0).astype(np.uint8)
        cv2.imwrite('{}/img_{}/{}.png'.format(args.save_path, tag, i * input_.numpy().shape[0] + j), img)
        gt = target_.cpu().numpy()[j]
        pred = (output_.data).cpu().numpy()[j]
        vis = meta_.cpu().numpy()[j][5:]
        for t in range(ref.J):
          f.write('{} {} {} '.format(pred[t * 3], pred[t * 3 + 1], pred[t * 3 + 2]))
        f.write('\n')
        for t in range(ref.J):
          f.write('{} {} {} '.format(gt[t, 0], gt[t, 1], gt[t, 2]))
        f.write('\n')
        if args.saveVis:
          for t in range(ref.J):
            f.write('{} 0 0 '.format(vis[t]))
          f.write('\n')

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

def dial_train(args, train_loader, model, optimizer, M, epoch, dial=False, nViews=ref.nViews):
      return dial_step(args, 'train', epoch, train_loader, model, optimizer, M = M, dial=dial)

def train(args, train_loader, model, optimizer, M, epoch, dial=False, nViews=ref.nViews):
  return step(args, 'train', epoch, train_loader, model, optimizer, M = M, dial=dial)

def validate(args, supTag, val_loader, model, M, epoch):
  return step(args, 'val' + supTag, epoch, val_loader, model, M = M)

def test(args, loader, model, M, f, tag):
  return step(args, 'test', 0, loader, model, M = M, f = f, tag = tag)
