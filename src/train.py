import torch
import numpy as np
from utils.utils import AverageMeter, show3D
from utils.eval import accuracy, shapeConsistency, accuracy_dis
import cv2
import ref
from progress.bar import Bar
from layers.ShapeConsistencyCriterion import ShapeConsistencyCriterion

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

def dial_step(args, split, epoch, loader, model, optimizer = None, M = None, f = None, tag = None, dial=False, nViews=ref.nViews):
  losses, mpjpe, mpjpe_r = AverageMeter(), AverageMeter(), AverageMeter()
  viewLosses, shapeLosses, supLosses = AverageMeter(), AverageMeter(), AverageMeter()
  
  if split == 'train':
    model.train()
  else:
    model.eval()
  bar = Bar('{}'.format(ref.category), max=len(loader))
  
  for i, data in enumerate(loader):
    (sourceInput, sourceLabel, sourceMeta), (targetInput, targetLabel, targetMeta) = data
    source_input_var = torch.autograd.Variable(sourceInput.cuda())
    source_label_var = torch.autograd.Variable(sourceLabel)
    model.set_domain(source=True)
    source_output = model(source_input_var)
    source_loss = ShapeConsistencyCriterion(nViews, supWeight = 1, unSupWeight = args.shapeWeight, M = M)(source_output.cpu(), source_label_var, torch.autograd.Variable(sourceMeta))
    
    target_input_var = torch.autograd.Variable(targetInput.cuda())
    target_label_var = torch.autograd.Variable(targetLabel)
    model.set_domain(source=False)
    target_output = model(target_input_var)
    target_loss = ShapeConsistencyCriterion(nViews, supWeight = 1, unSupWeight = args.shapeWeight, M = M)(target_output.cpu(), target_label_var, torch.autograd.Variable(targetMeta))
    
    if split == 'train':
      optimizer.zero_grad()
      source_loss.backward()
      target_loss.backward()
      optimizer.step()
    
    input = torch.cat((sourceInput, targetInput), 0)
    target = torch.cat((sourceLabel, targetLabel), 0)
    output = torch.cat((source_output, target_output), 0)
    meta = torch.cat((sourceMeta, targetMeta), 0)
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

    losses.update(source_loss.data[0] + target_loss.data[0], input.size(0))
    shapeLosses.update(shapeLoss, input.size(0))
    mpjpe.update(mpjpe_this, input.size(0))
    mpjpe_r.update(mpjpe_r_this, input.size(0))
    
    
    Bar.suffix = '{split:10}: [{0:2}][{1:3}/{2:3}] | Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | shapeLoss {shapeLoss.avg:.6f} | AE {mpjpe.avg:.6f} | ShapeDis {mpjpe_r.avg:.6f}'.format(epoch, i, len(loader), total=bar.elapsed_td, eta=bar.eta_td, loss=losses, mpjpe=mpjpe, split = split, shapeLoss = shapeLosses, mpjpe_r = mpjpe_r)
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
