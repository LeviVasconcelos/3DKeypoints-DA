
import torch
import numpy as np
from utils.utils import AverageMeter, show3D
from utils.eval import accuracy, shapeConsistency, accuracy_dis
import cv2
import ref
from progress.bar import Bar



def step(args, split, epoch, loader, model, loss, logger=None, optimizer = None, M = None, f = None, nViews=ref.nViews):
  losses, mpjpe, mpjpe_r = AverageMeter(), AverageMeter(), AverageMeter()
  viewLosses, shapeLosses, supLosses = AverageMeter(), AverageMeter(), AverageMeter()
  
  prior_loss = []
  regr_loss = []
  accuracy_this = []

  if split == 'train':
    model.train()
  else:
    model.eval()

  idx_0 = len(loader)*epoch

  for i, (input, target, meta) in enumerate(loader):
    input_var = torch.autograd.Variable(input.cuda())
    target_var = torch.autograd.Variable(target.cuda())
    output = model(input_var)
    
    cr_loss = loss(output)

    cr_regr_loss = ((output - target_var.view(target_var.shape[0],-1)) ** 2).sum() / ref.J / 3 / input.shape[0]


    current_acc = accuracy(output.data, target, meta)
    accuracy_this.append(current_acc)

    prior_loss.append(cr_loss.data[0])
    regr_loss.append(cr_regr_loss.data[0])
    
    if split == 'train':
      optimizer.zero_grad()
      cr_loss.backward()
      optimizer.step()
      logger.add_scalar('train/prior-loss', cr_loss.data[0], idx_0+i)
      logger.add_scalar('train/regr-loss', cr_regr_loss.data[0], idx_0+i)
      
  return np.array(accuracy_this).mean(), np.array(regr_loss).mean(), np.array(prior_loss).mean()




def train(args, train_loader, model, loss, logger, optimizer, epoch, nViews=ref.nViews):
  return step(args, 'train', epoch, train_loader, model, loss, logger, optimizer)

def validate(args, supTag, val_loader, model, loss, epoch):
  return step(args, 'val' + supTag, epoch, val_loader, model,loss)

def test(args, loader, model, loss):
  return step(args, 'test', 0, loader, model, loss)
