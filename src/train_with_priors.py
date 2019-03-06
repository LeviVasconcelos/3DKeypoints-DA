
import torch
import numpy as np
from utils.utils import AverageMeter, show3D
from utils.eval import accuracy, shapeConsistency, accuracy_dis
import cv2
import ref
from progress.bar import Bar



def mixed_step(args, split, epoch, loaders, model, loss, logger=None, optimizer = None, M = None, f = None, nViews=ref.nViews):
  losses, mpjpe, mpjpe_r = AverageMeter(), AverageMeter(), AverageMeter()
  viewLosses, shapeLosses, supLosses = AverageMeter(), AverageMeter(), AverageMeter()
  
  src_data_loader = loaders[1]
  loader=loaders[0]

  prior_loss = []
  regr_loss = []
  accuracy_this = []

  if split == 'train':
    model.train()
  else:
    model.eval()
  
  print('I am training where target exist')
  exit(1)
  idx_0 = len(loader)*epoch
  data_zip = enumerate(zip(loader,src_data_loader))
  for i, ((input, target, meta), (input_source, target_source, _)) in data_zip:
    input_var = torch.autograd.Variable(input.cuda())
    target_var = torch.autograd.Variable(target.cuda())
    input_source_var = torch.autograd.Variable(input_source.cuda())
    target_source_var = torch.autograd.Variable(target_source.cuda())
    output = model(input_var)
    
    cr_loss = loss(output, logger, idx_0+i, plot=i==0)

    cr_regr_loss = ((output - target_var.view(target_var.shape[0],-1)) ** 2).sum() / ref.J / 3 / input.shape[0]

    output2 = model(input_source_var)

    cr_regr_source_loss = ((output2 - target_source_var.view(target_source_var.shape[0],-1)) ** 2).sum() / ref.J / 3 / input_source.shape[0]

    floss = cr_regr_source_loss+0.1*cr_loss
    
    if split == 'train':
      optimizer.zero_grad()
      floss.backward()
      optimizer.step()
      logger.add_scalar('train/prior-loss', cr_loss.data[0], idx_0+i)
      logger.add_scalar('train/regr-loss', cr_regr_loss.data[0], idx_0+i)
      logger.add_scalar('train/source-regr-loss', cr_regr_source_loss.data[0], idx_0+i)
      
  return 0



def train_step(args, split, epoch, loader, model, loss, update=True, logger=None, optimizer = None, M = None, f = None, nViews=ref.nViews):
  losses, mpjpe, mpjpe_r = AverageMeter(), AverageMeter(), AverageMeter()
  viewLosses, shapeLosses, supLosses = AverageMeter(), AverageMeter(), AverageMeter()
  
  prior_loss = []

  model.train()

  idx_0 = len(loader)*epoch
  print('I am training where no target exist')
  for i, (input, _, _) in enumerate(loader):
    
    input_var = torch.autograd.Variable(input.cuda())
    output = model(input_var)
    
    cr_loss = loss(output, logger, idx_0+i, plot=i==0)

    prior_loss.append(cr_loss.data[0])

    if split == 'train':
      optimizer.zero_grad()
      cr_loss.backward() 
      optimizer.step()
      logger.add_scalar('train/prior-loss', cr_loss.data[0], idx_0+i)

  return np.array(prior_loss).mean()




def eval_step(args, split, epoch, loader, model, loss, update=True, logger=None, optimizer = None, M = None, f = None, nViews=ref.nViews):
  losses, mpjpe, mpjpe_r = AverageMeter(), AverageMeter(), AverageMeter()
  viewLosses, shapeLosses, supLosses = AverageMeter(), AverageMeter(), AverageMeter()
  
  prior_loss = []
  regr_loss = []
  accuracy_this = []

  model.eval()

  idx_0 = len(loader)*epoch

  for i, (input, target, meta) in enumerate(loader):
    input_var = torch.autograd.Variable(input.cuda(),volatile=True)
    target_var = torch.autograd.Variable(target.cuda(),volatile=True)
    output = model(input_var)
    
    cr_loss = loss(output, logger, idx_0+i, plot=i==0)

    cr_regr_loss = ((output - target_var.view(target_var.shape[0],-1)) ** 2).sum() / ref.J / 3 / input.shape[0]


    current_acc = accuracy(output.data, target, meta)
    accuracy_this.append(current_acc)

    prior_loss.append(cr_loss.data[0])
    regr_loss.append(cr_regr_loss.data[0])

  return np.array(accuracy_this).mean(), np.array(regr_loss).mean(), np.array(prior_loss).mean()




def train(args, train_loader, model, loss, update, logger, optimizer, epoch, nViews=ref.nViews):
  return train_step(args, 'train', epoch, train_loader[0], model, loss, update, logger, optimizer)

def validate(args, supTag, val_loader, model, loss, epoch):
  return eval_step(args, 'val' + supTag, epoch, val_loader, model,loss)

def test(args, loader, model, loss):
  return eval_step(args, 'test', 0, loader, model, loss)
