
import torch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from utils.utils import AverageMeter, show3D
from utils.eval import accuracy, shapeConsistency, accuracy_dis
import cv2
import ref
from progress.bar import Bar
from layers.prior_generator import compute_distances
from PIL import Image
import sys
import time
import torch.nn as nn
import itertools
from utils.visualization import chair_show3D, chair_show2D, human_show2D, human_show3D 
from layers.PriorConsistencyCriterion import compute_rotation_loss
import os
import copy

import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D




J = ref.J
edges = ref.edges 
S = 224

def show3D(ax, points, c = (255, 0, 0)):
    points = points.reshape(J, 3)
    x, y, z = np.zeros((3, J))
    for j in range(J):
        x[j] = points[j, 0] 
        y[j] = points[j, 1] 
        z[j] = points[j, 2] 
    ax.scatter(z, x, y, c = c)
    l = 0
    for e in edges:
        ax.plot(z[e], x[e], y[e], c =c)
        l += ((z[e[0]] - z[e[1]]) ** 2 + (x[e[0]] - x[e[1]]) ** 2 + (y[e[0]] - y[e[1]]) ** 2) ** 0.5

def show2D(img, points, c):
  points = ((points.reshape(J, 3) + 0.5) * S).astype(np.int32)
  points[:, 0], points[:, 1] = points[:, 1].copy(), points[:, 0].copy()  
  for j in range(J):
    cv2.circle(img, (points[j, 0], points[j, 1]), 3, c, -1)
  for e in edges:
    cv2.line(img, (points[e[0], 0], points[e[0], 1]),
                  (points[e[1], 0], points[e[1], 1]), c, 2)
  return img




def compute_images(img, pred, gt):

    gt = gt.reshape(J, 3)
    pred = pred.reshape(J, 3)

    fig = plt.figure()
    ax = fig.add_subplot((111),projection='3d')
    ax.set_xlabel('z') 
    ax.set_ylabel('x') 
    ax.set_zlabel('y')
    oo = 0.5
    xmax, ymax, zmax, xmin, ymin, zmin = oo, oo, oo, -oo, -oo, -oo
    show3D(ax, gt, 'r')
    img = show2D(img, gt, (0, 0, 255))
    show3D(ax, pred, 'b')
    img = show2D(img, pred, (255, 0, 0))
    max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
    for xb, yb, zb in zip(Xb, Yb, Zb):
      ax.plot([zb], [xb], [yb], 'w')
    #cv2.imshow('input', img)
    cv2.imwrite('./tmp/'+'2d.png',img)
    plt.savefig('./tmp/'+'3d.png')
    plt.close()
    return './tmp/'+'2d.png', './tmp/'+'3d.png'



def compute_images3D(img, pred, gt, index):

    gt = gt.reshape(J, 3)
    pred = pred.reshape(J, 3)

    fig = plt.figure()
    ax = fig.add_subplot((111),projection='3d')
    ax.set_xlabel('z') 
    ax.set_ylabel('x') 
    ax.set_zlabel('y')
    oo = 0.5
    xmax, ymax, zmax, xmin, ymin, zmin = oo, oo, oo, -oo, -oo, -oo
    show3D(ax, gt, 'r')
    show3D(ax, pred, 'b')
    max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
    for xb, yb, zb in zip(Xb, Yb, Zb):
      ax.plot([xb], [yb], [zb],'w')
    #cv2.imshow('input', img)
    plt.savefig('./tmp/'+index+'3d.png')
    plt.close()
    return './tmp/'+index+'3d.png'
    



def train_step(args, split, epoch, loader, model, loss, update_bn=True, logger=None, optimizer = None, M = None, f = None, nViews=ref.nViews,device='cuda', threshold = 0.9):
  prior_loss = []

  if update_bn>0:
    print('Epoch: ' + str(epoch+1)+ ': unsupervised training with BN')
    model.train()
  else:
    print('Epoch: ' + str(epoch+1)+ ': unsupervised training without BN')
    model.eval()

  idx_0 = len(loader)*epoch

  loss.epoch = epoch
  for i, (input, target, _) in enumerate(loader):
    input_var = input.to(device)
    target_var = target.to(device)
    output = model(input_var)
    cr_loss = loss(output, dt=target_var)
    if torch.isnan(output).sum() > 0:
        print('OUTPUT WITH NANS DURING TRAINING %d' % i)
        return
    cr_loss = (cr_loss).mean()
    #print(cr_loss.item())

    #rotation_loss = compute_rotation_loss(old_output.view(input.shape[0],10,3), output.view(input.shape[0],10,3), w)

    prior_loss.append(cr_loss.item())

    optimizer.zero_grad()
    total_loss = cr_loss
    total_loss.backward() 
    optimizer.step()
    logger.add_scalar('train/prior-loss', cr_loss.item(), idx_0+i)

  return np.array(prior_loss).mean()


def eval_step(args, split, epoch, loader, model, loss, update=True, optimizer = None, M = None, f = None, nViews=ref.nViews, plot_img = False, logger = None,device='cuda', unnorm_net=(lambda pose:pose), unnorm_tgt=(lambda pose:pose)):
  prior_loss = []
  regr_loss = []
  accuracy_this = []
  accuracy_shape = []

  model.eval()
  if update:
    print('Updating BN layers')
    model.train()
  draw_2d = chair_show2D if ref.category == 'Chair' else human_show2D
  draw_3d = chair_show3D if ref.category == 'Chair' else human_show3D

  for i, (input, target, meta) in enumerate(loader):
    input_var = input.to(device).detach()
    target_var = target.to(device)
    output = model(input_var).detach()
    unnormed_prediction = unnorm_net(output.view(input.shape[0], ref.J, 3))
    unnormed_gt = unnorm_tgt(target_var)
    if torch.isnan(output).sum() > 0:
        print('OUTPUT WITH NANS')
    cr_regr_loss = ((output - target_var.view(target_var.shape[0],-1)) ** 2).sum() / ref.J / 3 / input.shape[0]
    current_acc = accuracy(unnormed_prediction.view(target_var.shape[0], -1).data, unnormed_gt.data, meta)
    current_acc_shape = accuracy_dis(unnormed_prediction.view(target_var.shape[0], -1).data, unnormed_gt.data, meta)
    accuracy_this.append(current_acc.item())
    accuracy_shape.append(current_acc_shape.item())
    
    regr_loss.append(cr_regr_loss.item())
    cr_loss = loss(output, dt=target_var).mean()
    numpy_img = None
    #if plot_img:
          #numpy_img = (input.numpy()[0] * 255).transpose(1, 2, 0).astype(np.uint8)
    if plot_img and i<10:
          pred = unnormed_prediction.data.cpu().numpy()[0].copy()
          gt = unnormed_gt.data.cpu().numpy()[0].copy()
          #numpy_img = chair_show2D(numpy_img, pred, (255,0,0))
          #numpy_img = chair_show2D(numpy_img, gt, (0,0,255))
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
          #logger.add_image('Image 2D ' + str(i), (np.asarray(Image.open(filename_2d))).transpose(2,0,1), epoch)
          plt.close()

    prior_loss.append(cr_loss.item())

  return np.array(accuracy_this).mean(),np.array(accuracy_shape).mean(), np.array(regr_loss).mean(), np.array(prior_loss).mean()

def train_priors(args, train_loader, model, loss, update_bn, logger, optimizer, epoch, nViews=ref.nViews, threshold = 0.9):
  return train_step(args, 'train', epoch, train_loader[0], model, loss, update_bn, logger, optimizer, threshold=threshold)

def validate_priors(args, supTag, val_loader, model, loss, epoch, update = False, plot_img=False, logger=None, unnorm_net=(lambda pose:pose), unnorm_tgt=(lambda pose:pose)):
  return eval_step(args, 'val' + supTag, epoch, val_loader, model,loss,plot_img = plot_img, update=update, logger = logger, unnorm_net=unnorm_net, unnorm_tgt=unnorm_tgt)

def test(args, loader, model, loss,plot_img=False, logger=None):
  return eval_step(args, 'test', 0, loader, model, loss,plot_img = plot_img, logger = logger)
