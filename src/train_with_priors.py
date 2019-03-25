
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

from layers.PriorConsistencyCriterion import compute_rotation_loss

import copy

import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D





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
  losses, mpjpe, mpjpe_r = AverageMeter(), AverageMeter(), AverageMeter()
  viewLosses, shapeLosses, supLosses = AverageMeter(), AverageMeter(), AverageMeter()
  
  prior_loss = []

  if update_bn>0:
	print('Epoch: ' + str(epoch+1)+ ': unsupervised training with BN')
  	model.train()
  else:
	print('Epoch: ' + str(epoch+1)+ ': unsupervised training without BN')
	model.eval()

  idx_0 = len(loader)*epoch

  
  for i, (input, _, _) in enumerate(loader):
    input_var = input.to(device)
    output = model(input_var)

    cr_loss = loss(output)
    cr_loss = (cr_loss).mean()

    #rotation_loss = compute_rotation_loss(old_output.view(input.shape[0],10,3), output.view(input.shape[0],10,3), w)

    prior_loss.append(cr_loss.item())

    optimizer.zero_grad()
    total_loss = cr_loss
    total_loss.backward() 
    optimizer.step()
    logger.add_scalar('train/prior-loss', cr_loss.item(), idx_0+i)

  return np.array(prior_loss).mean()


def eval_step(args, split, epoch, loader, model, loss, update=True, optimizer = None, M = None, f = None, nViews=ref.nViews, plot_img = False, logger = None,device='cuda'):
  losses, mpjpe, mpjpe_r = AverageMeter(), AverageMeter(), AverageMeter()
  viewLosses, shapeLosses, supLosses = AverageMeter(), AverageMeter(), AverageMeter()
  
  prior_loss = []
  regr_loss = []
  accuracy_this = []
  accuracy_shape = []

  model.eval()

  idx_0 = len(loader)*epoch

  for i, (input, target, meta) in enumerate(loader):
    input_var = input.to(device)
    target_var = target.to(device)
    output = model(input_var)

    cr_regr_loss = ((output - target_var.view(target_var.shape[0],-1)) ** 2).sum() / ref.J / 3 / input.shape[0]

    current_acc = accuracy(output.data, target, meta)
    current_acc_shape = accuracy_dis(output.data, target, meta)
    accuracy_this.append(current_acc.item())
    accuracy_shape.append(current_acc_shape.item())
    
    regr_loss.append(cr_regr_loss.item())
    cr_loss = loss(output).mean()
    if plot_img and i<10:
			img = (input.numpy()[0] * 255).transpose(1, 2, 0).astype(np.uint8)
			cv2.imwrite('./tmp/'+str(i)+'01.png', img)
			gt = target.cpu().numpy()[0]
			pred = (output.data).cpu().numpy()[0]
			p3d = compute_images3D(cv2.imread('./tmp/'+str(i)+'01.png'),pred,gt, str(i))
	 		logger.add_image('Image 3D ' +str(i), (np.asarray(Image.open(p3d))).transpose(2,0,1), epoch)

    prior_loss.append(cr_loss.item())

  return np.array(accuracy_this).mean(),np.array(accuracy_shape).mean(), np.array(regr_loss).mean(), np.array(prior_loss).mean()




def train(args, train_loader, model, loss, update_bn, logger, optimizer, epoch, nViews=ref.nViews, threshold = 0.9):
  return train_step(args, 'train', epoch, train_loader[0], model, loss, update_bn, logger, optimizer, threshold=threshold)

def validate(args, supTag, val_loader, model, loss, epoch,plot_img=False, logger=None):
  return eval_step(args, 'val' + supTag, epoch, val_loader, model,loss,plot_img = plot_img, logger = logger)

def test(args, loader, model, loss,plot_img=False, logger=None):
  return eval_step(args, 'test', 0, loader, model, loss,plot_img = plot_img, logger = logger)
