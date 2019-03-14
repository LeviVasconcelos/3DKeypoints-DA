
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

import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D

EDGES = [(0,1),(0,2),(1,3),(2,3),(2,4),(2,6),(3,5),(3,7),(4,5),(4,8),(5,9)]

J = 10
edges = [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 5], 
         [4, 8], [5, 9], [3, 7], [2, 6]]
S = 224

def show3D(ax, points, c = (255, 0, 0)):
    points = points.reshape(J, 3)
    x, y, z = np.zeros((3, J))
    for j in range(J):
        x[j] = points[j, 0] 
        y[j] = points[j, 1] 
        z[j] = points[j, 2] 
    ax.scatter(x, y, z, c = c)
    l = 0
    for e in edges:
        ax.plot(x[e], y[e], z[e], c =c)
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
    cv2.imwrite('2d.png',img)
    plt.savefig('3d.png')
    plt.close()
    return '2d.png', '3d.png'
    



def train_step(args, split, epoch, loader, model, loss, update_bn=True, logger=None, optimizer = None, M = None, f = None, nViews=ref.nViews):
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

  for i, (input, target, _) in enumerate(loader):
    target_var = torch.autograd.Variable(target.cuda())
    dt = compute_distances(target_var)
    input_var = torch.autograd.Variable(input.cuda())
    output = model(input_var)
    cr_loss = loss(output, logger, i==0, dt=dt)

    prior_loss.append(cr_loss.data[0])


    if split == 'train':
      optimizer.zero_grad()
      cr_loss.backward() 
      optimizer.step()
      logger.add_scalar('train/prior-loss', cr_loss.data[0], idx_0+i)

  return np.array(prior_loss).mean()




def eval_step(args, split, epoch, loader, model, loss, update=True, optimizer = None, M = None, f = None, nViews=ref.nViews, plot_img = False, logger = None):
  losses, mpjpe, mpjpe_r = AverageMeter(), AverageMeter(), AverageMeter()
  viewLosses, shapeLosses, supLosses = AverageMeter(), AverageMeter(), AverageMeter()
  
  prior_loss = []
  regr_loss = []
  accuracy_this = []
  accuracy_shape = []

  model.eval()

  idx_0 = len(loader)*epoch

  for i, (input, target, meta) in enumerate(loader):
    input_var = torch.autograd.Variable(input.cuda(),volatile=True)
    target_var = torch.autograd.Variable(target.cuda(),volatile=True)
    output = model(input_var)

    cr_regr_loss = ((output - target_var.view(target_var.shape[0],-1)) ** 2).sum() / ref.J / 3 / input.shape[0]


    current_acc = accuracy(output.data, target, meta)
    current_acc_shape = accuracy_dis(output.data, target, meta)
    accuracy_this.append(current_acc)
    accuracy_shape.append(current_acc_shape)
    regr_loss.append(cr_regr_loss.data[0])
    dt = compute_distances(target_var)
    cr_loss = loss(output, logger, plot=i==0,dt=dt)
    if plot_img and i<10:
		img = (input.numpy()[0] * 255).transpose(1, 2, 0).astype(np.uint8)
		cv2.imwrite('01.png', img)
		gt = target.cpu().numpy()[0]
		pred = (output.data).cpu().numpy()[0]
		p2d, p3d = compute_images(cv2.imread('01.png'),pred,gt)
 		logger.add_image('Image 2D ' +str(i), (np.asarray(Image.open(p2d))).transpose(2,0,1), epoch)
 		logger.add_image('Image 3D ' +str(i), (np.asarray(Image.open(p3d))).transpose(2,0,1), epoch)


    prior_loss.append(cr_loss.data[0])
  return np.array(accuracy_this).mean(),np.array(accuracy_shape).mean(), np.array(regr_loss).mean(), np.array(prior_loss).mean()




def train(args, train_loader, model, loss, update_bn, logger, optimizer, epoch, nViews=ref.nViews):
  return train_step(args, 'train', epoch, train_loader[0], model, loss, update_bn, logger, optimizer)

def validate(args, supTag, val_loader, model, loss, epoch,plot_img=False, logger=None):
  return eval_step(args, 'val' + supTag, epoch, val_loader, model,loss,plot_img = plot_img, logger = logger)

def test(args, loader, model, loss,plot_img=False, logger=None):
  return eval_step(args, 'test', 0, loader, model, loss,plot_img = plot_img, logger = logger)
