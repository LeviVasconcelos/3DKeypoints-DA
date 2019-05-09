#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:52:07 2019

@author: levi
"""

import argparse
import copy
import os
from random import shuffle
import torch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mpl_toolkits.mplot3d
from datasets.humans36m import Humans36mRGBSourceDataset, Humans36mRGBTargetDataset 
from mpl_toolkits.mplot3d import Axes3D
import cv2

def horn87(pointsS, pointsT):
  centerS = pointsS.mean(axis = 1)
  centerT = pointsT.mean(axis = 1)
  pointsS = (pointsS.transpose() - centerS).transpose()
  pointsT = (pointsT.transpose() - centerT).transpose()
  M = np.dot(pointsS, pointsT.transpose(1, 0))
  N = np.array([[M[0, 0] + M[1, 1] + M[2, 2], M[1, 2] - M[2, 1], M[2, 0] - M[0, 2], M[0, 1] - M[1, 0]], 
                [M[1, 2] - M[2, 1], M[0, 0] - M[1, 1] - M[2, 2], M[0, 1] + M[1, 0], M[0, 2] + M[2, 0]], 
                [M[2, 0] - M[0, 2], M[0, 1] + M[1, 0], M[1, 1] - M[0, 0] - M[2, 2], M[1, 2] + M[2, 1]], 
                [M[0, 1] - M[1, 0], M[2, 0] + M[0, 2], M[1, 2] + M[2, 1], M[2, 2] - M[0, 0] - M[1, 1]]])
  v, u = np.linalg.eig(N)
  id = v.argmax()

  q = u[:, id]
  R = np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])], 
                [2*(q[2]*q[1]+q[0]*q[3]), q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2*(q[2]*q[3]-q[0]*q[1])], 
                [2*(q[3]*q[1]-q[0]*q[2]), 2*(q[3]*q[2]+q[0]*q[1]), q[0]**2-q[1]**2-q[2]**2+q[3]**2]])
  t = centerT - np.dot(R, centerS)

  return R.astype(np.float32), t.astype(np.float32) 

def createDirIfNonExistent(path):
    if not os.path.isdir(path):
        os.mkdir(path)
class opts():
      
      def __init__(self):
            self.parser = argparse.ArgumentParser(description='3D Keypoint')
  
      def init(self):
            self.parser.add_argument('-dataset', default='HumansRGB', type=str, 
                                     help='HumansDepth | HumansRGB')
            self.parser.add_argument('-subject', default=1, 
                                     type=int, help='Subject Id')
            self.parser.add_argument('-nImages', default=10, type=int, help='Number of samples to draw')
            self.parser.add_argument('-out_dir', default='../vis/')
      def parse(self):
            self.init()
            self.args = self.parser.parse_args()
            return self.args
      
J = 32
edges = [[13,25],  #right arm connection
         [13,17], #left arm connection
         [0,12], #spine1
         [12,13], #spine2
         [13, 14], [14, 15], #head
         [0,1], [1,2], [2,3], [3,4], [4,5], #right leg
         [0,6], [6,7], [7,8], [8,9], [9,10], #left leg
         [17,18],[18,19],[19,20],[20,21],[21,22],[22,23], #left arm
         [25,26],[26,27],[27,28],[28,29],[29,30],[30,31] #right arm
         ]

edges_2 = [[0, 1], [1, 2], [2, 3], [3, 6], [6, 7], [7, 8], [8, 12], 
           [12, 13], [13, 14], [14, 15], [15, 17], [17, 18], [18, 19], 
           [19, 25], [25, 26], [26, 27]]
S = 224

def show3D(ax, points, c = (255, 0, 0), edges = edges):
    points = points.reshape(J, 3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    oo = 0.5
    xmax, ymax, zmax, xmin, ymin, zmin = oo, oo, oo, -oo, -oo, -oo
    x, y, z = np.zeros((3, J))
    for j in range(J):
        x[j] = points[j, 0] 
        y[j] = points[j, 2] 
        z[j] = -points[j, 1] 
    ax.scatter(x, y, z, c = c)
    l = 0
    for e in edges:
        ax.plot(x[e], y[e], z[e], c =c)
        l += ((z[e[0]] - z[e[1]]) ** 2 + (x[e[0]] - x[e[1]]) ** 2 + (y[e[0]] - y[e[1]]) ** 2) ** 0.5
        
def _project_mono_2d(pose3d, proj, scale = 1):
      x3d = np.stack([pose3d[:, 0], pose3d[:, 2]], axis=-1)
      x_coord_2d = np.dot(x3d, proj[:2]) / pose3d[:,2] * scale
      y3d = np.stack([pose3d[:, 1], pose3d[:, 2]], axis=-1)
      y_coord_2d = np.dot(y3d, proj[2:]) / pose3d[:,2] * scale
      return np.stack([x_coord_2d, y_coord_2d], axis=-1)

def project_univ_2d(points_3d, similarity, proj):
      R,t = similarity
      mono_space = np.dot(R, points_3d.transpose()).transpose() + t
      return _project_mono_2d(mono_space, proj, 224. / 1000.)

def show2D(img, points, c):
  points = points.astype(np.int32)
  points[:, 0], points[:, 1] = points[:, 1].copy(), points[:, 0].copy()  
  for j in range(J):
    x = points[j, 0] if points[j, 0] > 0 else -points[j, 0]
    y = points[j, 1] if points[j, 1] > 0 else -points[j, 1]
    cv2.circle(img, (y, x), 3, c, -1)
  return img

def showPair(imgs, gts, img_fnames, annot_fnames, univ_pose, out_dir, to_camera, projs):
      gt_1, gt_2 = gts
      diff = np.abs(gt_1 - gt_2)
      fig = plt.figure()
      img_1, img_2 = imgs
      img1 = fig.add_subplot((231))
      img1.set_title('red')
      projected_gt1 = project_univ_2d(univ_pose, to_camera[0], projs[0])
      plt.imshow(show2D(img_1, projected_gt1, (255,0,0)))

      img2 = fig.add_subplot((234))
      img2.set_title('blue')
      projected_gt2= project_univ_2d(univ_pose, to_camera[1], projs[1])
      plt.imshow(show2D(img_2, projected_gt2, (0,0,255)))

      ax1 = fig.add_subplot((232), projection='3d')
      show3D(ax1, gt_1, 'r')
      
      ax2 = fig.add_subplot((235), projection='3d')
      show3D(ax2, gt_2, 'b')
      
      ax = fig.add_subplot((233), projection='3d')
      show3D(ax, gt_1, 'r')
      show3D(ax, gt_2, 'b')
      show3D(ax, univ_pose, 'g')
      
      ax3 = fig.add_subplot((236), projection='3d')
      show3D(ax3, univ_pose, 'g')
      ax.set_ylabel('x') 
      ax.set_zlabel('y')
      oo = 0.5
      xmax, ymax, zmax, xmin, ymin, zmin = oo, oo, oo, -oo, -oo, -oo
      show3D(ax, gt_1, 'r')
      show3D(ax, gt_2, 'b')
      max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
      Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
      Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
      Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
      for xb, yb, zb in zip(Xb, Yb, Zb):
            ax1.plot([zb], [xb], [yb], 'w')
            ax2.plot([zb], [xb], [yb], 'w')
            ax3.plot([zb], [xb], [yb], 'w')
            ax.plot([zb], [xb], [yb], 'w')
            
      img_f1, img_f2 = img_fnames
      plt.savefig(os.path.join(out_dir, 'plot' + img_f1 + '--' + img_f2 + '.jpg'))



def main():
      args = opts().parse()
      rgb = (args.dataset == 'HumansRGB')
      h36m_source = Humans36mRGBSourceDataset('train', 4) 
      h36m_target = Humans36mRGBTargetDataset('train', 4)
      source_mean, source_std = h36m_source._get_normalization_statistics()
      target_mean, target_std = h36m_target._get_normalization_statistics()
      fig, axis = plt.subplot()
      #show_3D(axis, std, 'b')
      show3D(axis, source_mean, 'r')
      show3D(axis, target_mean, 'r')
      plt.show()
      
      '''out_dir = os.path.join(args.out_dir, args.dataset + '_S' + str(args.subject))
      imgs, annots, meta, mono_pose3d, univ_pose3d, original_pose3d, intrinsics, _ = h36m.__getitem__(2)
      to_camera = []
      for camera_pose in mono_pose3d:
            R,t = horn87(original_pose3d[0].transpose(), camera_pose.transpose())
            to_camera += [(R,t)]
      
      for i in range(args.nImages):
            imgs, annots, meta, mono_pose3d, univ_pose3d, original_pose3d, intrinsics, poses2d = h36m.__getitem__(i)
            print 'before', imgs.shape
            #transpose images and cast as numpy
            imgs = imgs.numpy()
            #imgs.transpose(0, 3, 1, 2)
            imgs = imgs.transpose(0, 2, 3, 1)
            name_1 = "%d-0" % i
            name_2 = "%d-1" % i
            showPair((imgs[0], imgs[1]), (mono_pose3d[0], mono_pose3d[1]), (name_1, name_2), None, original_pose3d[0], out_dir, to_camera, intrinsics)
            print 'after', imgs.shape'''
            
            
      
