#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:59:11 2019

@author: levi
"""
import argparse
import copy
import os
from random import shuffle

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
import cv2

ShapeNet_dir = '../../../3DKeypoints-DA/data/ShapeNet/'
ModelNet_dir = '../../../3DKeypoints-DA/data/ModelNet/'
DCNN_dir = '../data/3DCNN/'
Redwood_dir = '../../../3DKeypoints-DA/data/Redwood_depth/'
RedwoodRGB_dir = '../../../3DKeypoints-DA/data/Redwood_RGB/'
Humans_dir = '/hardmnt/rebel1/data/data/processed/'

DatasetPath = {
            'Redwood_depth': '../../../3DKeypoints-DA/data/Redwood_depth/',
            'ModelNet': '../../../3DKeypoints-DA/data/ModelNet/',
            'RedwoodRGB': '../../../3DKeypoints-DA/data/Redwood_RGB/',
            'Humans_depth': '/hardmnt/rebel1/data/data/processed/',
            'ShapeNet' : '../../../3DKeypoints-DA/data/ShapeNet/'
            }

DatasetFilter = {
            'Redwood_depth': lambda fname, obj_id: fname.split('_')[0] == str(obj_id),
            'ModelNet' : lambda fname, obj_id: fname.split('_')[1] == '%.4d' % (obj_id),
            'ShapeNet': lambda fname, obj_id: fname.split('_')[0] == str(obj_id),
                  }
DatasetGetView = {
            'Redwood_depth': lambda fname : int(fname.split('_')[1].split('.')[0]),
            'ModelNet' : lambda fname: int(fname.split('_')[2].split('.')[0]),
            'ShapeNet' : lambda fname: int(fname.split('_')[1].split('.')[0]),
            }

DatasetSubSubPath = {
            'Redwood_depth' : '',
            'ModelNet': 'train',
            'ShapeNet': ''
            }
DatasetImageSubPath = {
            'Redwood_depth' : 'images',
            'ModelNet': 'images',
            'ShapeNet': 'images_annot'
            }
DatasetAnnotSubPath = {
            'Redwood_depth' : 'annots',
            'ModelNet': 'annots',
            'ShapeNet': 'annots'
            }
def createDirIfNonExistent(path):
    if not os.path.isdir(path):
        os.mkdir(path)
class opts():
      
      def __init__(self):
            self.parser = argparse.ArgumentParser(description='3D Keypoint')
  
      def init(self):
            self.parser.add_argument('-dataset', default='Redwood_depth', type=str, 
                                     help='Redwood | ShapeNet | RedwoodRGB | 3DCNN')
            self.parser.add_argument('-objID', default=0, 
                                     type=int, help='Object Id')
            self.parser.add_argument('-out_dir', default='../vis/')
      def parse(self):
            self.init()
            self.args = self.parser.parse_args()
            return self.args


J = 10
edges = [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 5], 
         [4, 8], [5, 9], [3, 7], [2, 6]]
S = 224

def show3D(ax, points, c = (255, 0, 0), edges = edges):
    points = points.reshape(J, 3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.set_xlabel('')
    #ax.set_ylabel('x') 
    #ax.set_zlabel('y')
    oo = 0.5
    xmax, ymax, zmax, xmin, ymin, zmin = oo, oo, oo, -oo, -oo, -oo
    #print c, points
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
    max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([zb], [xb], [yb], 'w')
        
def show2D(img, points, c):
  points = ((points.reshape(J, 3) + 0.5) * S).astype(np.int32)
  points[:, 0], points[:, 1] = points[:, 1].copy(), points[:, 0].copy()  
  for j in range(J):
    cv2.circle(img, (points[j, 0], points[j, 1]), 3, c, -1)
  #print points
  for e in edges:
    cv2.line(img, (points[e[0], 0], points[e[0], 1]),
                  (points[e[1], 0], points[e[1], 1]), c, 2)
  return img

def MeanEuclidianDistance(c1, c2):
      diff = c1 - c2
      

def showPair(imgs, gts, img_fnames, annot_fnames, out_dir):
      gt_1, gt_2 = gts
      diff = np.abs(gt_1 - gt_2)
      fig = plt.figure()
      img_1, img_2 = imgs
      img1 = fig.add_subplot((231))
      img1.set_title('red')
      plt.imshow(show2D(img_1, gt_1, (255,0,0)))

      img2 = fig.add_subplot((234))
      img2.set_title('blue')
      plt.imshow(show2D(img_2, gt_2, (0,0,255)))

      ax1 = fig.add_subplot((232), projection='3d')
      show3D(ax1, gt_1, 'r')
      
      ax2 = fig.add_subplot((235), projection='3d')
      show3D(ax2, gt_2, 'b')
      
      ax = fig.add_subplot((233), projection='3d')
      show3D(ax, gt_1, 'r')
      show3D(ax, gt_2, 'b')
      #ax.set_ylabel('x') 
      #ax.set_zlabel('y')
      #oo = 0.5
      #xmax, ymax, zmax, xmin, ymin, zmin = oo, oo, oo, -oo, -oo, -oo
      #show3D(ax, gt_1, 'r')
      #show3D(ax, gt_2, 'b')
      #max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
      #Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
      #Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
      #Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
      #for xb, yb, zb in zip(Xb, Yb, Zb):
      #      ax.plot([zb], [xb], [yb], 'w')



      #cv2.imwrite(os.path.join(out_dir, img_fname.split('/')[-1]),img)
      annot_1, annot_2 = annot_fnames
      img_f1, img_f2 = img_fnames
      img_title = 'imgs_' + img_f1.split('/')[-1].split('.')[0].split('_')[-1] + '-' + img_f2.split('/')[-1].split('.')[0].split('_')[-1]
      annots_title = 'annots_' + annot_1.split('/')[-1].split('.')[0].split('_')[-1] + '-' + annot_2.split('/')[-1].split('.')[0].split('_')[-1]
      print annot_fnames
      plt.savefig(os.path.join(out_dir, 'annot' + img_title + '--' + annots_title + '.jpg'))
      np.savetxt(os.path.join(out_dir, 'diff_' + annots_title + '.txt'), diff)
      #cv2.imshow(img_fname.split('/')[-1], img)
      #plt.show()
      #cv2.waitKey()
      
      
      

def main():
      args = opts().parse()
      filename_filter = DatasetFilter[args.dataset]
      dataset_path = DatasetPath[args.dataset]
      images_path = os.path.join(dataset_path, DatasetImageSubPath[args.dataset], DatasetSubSubPath[args.dataset])
      annots_path = os.path.join(dataset_path, DatasetAnnotSubPath[args.dataset], DatasetSubSubPath[args.dataset])
      images = [os.path.join(images_path,f) for f in os.listdir(images_path) if filename_filter(f, args.objID)]
      annots = [os.path.join(annots_path,f) for f in os.listdir(annots_path) if filename_filter(f, args.objID)]
      images.sort()
      images_ = copy.copy(images)
      shuffle(images_)
      annots.sort()
      createDirIfNonExistent(args.out_dir)
      out_dir = os.path.join(args.out_dir, args.dataset + '_' + str(args.objID))
      createDirIfNonExistent(out_dir)
      print images_path
      print annots_path
      print len(images)
      print len(annots)
      print images[0]
      print annots[0]
      images.sort()
      annots.sort()
      find_ID = lambda name : DatasetGetView[args.dataset](name.split('/')[-1])
      for img1_fname, img2_fname, annot in zip(images, images_, annots):
            gt_1 = np.loadtxt(annot)[:10]
            img_1 = cv2.imread(img1_fname)
            img_2 = cv2.imread(img2_fname)
            gt_2_fname = [f for f in annots if find_ID(img2_fname) == find_ID(f)]
            #print(len(gt_2_fname))
            assert(len(gt_2_fname) == 1)
            gt_2 = np.loadtxt(gt_2_fname[0])[:10]
            #print gt.shape
            showPair((img_1, img_2), (gt_1, gt_2), (img1_fname, img2_fname), (annot, gt_2_fname[0]), out_dir) 

if __name__ == '__main__':
  main()