import numpy as np
import os
from utils.visualization import chair_show3D, chair_show2D, human_show2D, human_show3D, human_from_3D 
from os import listdir
from os.path import isfile, join
from opts import opts
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
from utils.utils import createDirIfNonExistent
args = opts().parse()


def horn87(pointss, pointst):
    centers = pointss.mean(axis = 1)
    centert = pointst.mean(axis = 1)
    pointss = (pointss.transpose() - centers).transpose()
    pointst = (pointst.transpose() - centert).transpose()
    m = np.dot(pointss, pointst.transpose(1, 0))
    n = np.array([[m[0, 0] + m[1, 1] + m[2, 2], m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]], 
                  [m[1, 2] - m[2, 1], m[0, 0] - m[1, 1] - m[2, 2], m[0, 1] + m[1, 0], m[0, 2] + m[2, 0]], 
                  [m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], m[1, 1] - m[0, 0] - m[2, 2], m[1, 2] + m[2, 1]], 
                  [m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], m[2, 2] - m[0, 0] - m[1, 1]]])
    v, u = np.linalg.eig(n)
    id = v.argmax()
  
    q = u[:, id]
    r = np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])], 
                  [2*(q[2]*q[1]+q[0]*q[3]), q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2*(q[2]*q[3]-q[0]*q[1])], 
                  [2*(q[3]*q[1]-q[0]*q[2]), 2*(q[3]*q[2]+q[0]*q[1]), q[0]**2-q[1]**2-q[2]**2+q[3]**2]])
    t = centert - np.dot(r, centers)
  
    return r.astype(np.float32), t.astype(np.float32) 

def rotate(points, gt):
    R,t = horn87(points.transpose(), gt.transpose())
    #R = RotMat('Y', math.pi)
    ret = np.matmul(R,points.transpose()).transpose()
    return ret

def SavePose(img, prediction, gt_, uncentred, intrinsics, filename):
    np.savez(filename + 'npz', img=img, pred=prediction, gt=gt_, gt_uncentred=uncentred, intrinsics=intrinsics)

def DrawImage(img, prediction, gt_, uncentred, intrinsics, tag='', draw_gt=False):
    numpy_img = img.copy()
    pred = prediction.copy()
    gt = gt_.copy()
    pred = rotate(pred, gt)
    gt_uncentred = uncentred.copy()
    if draw_gt:
        numpy_img = human_from_3D(numpy_img, gt_uncentred, intrinsics,
                                    (0,0,180), 224./1000.)
    else:
        numpy_img = human_from_3D(numpy_img, pred + gt_uncentred[0], intrinsics, 
                                    (180,0,0), 224./1000., flip=False)
    return numpy_img

def draw_skeleton(img, skeleton, tag, draw_gt=False):
    return DrawImage(img, skeleton['pred'], skeleton['gt'], skeleton['gt_uncentred'], skeleton['intrinsics'], tag=tag, draw_gt=draw_gt)

def load_folder(path):
    files = [f for f in listdir(path) ]
    image_files = [os.path.join(path,f) for f in  files if f.split('.')[-1] == 'png']
    skeleton_files = [os.path.join(path,f) for f in files if f.split('.')[-1] == 'npz']
    image_files.sort()
    skeleton_files.sort()
    print('loaded %d from %s [%d]' %(len(skeleton_files), path, len(files)))
    return image_files, skeleton_files

def _load_skeleton(path):
    data = np.load(path)
    return data['img'], data['pred'], data['gt'], data['gt_uncentred'], data['intrinsics']

def Skeleton():
    return { 'img':[], 'pred':[], 'gt':[], 'gt_uncentred':[], 'intrinsics':[]}

def load_data(path):
    ret = Skeleton()
    ret['img'], ret['pred'], ret['gt'], ret['gt_uncentred'], ret['intrinsics'] = _load_skeleton(path)
    return ret

def make_pretty(images, titles):
    nImages = len(images)
    assert(nImages == 4)
    fig = plt.figure(figsize=(14,3))
    rows = 1
    cols = 4
    for i in range(nImages):
        img = fig.add_subplot(rows, cols, i+1)
        img.set_title(titles[i])
        img.axis('off')
        img.set_xlim(224)
        img.set_ylim(224)
        img.margins(0)
        plt.imshow(images[i])
    plt.savefig('image.png')
    return cv2.imread('image.png')


def save_drawing(img, i):
    save_root = os.path.join(args.root_folder, 'saved_images')
    createDirIfNonExistent(save_root) 
    filename =  os.path.join(save_root, 'img_%d.png' % i)
    cv2.imwrite(filename, img)
    with open(os.path.join(save_root, 'indexes.txt'), 'a+') as f:
        f.write(str(i) + '\n')
        

def visualize(root_folder):
    _, final_skeletons = load_folder(os.path.join(root_folder, 'final'))
    _, source_skeletons = load_folder(os.path.join(root_folder, 'source'))
    _, middle_skeletons = load_folder(os.path.join(root_folder, 'middle'))
    _, huang_skeletons = load_folder(os.path.join(root_folder, 'huang'))
    i = 0
    while i < len(final_skeletons):
       source_data = load_data(source_skeletons[i])
       middle_data = load_data(middle_skeletons[i])
       final_data = load_data(final_skeletons[i])
       huang_data = load_data(huang_skeletons[i])
       
       img = source_data['img']
       make_pretty 
       source_img = draw_skeleton(img, source_data, 'source')
       final_img = draw_skeleton(img, final_data, 'final')
       middle_img = draw_skeleton(img, middle_data, 'middle')
       huang_img = draw_skeleton(img, huang_data, 'huang')
       ground_truth = draw_skeleton(img, final_data, 'gt', draw_gt=True) 
       
       titles = ['Baseline', 'Huang', 'Ours', 'GT']
       images = [source_img, huang_img, final_img, ground_truth]
       img = make_pretty(images, titles)
       cv2.imshow('picker %d' % i,img)
       key = cv2.waitKey(0)
       if key == 83:
          i += 1 
       elif key == 81:
           i = i-1 if (i-1) >= 0 else 0
       elif key == 32:
           save_drawing(img, i)
       elif key == 27:
           break

if __name__ == '__main__':
    visualize(args.root_folder)
