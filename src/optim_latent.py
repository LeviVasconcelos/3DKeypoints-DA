import os

import cv2

import ref
import numpy as np
from utils.horn87 import horn87, RotMat, Dis
from progress.bar import Bar
from utils.debugger import Debugger
import torch

oo = 1e18

DEBUG = False
SAVE_LATENT = True
INITIAL_LATENT_ROOT_DIR = '../exp/Chair/initial_latent_models'
FINAL_LATENT_ROOT_DIR = '../exp/Chair/final_latent_models'
MAX_LATENT_COUNT = 10

def createDirIfNonExistent(path):
      if not os.path.isdir(path):
            os.mkdir(path)

def getY(dataset):
  N = dataset.nImages
  Y = np.zeros((N, ref.J, 3))
  Y_raw = np.zeros((N, ref.J, 3))
  for i in range(N):
    y = dataset.annot[i, 0].copy()
    rotY, rotZ = dataset.meta[i, 0, 3:5].copy() / 180. * np.arccos(-1)
    Y[i] = np.dot(np.dot(RotMat('Z', rotZ), RotMat('Y', rotY)), y.transpose(1, 0)).transpose(1, 0)
    Y_raw[i] = y.copy()
  return Y, Y_raw

def getYHumans(dataset):
  N = dataset.nImages
  Y_raw = np.zeros((N, ref.J, 3))
  for i in range(dataset.len):
    _, y, _ = dataset.__getitem__(i)
    for k in range(dataset.nViews):
        Y_raw[(i*dataset.nViews) + k] = y[k].copy()
  return Y_raw, Y_raw
 
  
def initLatent(loader, model, Y, nViews, S, AVG = False, dial=False):
  model.eval()
  if dial:
    print 'dial activated (from init_latent)'
    model.train()
  nIters = len(loader)
  N = loader.dataset.nImages 
  M = np.zeros((N, ref.J, 3))
  bar = Bar('==>', max=nIters)
  sum_sigma2 = 0
  cnt_sigma2 = 1
  initial_latent_count = 0
  for i, (input, target, meta) in enumerate(loader):
    output = (model(input.cuda()).data).cpu().numpy()
    G = output.shape[0] / nViews
    output = output.reshape(G, nViews, ref.J, 3)
    if AVG:
      for g in range(G):
        id = int(meta[g * nViews, 1])
        for j in range(nViews):
          RR, tt = horn87(output[g, j].transpose(), output[g, 0].transpose())
          MM = (np.dot(RR, output[g, j].transpose())).transpose().copy()
          M[id] += MM.copy() / nViews
    else:
      for g in range(G):
        #assert meta[g * nViews, 0] > 1 + ref.eps
        p = np.zeros(nViews)
        sigma2 = 0.1
        for j in range(nViews):
          for kk in range(Y.shape[0] / S):
            k = kk * S
            d = Dis(Y[k], output[g, j])
            sum_sigma2 += d 
            cnt_sigma2 += 1
            p[j] += np.exp(- d / 2 / sigma2)
            
        id = int(meta[g * nViews, 1])
        M[id] = output[g, p.argmax()]


        max_count = 10
        if SAVE_LATENT and g == 0 and initial_latent_count < MAX_LATENT_COUNT:
              initial_latent_count += 1
              createDirIfNonExistent(INITIAL_LATENT_ROOT_DIR)
              dir_name = 'latent_%d' % (id)
              dir_name = os.path.join(INITIAL_LATENT_ROOT_DIR, dir_name)
              createDirIfNonExistent(dir_name)
              lt_fname = 'latent_model_id_%d_img_%d.lt' % (id, p.argmax())
              lt_fname = os.path.join(dir_name, lt_fname)
              np.save(lt_fname, M[id])
              for j in range(nViews):
                    m_fname = 'M_%d' % (j)
                    m_fname = os.path.join(dir_name, m_fname)
                    np.save(m_fname, output[g, j])
                    img_fname = 'img_%d.png'% (j)
                    img_fname = os.path.join(dir_name, img_fname)
                    cv2.imwrite(img_fname, 
                                (input[g * nViews + j] * 255).numpy().transpose(1, 2, 0).astype(np.uint8))
        
        if DEBUG and g == 0:
          print 'M[id]', id, M[id], p.argmax()
          debugger = Debugger()
          for j in range(nViews):
            RR, tt = horn87(output[g, j].transpose(), output[g, p.argmax()].transpose())
            MM = (np.dot(RR, output[g, j].transpose())).transpose().copy()
            debugger.addPoint3D(MM, 'b')
            debugger.addImg(input[g * nViews + j].numpy().transpose(1, 2, 0), j)
          debugger.showAllImg()
          debugger.addPoint3D(M[id], 'r')
          debugger.show3D()
        
    
    Bar.suffix = 'Init    : [{0:3}/{1:3}] | Total: {total:} | ETA: {eta:} | Dis: {dis:.6f}'.format(i, nIters, total=bar.elapsed_td, eta=bar.eta_td, dis = sum_sigma2 / cnt_sigma2)
    bar.next()
  bar.finish()
  #print 'mean sigma2', sum_sigma2 / cnt_sigma2
  return M
  
def stepLatent(loader, model, M_, Y, nViews, lamb, mu, S, call_count=-1, dial=False):
  model.eval()
  if dial:
    model.train()
    print 'dial activated (from step_latent)'
  nIters = len(loader)
  if nIters == 0:
    return None
  N = loader.dataset.nImages
  M = np.zeros((N, ref.J, 3))
    
  bar = Bar('==>', max=nIters)
  ids = []
  Mij = np.zeros((N, ref.J, 3))
  err, num = 0, 0
  latent_count = 0
  for i, (input, target, meta) in enumerate(loader):
    output = (model(input.cuda()).data).cpu().numpy()
    G = output.shape[0] / nViews
    output = output.reshape(G, nViews, ref.J, 3)
    for g in range(G):
      #assert meta[g * nViews, 0] > 1 + ref.eps
      id = int(meta[g * nViews, 1])
      ids.append(id)
      #debugger = Debugger()
      for j in range(nViews):
        Rij, tt = horn87(output[g, j].transpose(), M_[id].transpose())
        Mj = (np.dot(Rij, output[g, j].transpose()).copy()).transpose().copy()
        err += ((Mj - M_[id]) ** 2).sum()
        num += 1
        Mij[id] = Mij[id] + Mj / nViews 
        #print 'id, j, nViews', id, j, nViews
        #debugger.addPoint3D(Mj, 'b')
      #debugger.addPoint3D(M_[id], 'r')
      #debugger.show3D()
      
    Bar.suffix = 'Step Mij: [{0:3}/{1:3}] | Total: {total:} | ETA: {eta:} | Err : {err:.6f}'.format(i, nIters, total=bar.elapsed_td, eta=bar.eta_td, err = err / num)
    bar.next()
  bar.finish()
  if mu < ref.eps:
    for id in ids:
      M[id] = Mij[id]
    return M
  
  Mi = np.zeros((N, ref.J, 3))
  bar = Bar('==>', max=len(ids))
  err, num = 0, 0
  for i, id in enumerate(ids):
    dis = np.ones((Y.shape[0])) * oo
    for kk in range(Y.shape[0] / S):
      k = kk * S
      dis[k] = Dis(Y[k], M_[id])
    minK = np.argmin(dis)
    Ri, tt = horn87(Y[minK].transpose(), M_[id].transpose())
    Mi_ = (np.dot(Ri, Y[minK].transpose())).transpose()
    Mi[id] = Mi[id] + Mi_
    err += dis[minK]
    num += 1
    Bar.suffix = 'Step Mi : [{0:3}/{1:3}] | Total: {total:} | ETA: {eta:} | Err: {err:.6f}'.format(i, len(ids), total=bar.elapsed_td, eta=bar.eta_td, err = err / num)
    bar.next()
  bar.finish()
  
  tI = np.zeros((Y.shape[0] / S, 3))
  MI = np.zeros((N, ref.J, 3))
  cnt = np.zeros(N)
  bar = Bar('==>', max=Y.shape[0] / S)
  err, num = 0, 0
  for kk in range(Y.shape[0] / S):
    k = kk * S
    dis = np.ones((N)) * oo
    for id in ids:
      dis[id] = Dis(Y[k], M_[id])
    minI = np.argmin(dis)
    RI, tt = horn87(Y[k].transpose(1, 0), M_[minI].transpose(1, 0))
    MI_ = (np.dot(RI, Y[k].transpose())).transpose()
    err += ((MI_ - M_[minI]) ** 2).sum()
    num += 1
    MI[minI] = MI[minI] + MI_
    cnt[minI] += 1
    Bar.suffix = 'Step MI : [{0:3}/{1:3}] | Total: {total:} | ETA: {eta:} | Err: {err:.6f}'.format(kk, Y.shape[0] / S, total=bar.elapsed_td, eta=bar.eta_td, err = err / num)
    bar.next()
  bar.finish()
  
  for id in ids:
    M[id] = (Mij[id] * (lamb / mu) + Mi[id] + MI[id] / (Y.shape[0] / S) * len(ids)) / (lamb / mu + 1 + cnt[id] / (Y.shape[0] / S) * (len(ids)))
    #print 'Latent updated!'
    if SAVE_LATENT and latent_count < MAX_LATENT_COUNT:
      latent_count += 1
      createDirIfNonExistent(FINAL_LATENT_ROOT_DIR)
      dir_name = 'final_latent_%d' % (id)
      dir_name = os.path.join(FINAL_LATENT_ROOT_DIR, dir_name)
      createDirIfNonExistent(dir_name)
      lt_fname = 'final_latent_model_%d.lt' % (call_count)
      lt_fname = os.path.join(dir_name, lt_fname)
      np.save(lt_fname, M[id])


  if DEBUG:
    for id in ids:
      debugger = Debugger()
      debugger.addPoint3D(M[id], 'b')
      debugger.addPoint3D(M_[id], 'r')
      debugger.show3D()
  return M
