#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:30:29 2019
@author: levi
"""

import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
import h5py

import ref

from h36m_metadata import H36M_Metadata

def _draw_annot_from_file(img, bbox, pose):
      img2 = cv2.imread(img)
      cv2.rectangle(img2, tuple(bbox[0]), tuple(bbox[1]), (0, 255, 0), 3)
      for i in pose:
            cv2.circle(img2, tuple(i), 1, (255,0,0), -1)
      return img2

def _draw_annot(img, pose):
      print(img.shape)
      img2 = img.copy()
      for i in pose:
            cv2.circle(img2, tuple(i), 1, (255,0,0), -1)
      return img2

def Humans36mRGBSourceDataset(split, nViews, nImages=2000, normalized=True):
      subjects = [0, 1, 2] if split == 'train' else [5,6]
      return Humans36mDataset(nViews, split, True, nImages, subjects, meta_val=1, normalized=normalized)

def Humans36mRGBTargetDataset(split, nViews, nImages=2000, normalized=True):
      subjects = [3, 4] if split == 'train' else [5,6]
      return Humans36mDataset(nViews, split, True, nImages, subjects, meta_val=5, normalized=normalized)


def Humans36mDepthSourceDataset(split, nViews, nImages=2000, normalized=True):
      subjects = [0, 1, 2] if split == 'train' else [5,6]
      return Humans36mDataset(1, split, False, nImages, subjects, meta_val=1, normalized=normalized)

def Humans36mDepthTargetDataset(split, nViews, nImages=2000, normalized=True):
      subjects = [3, 4] if split == 'train' else [5,6]
      return Humans36mDataset(1, split, False, nImages, subjects, meta_val=5, normalized=normalized)


class Humans36mDataset(data.Dataset):
      def __init__(self, nViews, split='train', rgb=True, nPerSubject=2000, subjects = [0], meta_val=1, normalized=True):
            self.normalized = normalized
            self.root_dir = ref.Humans_dir
            self.rgb = rgb
            self.nViews = nViews if self.rgb else 1
            self.split = split
            self.kp_to_use = np.asarray([0, 1, 2, 3, 6, 7, 8, 13, 14, 17, 18, 19, 25, 26, 27])
            self.K = len(self.kp_to_use)
            self.meta_val = meta_val
            #self.kTrainSplit = 3
            self.metadata = H36M_Metadata(os.path.join(self.root_dir, 'metadata.xml'))
            self.imagesPerSubject = nPerSubject
            self.kBlacklist = { 
                        ('S11', '2', '2'),  # Video file is corrupted
                        ('S7', '15', '2'), # TOF video does not exists.
                        ('S5', '4', '2'), # TOF video does not exists.
                        }
            kSubjects = np.asarray(['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11'])
            self.subjects_to_include = kSubjects[subjects]
            self.kFolders = {
                        'rgb_cameras' : 'imageSequence',
                        'tof_data' : 'ToFSequence'
                        }
            self.kCameras = [str(x) for x in self.metadata.camera_ids]
            self.kMaxViews = len(self.kCameras)
            self.kAnnotationsFilename = 'annot.h5'
            self._build_indexes()
            self._build_access_index()
            self._build_meta()
            self._normalization = self._normalize_pose if self.normalized else (lambda a : a)
            print('**Dataset Loaded: split [%s], len[%d], views[%d], rgb[%s]' % (self.split, self.len, self.nViews, 'True' if self.rgb else 'False'))
      
      def _build_meta(self):
            self.meta = np.zeros((self.len, self.kMaxViews, ref.metaDim))
            for i in range(self.len):
                  for j in range(self.kMaxViews):
                        if self.rgb:
                              self.meta[i,j,0] = self.meta_val if self.split == 'train' else -self.meta_val
                        else:
                              self.meta[i,j,0] = self.meta_val if self.split == 'train' else -self.meta_val
                        self.meta[i,j, 1] = i
                        self.meta[i,j, 2] = j
                        self.meta[i, j, 3:5] = np.zeros((2), dtype=np.float32) / 180. * np.arccos(-1)
      
      def _process_subaction(self, subject, action, subaction):
            folder = os.path.join(self.root_dir, subject, self.metadata.action_names[action] + '-' + subaction)
            rgb_cameras_folder = os.path.join(folder, self.kFolders['rgb_cameras'])
            tof_folder = os.path.join(folder, self.kFolders['tof_data'])
            
            rgb_folder = os.path.join(rgb_cameras_folder, self.kCameras[0])
            #print('index length: ', len(index))
            self.means = []
            self.std = []
            # Fill in annotations
            annot_file = os.path.join(folder, self.kAnnotationsFilename)
            with h5py.File(annot_file, 'r') as file:
                  frames = file['frame']
                  unique_frames = np.unique(frames)
                  pose3d_norm = file['pose/3d-norm'].value
                  bbox = file['bbox'].value
                  pose2d = file['pose/2d'].value
                  cameras = file['camera'].value
                  pose3d = file['pose/3d'].value
                  # Normalization by the pelvis
                  pose3d = pose3d - pose3d[:, 0].reshape(pose3d.shape[0], 1, pose3d.shape[2])
                  pose3d_univ = file['pose/3d-univ'].value
                  pose3d_original = file['pose/3d-original'].value
                  intrinsic = file['intrinsics']
                  instrinsic_univ = file['intrinsics-univ']
                  index = [{'Views': [], 
                            'Annot': {
                                        'bbox': [], 
                                        '2d': [], 
                                        '3d':[], 
                                        'intrinsic':[], 
                                        'instrinsic-univ':[],
                                        '3d-univ' : [],
                                        '3d-orignial' : [],
                                        }, 
                            'TOF': []}.copy() for i in range(len(unique_frames))]
                  mapping = { f:i for i,f in enumerate(unique_frames) }
                  #print(pose3d_norm.shape)
                  try:
                        for i,f in enumerate(frames):
                              k = mapping[f]
                              rgb_folder = os.path.join(rgb_cameras_folder, str(cameras[i]))
                              filename = 'img_%06d.jpg' % f
                              tof_filename = 'tof_range%06d.jpg' % f
                              index[k]['Views'] += [os.path.join(rgb_folder, filename)]
                              index[k]['Annot']['bbox'] += [bbox[i]]
                              index[k]['Annot']['2d'] += [pose2d[i]]
                              index[k]['Annot']['3d'] += [pose3d[i][self.kp_to_use]]
                              index[k]['Annot']['3d-univ'] += [pose3d_univ[i]]
                              #cam = intrinsic[str(cameras[i])].value
                              index[k]['Annot']['intrinsic'] += [intrinsic[str(cameras[i])].value]
                              index[k]['Annot']['instrinsic-univ'] = [instrinsic_univ[str(cameras[i])].value]
                              index[k]['Annot']['3d-orignial'] += [pose3d_original[i]]
                              if len(index[k]['TOF']) == 0:
                                    index[k]['TOF'] = [os.path.join(tof_folder, tof_filename)]
                  except IndexError as e:
                        print(e)
            return index, pose3d[:, self.kp_to_use]
      
      def _build_indexes(self):
            self.dataset_indexes = []
            self.subject_max_idx = []
            self.all_poses = np.asarray([])
            subactions = []
            for subject in self.subjects_to_include:
                  subactions += [ 
                              (subject, action, subaction) 
                              for action, subaction in self.metadata.sequence_mappings[subject].keys() 
                              if int(action) > 1 and 
                              action not in ['54138969', '55011271', '58860488', '60457274']  # Exclude '_ALL' and Cameras
                              ]
            #print(subactions)
            last_subject, _, _ = subactions[0]
            for subject, action, subaction in subactions:
                  if (subject, action, subaction) in self.kBlacklist:
                        continue
                  if last_subject != subject:
                        last_subject = subject
                        self.subject_max_idx += [len(self.dataset_indexes)]
                  indexes, poses = self._process_subaction(subject, action, subaction)
                  self.all_poses = poses if len(self.all_poses) == 0 else np.concatenate((self.all_poses, poses))
                  self.dataset_indexes += indexes
            #print('Subject max idxes: ', self.subject_max_idx)
            self.subject_max_idx += [len(self.dataset_indexes)]
            self.len = len(self.dataset_indexes)
            self._compute_pose_statistics_and_free_poses()
            
      def _compute_pose_statistics_and_free_poses(self):
            self.poses_mean = np.mean(self.all_poses, 0)
            self.poses_std = np.std(self.all_poses, 0)
            del self.all_poses
      
      def _normalize_pose(self, pose):
            return (pose - self.poses_mean) / (self.poses_std + 1e-7)
      
      def _unnormalize_pose(self, pose):
	      return pose *  (torch.from_numpy(self.poses_std).float().to('cuda').unsqueeze(0) + 1e-7) + torch.tensor(self.poses_mean).float().to('cuda').unsqueeze(0)
      
      def _build_access_index(self):
            self.access_order = []
            last_subject = 0
            for i in self.subject_max_idx:
                  #print('building: ', last_subject, i)
                  to_use_images = np.arange(last_subject,i,1)[:self.imagesPerSubject]
                  #self.access_order += np.random.permutation(np.arange(last_subject,i,1))[:self.imagesPerSubject].tolist()
                  self.access_order += np.random.permutation(to_use_images).tolist()
                  last_subject = i
            np.random.shuffle(self.access_order)
            self.len = len(self.access_order)
            self.nImages = self.len * self.nViews
            
            
      def shuffle(self):
            self._build_access_index()
      
      def _get_ref(self, idx):
            return self.dataset_indexes[self.access_order[idx]]
      
      def _load_image(self, idx, view=0):
            image_type = 'Views' if self.rgb else 'TOF'
            try:
                  filename = self._get_ref(idx)[image_type][view]
                  #print('filename ', filename, ' view: ', view)
            except IndexError:
                  print('trying to access: ' + str(self.access_order[idx]) + ' out of: ' + str(self.len) + ' || ' + str(len(self.dataset_indexes)))
                  print(idx, self.access_order[idx], 'view: ', view)
                  #print('filename: ', filename)
            return cv2.imread(filename)
      
      def _get_normalization_statistics(self):
            return self.poses_mean, (self.poses_std + 1e-7)
      
      def __getitem__(self, index):
            idx = index % self.len
            #print(idx, self.access_order[idx])
            imgs = np.zeros((self.nViews, 224, 224, 3), dtype=np.float32)
            annots = np.zeros((self.nViews, self.K, 3), dtype=np.float32)
            mono_pose3d =  np.zeros((self.nViews, self.K, 3), dtype=np.float32)
            univ_pose3d = np.zeros((self.nViews, self.K, 3), dtype=np.float32)
            orig_pose3d = np.zeros((self.nViews, self.K, 3), dtype=np.float32)
            meta = np.zeros((self.nViews, ref.metaDim))
            pose_2d = []
            intrinsics = []
            
            for k in range(self.nViews):
                  imgs[k] = self._load_image(idx, k).astype(np.float32)
                  if self.rgb:
                       annots[k] = self._normalization(self._get_ref(idx)['Annot']['3d'][k].copy())
                       #annots[k] = self._get_ref(idx)['Annot']['3d'][k].copy()
                  else:
                       annots[k] = self._normalization(self._get_ref(idx)['Annot']['3d'][1].copy())
                       #annots[k] = self._get_ref(idx)['Annot']['3d'][1].copy()
                  #annots[k] = self._get_ref(idx)['Annot']['3d-norm'][k].copy()
                  #mono_pose3d[k] = self._get_ref(idx)['Annot']['3d'][k].copy()
                  #univ_pose3d[k] = self._get_ref(idx)['Annot']['3d-univ'][k].copy()
                  #orig_pose3d[k] = self._get_ref(idx)['Annot']['3d-orignial'][k].copy()
                  #pose_2d += [self._get_ref(idx)['Annot']['2d'][k].copy()]
                  #intrinsics += [self._get_ref(idx)['Annot']['intrinsic'][k].copy()]
                  meta[k] = self.meta[idx, k]
            imgs = imgs.transpose(0, 3, 1, 2) / 255.
            inp = torch.from_numpy(imgs)
            return inp, annots, meta #, mono_pose3d, univ_pose3d, orig_pose3d, intrinsics, pose_2d
      
      def __len__(self):
            return self.len
