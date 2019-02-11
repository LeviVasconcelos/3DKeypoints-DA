import torch.utils.data as data
import numpy as np
import torch
import cv2
import ref

class Fusion(data.Dataset):
  def __init__(self, SourceDataset, TargetDataset, nViews, targetRatio, totalTargetIm = 1):
    self.nViews = nViews
    self.targetRatio = targetRatio
    if ref.category == 'Chair':
      self.sourceDataset = SourceDataset('train', nViews)
      self.targetDataset = TargetDataset('train', nViews, totalTargetIm)
    else:
      self.sourceDataset = SourceDataset('train', nViews)
      self.targetDataset = TargetDataset('train', nViews)
    self.nSourceImages = len(self.sourceDataset)
    self.nTargetImages = int(self.nSourceImages * self.targetRatio)

    print '#Source images: {}, #Target images: {}'.format(self.nSourceImages, self.nTargetImages)
    
  def __getitem__(self, index):
    if index < self.nSourceImages: 
      return self.sourceDataset[index]
    else:
      return self.targetDataset[index - self.nSourceImages]

  def __len__(self):
    return (self.nSourceImages + self.nTargetImages)


#### Unpack function
__DEBUG = False
def unpack_splitted(data):
    input, target, meta = data
    sourceInput, sourceLabel, sourceMeta = torch.Tensor(), torch.Tensor(), torch.Tensor()
    targetInput, targetLabel, targetMeta = torch.Tensor(), torch.Tensor(), torch.Tensor()
    source_output, target_output = torch.Tensor(), torch.Tensor()
    inputDomains = [meta[i,0] for i in range(input.shape[0])]
    source_input_list = [x for i,x in enumerate(input) if abs(meta[i,0]) == 1]
    if (len(source_input_list) > 0): 
          sourceInput = torch.stack(source_input_list)
          sourceLabel = torch.stack([x for i,x in enumerate(target) if abs(meta[i,0]) == 1])
          sourceMeta = torch.stack([x for i,x in enumerate(meta) if abs(meta[i,0]) == 1])
          sourceDomains = [sourceMeta[i,0] for i in range(sourceMeta.shape[0])]
          if __DEBUG:
            print "source input shape: " + str(sourceInput.size())
            print "source label shape: " + str(sourceLabel.size())
            print "source meta shape: " + str(sourceMeta.size())
            print "source domain: " + str(sourceDomains)
    
    target_input_list = [x for i,x in enumerate(input) if abs(meta[i,0]) > 1]
    if (len(target_input_list) > 0):
          targetInput = torch.stack(target_input_list)
          targetLabel = torch.stack([x for i,x in enumerate(target) if abs(meta[i,0]) > 1])
          targetMeta = torch.stack([x for i,x in enumerate(meta) if abs(meta[i,0]) > 1])
          targetDomains = [targetMeta[i,0] for i in range(targetMeta.shape[0])]
          if __DEBUG:
            print "targe input shape:" + str(targetInput.size())
            print "target label shape:" + str(targetLabel.size())
            print "target meta shape:" + str(targetMeta.size())
            print "target Domain: " + str(targetDomains)
    return ((sourceInput, sourceLabel, sourceMeta), (targetInput, targetLabel, targetMeta))
      