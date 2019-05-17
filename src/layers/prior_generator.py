import torch
import torch.nn as nn
import ref
# Computes the pairwise distances within x, where x has dimensions BxKxD, K=10 and D=3 in our case and B=batch-size

#### NOTE #####
# From the computations here defined, distance from a keypoint of another is just: dist[:,idx1, idx2]
# Extracting the proportion from 1 edge (idx1->idx2) to another (idx3->idx4) is just: prop[:,idx1*10+idx2, idx3*10+idx4], 
# where 10 is the number of keypoints 

def compute_distances(x, eps=10**(-6)):
    if len(x.shape)==2:
    	x=x.unsqueeze(0)
    if (torch.isnan(x).sum() > 0):
        print('X ALREADY WITH NANS TO COMPUTE DISTS')
    # Computes the squared norm of X
    x_squared=x.norm(p=2,dim=2).pow(2) # B x K
    x_squared_left = x_squared.unsqueeze(-1) # B x K x 1
    x_squared_right = x_squared.unsqueeze(1) # B x 1 x K
    x_transposed = x.permute(0,2,1) # B x K x D
    xxT = torch.bmm(x,x_transposed)
    dists = x_squared_left + x_squared_right - 2*xxT  +eps
    if (torch.isnan(dists).sum() > 0):
        print('Distances NaN')
    #if ((torch.abs(dists[dists < 0]) > 0.05).sum() > 0):
    #    print('NEGATIVE DISTANCES')
    #    print('negative distances: ', dists[dists < 0])
    dists = (nn.functional.relu(dists)).pow(0.5)
    dists[dists > ref.distance_threshold] = ref.distance_threshold
    #if (torch.isnan(dists).sum() > 0):
    #    print('Distances naN after Relu')
    return  dists # B x K x K


def replicate_mask(x): # x must be of dimension BxK
    # Computes the outer product of a vector by itself
    x_left = x.unsqueeze(-1) # B x K x 1
    x_right = x.unsqueeze(1) # B x 1 x K

    xxT = torch.bmm(x_left,x_right) # B x K x K

    return  xxT

def get_shape_index(absolute_idx, shape):
      print('debug shape_index: ', absolute_idx, shape)
      shape = torch.Tensor(list(shape))
      tensor_idx = [0] * len(shape)
      buf = absolute_idx
      for i in range(len(tensor_idx)):
          num_el = torch.prod(shape[(i+1):]).item()
          tensor_idx[i] = int(buf // num_el)
          buf = buf % num_el
      print('found_idx: ', tuple(tensor_idx))
      return tuple(tensor_idx)
###########################################################################################################
mask_no_self_connections = torch.FloatTensor(1,ref.J,ref.J,ref.J,ref.J).zero_() + 1.
self_keypoint_props =  torch.FloatTensor(1,ref.J,ref.J,ref.J,ref.J).zero_()

for i in range(ref.J):
        self_keypoint_props[0,i,i,i,i]=1.
        for j in range(ref.J):
              if i==j: 
    		mask_no_self_connections[0,i,j,:,:]=0.0
    		#self_keypoint_props[0,i,j,i,j] = 1.
    		continue
              for l in range(ref.J):
                    for m in range(ref.J):
                          if l==m or (i==l and j==m) or (i==m and j==l):
                                mask_no_self_connections[0,i,j,l,m]=0.0
normalizer = (mask_no_self_connections + self_keypoint_props).to('cuda')

def compute_proportions(dists,eps=10**(-8), diag_mask=None): # x has dimensions B x K x K
    x = dists.view(dists.shape[0],-1) # B x K^2
    numerator = (x.unsqueeze(2)) # B x K^2 x 1
    denominator = 1./(x.unsqueeze(1)+1e-7) # B x 1 x K^2
    mm = torch.bmm(numerator,denominator) # B x K^2 x K^2
    if (torch.isnan(mm).sum() > 0):
       print('NaN computing proportions...')
    assert(torch.isnan(mm).sum() < 1)
    non_diag = mm.view(x.shape[0], ref.J, ref.J, ref.J, ref.J) * normalizer 
    idx = None
    if (torch.abs(non_diag) > 1e+6).sum() > 0:
         argmax = torch.argmax(non_diag)
         idx = get_shape_index(argmax, non_diag.shape)
         print('dists shape: ', dists.shape)
         print('[FROM CP_PROP] idx: ', idx)
         print('[FROM CP_PROP] non_diag.max: ', torch.max(non_diag))
         print('[FROM CP_PROP] proportions max (true): ', mm.view(x.shape[0], ref.J, ref.J, ref.J, ref.J)[idx])
         print('[FROM CP_PROP] proportions max: ', non_diag[idx])
         print('[FROM CP_PROP] distance bij [%d, %d, %d] : ' % (idx[0],idx[1], idx[2]), dists[idx[0]][idx[1:3]])
         print('[FROM CP_PROP] distance bkl: [%d, %d, %d] : ' % (idx[0],idx[3], idx[4]), dists[idx[0]][idx[3:5]])
         print('[FROM CP_PROP] reproduced_prop: ', dists[idx[0]][idx[1:3]]/(dists[idx[0]][idx[3:5]] + 1e-7))
         if (torch.abs(x) > 10).sum() > 0:
              print('FromProps: x weird ################')
    return mm, idx

def compute_masked_proportions(x,mask,eps=10**(-6)): # x has dimensions B x K x K
    x = x.view(x.sehape[0],-1) # B x K^2
    numerator = x.unsqueeze(2)*mask # B x K^2 x 1
    denominator = 1./(x.unsqueeze(1)+eps) # B x 1 x K^2
    denominator = denominator + 10*mask
    mm = torch.bmm(numerator,denominator) # B x K^2 x K^2
    return mm


def check_dist(x,d):
	for i in range(10):
		for j in range(10):
			mdist = (x[0,i]-x[0,j]).norm()**2
			if (d[0,i,j]-mdist)>10**(-6):
				print(d[0:,i,j]-mdist)


def check_props(x,p, eps=10**(-6)):
	for i in range(100):
		for j in range(100):
			mp = x[0,i]/(x[0,j]+eps)
			if (p[0,i,j]-mp)>10**(-6):
					print(p[0,i,j]-mp)






