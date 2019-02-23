import torch
import torch.nn as nn

# Computes the pairwise distances within x, where x has dimensions BxKxD, K=10 and D=3 in our case and B=batch-size

def compute_distances(x):
    # Computes the squared norm of X
    x_squared=x.norm(dim=2).pow(2)

    x_squared_left = x_squared.unsqueeze(-1) # B x K x 1
    x_squared_right = x_squared.unsqueeze(1) # B x 1 x K

    x_transposed = x.permute(0,2,1) # B x K x D
    xxT = torch.bmm(x,x_transposed)

    return x_squared_left + x_squared_right - 2*xxT # B x K x K

def compute_proportions(x,eps=0.00001) # x has dimensions B x K x K

    numerator = x.unsqueeze(1)
    denominator = 1./(x.unsueeze(2)+eps)

    return numerator/denominator
