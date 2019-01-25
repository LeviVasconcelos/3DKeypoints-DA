import torch
import torch.nn as nn

class DomainAdaptationLayer(nn.Module):
    def __init__(self, planes):
        super(DomainAdaptationLayer, self).__init__()
        
        self.bn_source = nn.BatchNorm2d(planes)
        nn.init.constant_(self.bn_source.weight, 1)
        nn.init.constant_(self.bn_source.bias, 0)
        self.bn_source.weight.requires_grad = False
        self.bn_source.bias.requires_grad = False
        
        self.bn_target = nn.BatchNorm2d(planes)
        nn.init.constant_(self.bn_target.weight, 1)
        nn.init.constant_(self.bn_target.bias, 0)
        self.bn_target.weight.requires_grad = False
        self.bn_target.bias.requires_grad = False
        
        self.weight = nn.parameter.Parameter(torch.Tensor(planes))
        self.bias = nn.parameter.Parameter(torch.Tensor(planes))
        
        self.index = 0
  
    def set_domain(self, source=True):
        self.index = 0 if source else 1
  
    def forward(self, x):
        if self.index == 0:
            out = self.bn_source(x)
        else:
            out = self.bn_target(x)
        
        res = self.weight.view(1, self.weight.size()[0], 1, 1) * out + self.bias.view(1, self.weight.size()[0], 1, 1)
        return res
