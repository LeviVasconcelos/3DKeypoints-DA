import copy
import itertools

import torch
import torch.nn as nn
import torchvision.models as torch_models
import torchvision.models.resnet as resnet

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class DomainClassifier(nn.Module):

    def __init__(self, inplanes, blocks):
        super(DomainClassifier, self).__init__()
	self.layer4 = self._make_layer(resnet.Bottleneck, inplanes/2* resnet.Bottleneck.expansion, inplanes, blocks, stride=2)
        self.fc = nn.Linear(inplanes*resnet.Bottleneck.expansion, 2)
	self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
	self.inplanes=inplanes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
		nn.init.constant(m.bias, 0)

    def forward(self, x):
	x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
	x = self.fc(x)
        return x




    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or inplanes != planes * block.expansion:
		    downsample = nn.Sequential(
		        conv1x1(inplanes, planes * block.expansion, stride),
		        nn.BatchNorm2d(planes * block.expansion),)

		layers = []
		layers.append(block(inplanes, planes, stride, downsample))
		inplanes = planes * block.expansion
		for _ in range(1, blocks):
		    layers.append(block(inplanes, planes))

		return nn.Sequential(*layers)


class DistanceProjector(nn.Module):

    def __init__(self, inplanes, outplanes):
        super(DistanceProjector, self).__init__()
        self.fc = nn.Linear(inplanes, outplanes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal(m.weight, 0,0.001)

    def forward(self, x):
	x = self.fc(x)
        return x



class ResNet_ADDA(resnet.ResNet):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_ADDA, self).__init__(block, layers, num_classes=num_classes)
  


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        return x


def resnet50(num_classes=1000, pretrained=False):
    model = ResNet_ADDA(resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
                model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def get_discriminator():
	return DomainClassifier(512,3) 
