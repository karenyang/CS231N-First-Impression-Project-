from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import copy


# Adopted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class ResNet50_Noa_Karen(nn.Module):
    def __init__(self,model_pretained=None,num_classes=5):
        super(ResNet50_Noa_Karen, self).__init__()
        
        self.resnet = model_pretained
        self.sigmoid = nn.Sigmoid()   
        
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x) # everything before the final fc layer       
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)
        x = self.sigmoid(x)
        return x

# Adopted from: https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
# TODO: check whether this is usable and within memory limits
class DenseNet161_Noa_Karen(nn.Module):
    def __init__(self,model_pretained=None,num_classes=5):
        super(DenseNet161_Noa_Karen, self).__init__()      
        self.densenet = model_pretained
        self.sigmoid = nn.Sigmoid()           
    def forward(self, x):
        # TODO
        #x = self.densenet.features(x) # everything before the final classifier layer
        #x = x.view(x.size(0), -1)
        
        #x = self.new_fc(x)
        x = self.sigmoid(x)
        return x

def resnet50_noa_karen(num_classes = 5,verbose=False):
    model_pretained = torchvision.models.resnet50(pretrained=True)
    # All of the parameters are freezed, not to change (newly constructed layers' params won't be influenced)
    for param in model_pretained.parameters():
        param.requires_grad = False   
        
    model_pretained.fc = nn.Linear(model_pretained.fc.in_features, num_classes)
    model_base = ResNet50_Noa_Karen(model_pretained=model_pretained,num_classes=num_classes)
    if verbose:
        print(model_base.modules)
    return model_base

def densenet161_noa_karen(num_classes = 5, verbose=False):
    model_pretained = torchvision.models.densenet161(pretrained=True)
    # All of the parameters are freezed, not to change (newly constructed layers' params won't be influenced)
    for param in model_pretained.parameters():
        param.requires_grad = False    
    model_pretained.classifier = nn.Linear(model_pretained.classifier.in_features, num_classes)
    model_base = DenseNet161_Noa_Karen(model_pretained=model_pretained,num_classes=num_classes)
    if verbose:
        print(model_base.modules)
    return model_base
