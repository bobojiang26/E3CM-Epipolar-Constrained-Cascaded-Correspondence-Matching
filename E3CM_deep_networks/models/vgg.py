import numpy as np
import cv2 as cv
import torch
import torch.hub
from torchvision import models, transforms
from collections import namedtuple
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Vgg19(torch.nn.Module):
    
    # modified from the original @ https://github.com/chenyuntc/pytorch-book/blob/master/chapter08-neural_style/PackedVGG.py
    
    def __init__(self, batch_normalization = True, device=None, required_layers = None):
        
        if required_layers == None and batch_normalization:
            required_layers = [3, 10, 17, 30, 46]
        elif required_layers == None:
            required_layers = [2, 7, 12, 21, 32]
            
        # features 2，7，12，21, 30, 32: conv1_2,conv2_2,relu3_2,relu4_2,conv5_2,conv5_3
        super(Vgg19, self).__init__()
        self.device=device
        if batch_normalization:
            features = list(models.vgg19_bn(pretrained = True).features)[:47] # get vgg features
        else:
            features = list(models.vgg19(pretrained = True).features)[:33] # get vgg features
        
        self.features = torch.nn.ModuleList(features).eval() # construct network in eval mode
        
        for param in self.features.parameters(): # we don't need graidents, turn them of to save memory
            param.requires_grad = False
                
        self.required_layers = required_layers # record required layers to save them in forward
        
        for layer in required_layers[:-1]:
            self.features[layer+1].inplace = False # do not overwrite in any layer
        
    def forward(self, x):
        results = []  #activation of required layers
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in self.required_layers:
                results.append(x)
                #print(x.shape)
        vgg_outputs = namedtuple("VggOutputs", ['conv1_2', 'conv2_2', 'relu3_2', 'relu4_2', 'conv5_3'])
        
        return vgg_outputs(*results)