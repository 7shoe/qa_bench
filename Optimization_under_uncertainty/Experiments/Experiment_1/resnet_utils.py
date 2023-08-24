import os
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_c:int, downsample:bool, bias:bool=False):
        super().__init__()
            
        self.in_c       = in_c
        self.bias       = bias
        self.downsample = downsample

        if(self.downsample):
            self.conv1 = nn.Conv2d(self.in_c, 2*self.in_c, kernel_size=3, stride=2, padding=1, bias=self.bias)
            self.bn1   = nn.BatchNorm2d(2*self.in_c)
            self.relu  = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(2*self.in_c, 2*self.in_c, kernel_size=3, stride=1, padding=1, bias=self.bias)
            self.bn2   = nn.BatchNorm2d(2*self.in_c)
            self.down  = nn.Sequential(nn.Conv2d(in_c, 2*self.in_c, kernel_size=1, stride=2, bias=self.bias),
                                       nn.BatchNorm2d(2*self.in_c)) 
        else:
            self.conv1 = nn.Conv2d(self.in_c, self.in_c, kernel_size=3, stride=1, padding=1, bias=self.bias)
            self.bn1   = nn.BatchNorm2d(self.in_c)
            self.relu  = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(self.in_c, self.in_c, kernel_size=3, stride=1, padding=1, bias=self.bias)
            self.bn2   = nn.BatchNorm2d(self.in_c)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if(self.downsample):
            identity = self.down(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self, depth:int, nClasses:int=10, inputDim:Tuple[int] = (3,32,32)):
        super().__init__()
        
        assert depth<=5, 'Depth must be smaller or equal to 5.'
        
        self.depth = depth
        self.nClasses = nClasses
        self.inputDim = inputDim
        self.flatten  = nn.Flatten()
        
        # initial layer
        modDict = nn.ModuleDict([['layer0', nn.Sequential(nn.Conv2d(inputDim[0], 64, kernel_size=3, stride=2, padding=3),
                                                          nn.BatchNorm2d(64),
                                                          #nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                                          nn.ReLU())]])
        # core layers
        loc_c = 64
        for i in range(1, depth+1):
            # current layer
            if(i==1):
                modDictLoc = nn.ModuleDict([(f'layer{i}', nn.Sequential(BasicBlock(in_c=loc_c, downsample=False),
                                                                        BasicBlock(in_c=loc_c, downsample=False)))])
            else:
                modDictLoc = nn.ModuleDict([(f'layer{i}', nn.Sequential(BasicBlock(in_c=loc_c,   downsample=True),
                                                                        BasicBlock(in_c=2*loc_c, downsample=False)))])
                loc_c*=2
            # update
            modDict.update(modDictLoc)


        # check feature dimension
        modDict.update(nn.ModuleDict([(f'layer{depth+1}', nn.AdaptiveAvgPool2d(output_size=(1,1)))]))
        x_in = torch.zeros((1,*inputDim))
        for i in range(depth+2):
            x_in = modDict[f'layer{i}'](x_in)
        self.in_feat = np.prod(x_in.shape[1:])

        # end layer
        modDict.update(nn.ModuleDict([(f'layer{depth+2}', nn.Sequential(nn.Flatten(),
                                                                        nn.Linear(in_features=self.in_feat, out_features=nClasses, bias=True)))]))

        # register parameters
        self.modDict = modDict
        
    def forward(self, x, emb_i=-1):
        assert -1<=emb_i<=self.depth+1, "Embedding argument `emb_i must be in {0,1,...,depth+1}."
        
        for i in range(self.depth+3):
            x = self.modDict[f'layer{i}'](x)
            if(emb_i==i):
                return self.flatten(x)
            
        return x