import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchmetrics

from resnet_utils import *


def getEmbXy(i_epoch:int, d_depth:int, tensorFlag:bool=False):
    '''
    Returns design matrix X/target vector y of embedding vectors for a given epoch
    '''
    assert i_epoch in [-1,0, 5, 20, 99], "Only epochs: {0, 5, 20, 99}"
    assert d_depth in [0,1,2,3,4], "Depth parameter (from which embedding are extracted) must be in {0,1,...,4}."
    
    # path
    path_loc = Path(f'./checkpoints/resnet_3_mnist_e_{i_epoch}')

    # transforms
    TraFo = transforms.Compose([
        transforms.PILToTensor(),
        #transforms.CenterCrop((28,28)),
        transforms.RandomRotation(0.1)
    ])

    # dsets
    dset_train = datasets.MNIST(root='/eagle/projects/candle_aesp/siebenschuh/PT', train=True,  transform=TraFo)
    dset_test  = datasets.MNIST(root='/eagle/projects/candle_aesp/siebenschuh/PT', train=False, transform=TraFo)
    
    # RAW data return
    if(i_epoch==-1):
        X_tr = dset_train.data.numpy().reshape(len(dset_train), -1)
        y_tr = dset_train.targets.numpy()
        
        X_te = dset_test.data.numpy().reshape(len(dset_test), -1)
        y_te = dset_test.targets.numpy()
        
        # tensor-type output
        if(tensorFlag):
            return (dset_train.data, y_tr, dset_test.data, y_te)
        
        return (X_tr, y_tr, X_te, y_te)
    
    # load model
    model = ResNet(depth=3, inputDim=(1,28,28))
    model.load_state_dict(torch.load(path_loc))

    # dset loader
    b_size   = 1024
    loader_train = DataLoader(dataset=dset_train, batch_size=b_size, num_workers=5, shuffle=False)
    loader_test  = DataLoader(dataset=dset_test, batch_size=b_size, num_workers=5, shuffle=False)

    # device
    device = torch.device('cuda:0')

    # load model
    model = ResNet(depth=3, inputDim=(1,28,28))
    model.load_state_dict(torch.load(path_loc))
    model = model.to(device)

    # embeddings
    out_List      = []
    out_test_List = []

    # inference
    for j, batch in enumerate(loader_train):
        # unpack
        inputs, labels = batch
        # float
        inputs = inputs.type(torch.float)

        # -> device 13s
        inputs = inputs.to(device)
        labels = labels.to(device)

        # outputs
        outputs = model(inputs, d_depth)
        out_List.append(outputs.cpu().detach())

    # inference
    for j, batch in enumerate(loader_test):
        # unpack
        inputs, labels = batch
        # float
        inputs = inputs.type(torch.float)

        # -> device 13s
        inputs = inputs.to(device)
        labels = labels.to(device)

        # outputs
        outputs = model(inputs, d_depth)
        out_test_List.append(outputs.cpu().detach())

    # output tensor
    out     = torch.cat(out_List)
    outTest = torch.cat(out_test_List)

    # RAW
    y_tr = dset_train.targets.numpy()
    y_te = dset_test.targets.numpy()

    # EMB
    # - # design matrices
    X_emb_tr = out.numpy() #.shape
    X_emb_te = outTest.numpy()
    
    if(tensorFlag):
        return (out_List, y_tr, out_test_List, y_te)
    
    return (X_emb_tr, y_tr, X_emb_te, y_te)


def createTables(X_tr, y_tr, X_te, y_te):
    '''
    create train and test tables for various objective functions
    '''

    # parameter
    a1 = torch.zeros((28,28))
    for i in range(28):
        for j in range(28):
            if(29>=abs(i+j)>=27 or abs(i-j)<=1):
                a1[i,j]=1

    a2 = torch.zeros((14,14))
    for i in range(14):
        for j in range(14):
            if(i==7 or j==7):
                a2[i,j]=1


    def f1(x):
        '''Mean over image'''
        return x.mean(dim=(1,2)) 

    def f2(x):
        '''93.5% quantile over image'''
        return torch.quantile(a=x.reshape(x.shape[0],-1), q=0.935, dim=1)*-1

    def f3(x):
        '''Standard deviation over image'''
        return x.mean(dim=(1,2))

    def f4(x):
        '''Sum of mean and standard deviation over image'''
        return x.mean(dim=(1,2)) + x.mean(dim=(1,2)) 

    # set paras       
    a = a1
    b = a2

    # mappings
    def conv1(x:torch.Tensor):
        '''Cross Multiplication'''
        return (a*x).sum(dim=1).sum(dim=1)

    def conv2(x:torch.Tensor):
        '''Cross-Multiplication & pooling'''
        p = nn.MaxPool2d(2)
        return p(a*x).sum(dim=1).sum(dim=1)

    def conv3(x:torch.Tensor):
        '''Cross-Multiplication & pooling'''
        p = nn.MaxPool2d(2)
        return (p(a*x)*b).sum(dim=1).sum(dim=1)

    def conv4(x:torch.Tensor):
        '''Cross-Multiplication & pooling'''
        p = nn.MaxPool2d(2)
        return (p(p(a*x)*b)).sum(dim=1).sum(dim=1)

    def labelMap(x:torch.Tensor, y:torch.Tensor, cMap:Callable):

        f_x = cMap(x)

        f_ = (f_x - f_x.mean()) / f_x.std()
        y_ = (y - y.mean()) / y.std()

        return f_+y_


    data = {}
    for mode in ['tr', 'te']:

        # simple maps
        for f,name in zip([f1, f2, f3, f4], ['mu', 'q935', 'sig', 'mu+sig']):
            data.update({f'{name}_{mode}' : (f(X_tr) if mode=='tr' else f(X_te)).numpy()})

        # convolution maps
        for cM in [conv1, conv2, conv3, conv4]:
            data.update({f'{cM.__name__}_{mode}' : (cM(X_tr) if mode=='tr' else cM(X_te)).numpy()})

        # convolution maps + labels
        y = y_tr if mode=='tr' else y_te
        x = X_tr if mode=='tr' else X_te
        for cM in [conv1, conv2, conv3, conv4]:
            data.update({f'{cM.__name__}+label_{mode}' : labelMap(x=x, y=y, cMap=cM).numpy()})

    # add `test`
    data_tr = {k:v for k,v in data.items() if '_tr' in k}
    data_te = {k:v for k,v in data.items() if '_te' in k}

    # dataframe
    df_tr = pd.DataFrame(data_tr)
    df_te = pd.DataFrame(data_te)
    
    return df_tr, df_te

# helper
def lp(a:torch.tensor, b:torch.tensor, p:int=2) -> torch.tensor:
    return torch.norm(a-b,p=p, dim=1)

def standardize(x:torch.Tensor):
    return (x - x.mean()) / x.std()

def min_max(x:torch.Tensor):
    return (x - x.min()) / (x.max() - x.min())