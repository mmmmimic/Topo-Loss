# imports
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from matplotlib import pyplot as plt

import os

from copy import deepcopy

from topologylayer.nn import LevelSetLayer2D, TopKBarcodeLengths  

class TopoLoss(nn.Module):
    '''
    topological loss for semantic segmentation
    '''

    def __init__(self, img_size, topo_weight=1, sqdiff_weight=10, max_k=20):
        super().__init__()
        self.dgminfo = LevelSetLayer2D(size=img_size, sublevel=False, maxdim=1)
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.sqdiff_weight = sqdiff_weight # weight of the reproduction loss
        self.topo_weight = topo_weight # weight of L_topo in Eq. 5
        self.max_k = max_k # only consider this many bars - most will be 0-length anyway (note taken from the official implementation)

    def forward(self, y_pred, y_raw, topo_label, idx=[1]):
        '''
        args:
            y_pred: predicted probability of the new network, Tensor[B, C, H, W]
            y_raw: predicted probability of the first-trained network, Tensor[B, C, H, W]
            topo_label: correct topology of the input, Tensor[B, C, 2], the first column is the number of connected components and the second column is the number of holes
            idx: indices of catogories of interest, List[N]. idx=[1] by default, which fits the binary segmentation task. 
        returns:
            loss
        '''
        l_topo = 0
        for i in idx:
            l_topo += self._get_l_topo(y_pred.clone(), topo_label.clone(), i)

        l_sqdiff = self.l2_loss(y_raw, y_pred)

        l = self.topo_weight*l_topo/len(idx) + self.sqdiff_weight*l_sqdiff

        return l

    def _get_l_topo(self, y_pred, topo_label, idx):
        '''
        args:
            y_pred: predicted probability of the new network, Tensor[B, C, H, W]
            y_raw: predicted probability of the first-trained network, Tensor[B, C, H, W]
            topo_label: correct topology of the input, Tensor[B, C, 2]
            idx: the index number of a catogory of interest, Int
        returns:
            l_topo in Eq. 4
        '''        
        batch_size = y_pred.size(0)

        topo_loss = 0
        ## note that topologyLayer only work on batchsize=1
        for b in range(batch_size):
            pred = y_pred[b, idx, ...] # [H, W]
            label = topo_label[b, idx, :] # [2]

            homo_info = self.dgminfo(pred)

            # betti number at dim 0 counts the number of connected components
            dim_0_sq_bars = TopKBarcodeLengths(dim=0, k=self.max_k)(homo_info)**2
            bar_signs = torch.ones(self.max_k).to(pred.device)
            beta_0 = label[0]
            bar_signs[:beta_0] = -1
            l0 = (dim_0_sq_bars * bar_signs).sum()

            # betti number at dim 1 counts the number of loops or holes
            dim_1_sq_bars = TopKBarcodeLengths(dim=1, k=self.max_k)(homo_info)**2
            bar_signs = torch.ones(self.max_k).to(pred.device)
            beta_1 = label[1]
            bar_signs[:beta_1] = -1
            l1 = (dim_1_sq_bars * bar_signs).sum()

            topo_loss = topo_loss + l0 + l1

        return topo_loss/batch_size

class DemoNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=out_channels, kernel_size=3, stride=1, padding=1),    
        )
    
    def forward(self, x):
        x = self.conv(x)
        return torch.softmax(x, dim=1)

if __name__ == "__main__":
    
    raw_net = DemoNet(3, 5)
    
    new_net = deepcopy(raw_net)
    
    x = torch.rand(2, 3, 10, 10)
    topo_label = torch.ones(2, 5, 2).long()
    
    idx = [1,3] # we are interested in class 1 and 3
    
    with torch.no_grad():
        raw_net.train
        y_raw = raw_net(x).detach()
    
    new_net.train
    y_pred = new_net(x)

    criteria = TopoLoss(img_size=(10, 10))

    loss = criteria(y_pred, y_raw, topo_label, idx)
    
    print(loss)
    
    loss.backward()
