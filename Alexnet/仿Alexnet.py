# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:24:47 2019

@author: Admin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels,kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False
    
    def set_mask(self, mask):
        self.mask =mask
        self.weight.data = self.weight.data*self.mask.data
        self.mask_flag = True
    
    def get_mask(self):
        print(self.mask_flag)
        return self.mask
    
    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight*self.mask
            return F.conv2d(x, weight, self.bias, self.stride,self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)
  
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = MaskedConv2d(in_channels=1,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=1)
        self.conv2 = MaskedConv2d(in_channels=64,out_channels=192,kernel_size=3,padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = MaskedConv2d(in_channels=192,out_channels=384,kernel_size=3,padding=1)
        self.conv4 = MaskedConv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = MaskedConv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dense1 = nn.Linear(256*6*6,4096)
        self.dense2 = nn.Linear(4096,4096)
        self.dense3 = nn.Linear(4096,10)

    def forward(self, x):
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        x=self.pool3(F.relu(self.conv5(F.relu(self.conv4(F.relu(self.conv3(x)))))))
        x=x.view(-1,256*6*6)
        x=self.dense3(F.relu(self.dense2(F.relu(self.dense1(x)))))
        return x

    def set_masks1(self, masks):
        self.conv1.set_mask(masks[0])
        self.conv2.set_mask(masks[1])
        self.conv3.set_mask(masks[2])
        self.conv4.set_mask(masks[3])
        self.conv5.set_mask(masks[4])
    def set_masks2(self, masks):
        self.conv1.set_mask(torch.from_numpy(masks[0]))
        self.conv2.set_mask(torch.from_numpy(masks[1]))