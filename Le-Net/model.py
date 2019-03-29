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

        self.conv1 = MaskedConv2d(1, 6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = MaskedConv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)


        self.linear1 = nn.Linear(16*5*5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)
        #定义卷积层，1个输入通道，6个输出通道，5*5的卷积filter，外层补上了两圈0,因为输入的是32*32
        #第二个卷积层，6个输入，16个输出，5*5的卷积filter 

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        return out

    def set_masks1(self, masks):
        self.conv1.set_mask(masks[0])
        self.conv2.set_mask(masks[1])
        
    def set_masks2(self, masks):
        self.conv1.set_mask(torch.from_numpy(masks[0]))
        self.conv2.set_mask(torch.from_numpy(masks[1]))