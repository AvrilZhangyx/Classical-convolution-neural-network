# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:15:03 2019

@author: Admin
"""

import torch.nn as nn
import torch.nn.functional as F
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dense1 = nn.Linear(256*5*5,4096)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(4096,4096)
        self.drop2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(4096,10)

    def forward(self,x):
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        x=self.pool3(F.relu(self.conv5(F.relu(self.conv4(F.relu(self.conv3(x)))))))
        x=x.view(-1,256*5*5)
        x=self.dense3(self.drop2(F.relu(self.dense2(self.drop1(F.relu(self.dense1(x)))))))
        return x
