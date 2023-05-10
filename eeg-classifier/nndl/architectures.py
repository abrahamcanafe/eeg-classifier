#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 20:49:48 2023

@author: abrahamcanafe
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

n_time_steps=500
"""
Note: n_spatial_filters is a misnomer and actually refers to the number of temporal filters used.
"""

# %% ######### DEEP CNN
class DeepConvNet(nn.Module):
    def __init__(self, input_shape=(22,n_time_steps), n_input_channels = 22, n_spatial_filters=50, n_classes=4):
        super().__init__()
        
            
        self.input_shape = input_shape
        self.n_input_channels = n_input_channels
        self.n_spatial_filters = n_spatial_filters
        self.n_classes = n_classes
               
        self.conv1 = nn.Conv2d(self.n_input_channels, self.n_spatial_filters, (1,10))
        self.conv_bn1 = nn.BatchNorm2d(self.n_spatial_filters)
        self.pool1 = nn.AvgPool2d((1,3))
        self.drop1 = nn.Dropout(0.5)
        
        self.conv2 = nn.Conv2d(self.n_spatial_filters, 100, (1,10))
        self.conv_bn2 = nn.BatchNorm2d(100)
        self.pool2 = nn.AvgPool2d((1,3))
        self.drop2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(100, 200, (1,10))
        self.conv_bn3 = nn.BatchNorm2d(200)
        self.pool3 = nn.AvgPool2d((1,3))
        self.drop3 = nn.Dropout(0.5)
        
        self.conv4 = nn.Conv2d(200, 400, (1,10))
        self.conv_bn4 = nn.BatchNorm2d(400)
        self.pool4 = nn.AvgPool2d((1,3))
        self.drop4 = nn.Dropout(0.5)

        self.fc_net1 = nn.LazyLinear(4)
        self.elu = nn.ELU()
        return
    
    
    def forward(self, x):
        h = x.view(-1, self.input_shape[0], 1, self.input_shape[1])
        
        h = self.conv1(h)
        h = self.elu(h)
        h = self.pool1(h)
        h = self.conv_bn1(h)
        h = self.drop1(h)
        
        h = self.conv2(h)
        h = self.elu(h)
        h = self.pool2(h)
        h = self.conv_bn2(h)
        h = self.drop2(h)
        
        h = self.conv3(h)
        h = self.elu(h)
        h = self.pool3(h)
        h = self.conv_bn3(h)
        h = self.drop3(h)
        
        #h = self.conv4(h)
        #h = self.elu(h)
        #h = self.pool4(h)
        #h = self.conv_bn4(h)
        #h = self.drop4(h)
                
        h = h.view(h.shape[0],-1)
        h = self.fc_net1(h)
        h = self.elu(h)
        
        return h
    
    
# %% ########## CNN with multiheaded attention
class DeepConvLSTM(nn.Module):
    def __init__(self, input_shape=(22,n_time_steps), n_input_channels = 22, n_spatial_filters=50, n_classes=4):
        super().__init__()
        
            
        self.input_shape = input_shape
        self.n_input_channels = n_input_channels
        self.n_spatial_filters = n_spatial_filters
        self.n_classes = n_classes
               
        self.conv1 = nn.Conv2d(self.n_input_channels, self.n_spatial_filters, (1,10))
        self.conv_bn1 = nn.BatchNorm2d(self.n_spatial_filters)
        self.pool1 = nn.AvgPool2d((1,3))
        self.drop1 = nn.Dropout(0.5)
        
        self.conv2 = nn.Conv2d(self.n_spatial_filters, 100, (1,10))
        self.conv_bn2 = nn.BatchNorm2d(100)
        self.pool2 = nn.AvgPool2d((1,3))
        self.drop2 = nn.Dropout(0.5)
     
        self.conv3 = nn.Conv2d(100, 200, (1,10))
        self.conv_bn3 = nn.BatchNorm2d(200)
        self.pool3 = nn.AvgPool2d((1,3))
        self.drop3 = nn.Dropout(0.5)
        
        
        self.conv4 = nn.Conv2d(200, 400, (1,10))
        self.conv_bn4 = nn.BatchNorm2d(400)
        self.pool4 = nn.AvgPool2d((1,3))
        self.drop4 = nn.Dropout(0.5)
     
        self.fc_net1 = nn.LazyLinear(100)
        self.lstm1 = nn.LSTM(100, 10, batch_first=True, dropout=0.2)
        self.drop_lstm1 = nn.Dropout(0.5)
        self.fc_net2 = nn.Linear(10,4)
        self.elu = nn.ELU()
        return
    
    
    def forward(self, x):
        h = x.view(-1, self.input_shape[0], 1, self.input_shape[1])
        
        h = self.conv1(h)
        h = self.elu(h)
        h = self.pool1(h)
        h = self.conv_bn1(h)
        h = self.drop1(h)
        
        h = self.conv2(h)
        h = self.elu(h)
        h = self.pool2(h)
        h = self.conv_bn2(h)
        h = self.drop2(h)
        
        #h = self.conv3(h)
        #h = self.elu(h)
        #h = self.pool3(h)
        #h = self.conv_bn3(h)
        #h = self.drop3(h)
        
        #h = self.conv4(h)
        #h = self.elu(h)
        #h = self.pool4(h)
        #h = self.conv_bn4(h)
        #h = self.drop4(h)
                
        h = h.view(h.shape[0],-1)
        h = self.fc_net1(h)
        
        h, (h_0, c_0) = self.lstm1(h)
        h = self.drop_lstm1(h)     
        
        h = self.fc_net2(h)
        h = self.elu(h)
        
        return h
    
    
# %% ###########
class DeepConvTransformer(nn.Module):
    def __init__(self, input_shape=(22,n_time_steps), n_input_channels = 22, n_spatial_filters=50, n_classes=4):
        super().__init__()
        
            
        self.input_shape = input_shape
        self.n_input_channels = n_input_channels
        self.n_spatial_filters = n_spatial_filters
        self.n_classes = n_classes
               
        self.conv1 = nn.Conv2d(self.n_input_channels, self.n_spatial_filters, (1,10))
        self.conv_bn1 = nn.BatchNorm2d(self.n_spatial_filters)
        self.pool1 = nn.AvgPool2d((1,3))
        self.drop1 = nn.Dropout(0.5)
        
        self.conv2 = nn.Conv2d(self.n_spatial_filters, 100, (1,10))
        self.conv_bn2 = nn.BatchNorm2d(100)
        self.pool2 = nn.AvgPool2d((1,3))
        self.drop2 = nn.Dropout(0.5)
     
        self.conv3 = nn.Conv2d(100, 200, (1,10))
        self.conv_bn3 = nn.BatchNorm2d(200)
        self.pool3 = nn.AvgPool2d((1,3))
        self.drop3 = nn.Dropout(0.5)
        
        
        self.conv4 = nn.Conv2d(200, 400, (1,10))
        self.conv_bn4 = nn.BatchNorm2d(400)
        self.pool4 = nn.AvgPool2d((1,3))
        self.drop4 = nn.Dropout(0.5)
     
        self.fc_net1 = nn.LazyLinear(100)
        
        self.transformer_encoder1 = nn.TransformerEncoderLayer(d_model=100, nhead=10, dropout=0.5, batch_first=True)
        self.drop_head1 = nn.Dropout(0.5)

        self.fc_net2 = nn.Linear(100,4)
        self.elu = nn.ELU()
        return
    
    
    def forward(self, x):
        h = x.view(-1, self.input_shape[0], 1, self.input_shape[1])
        
        h = self.conv1(h)
        h = self.elu(h)
        h = self.pool1(h)
        h = self.conv_bn1(h)
        h = self.drop1(h)
        
        h = self.conv2(h)
        h = self.elu(h)
        h = self.pool2(h)
        h = self.conv_bn2(h)
        h = self.drop2(h)
        
        #h = self.conv3(h)
        #h = self.elu(h)
        #h = self.pool3(h)
        #h = self.conv_bn3(h)
        #h = self.drop3(h)
        
        #h = self.conv4(h)
        #h = self.elu(h)
        #h = self.pool4(h)
        #h = self.conv_bn4(h)
        #h = self.drop4(h)
                
        h = h.view(h.shape[0],-1)
        h = self.fc_net1(h)

        h = self.transformer_encoder1(h)
        h = self.drop_head1(h)
        
        h = self.fc_net2(h)
        h = self.elu(h)
        
        return h  