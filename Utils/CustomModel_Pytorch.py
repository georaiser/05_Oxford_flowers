#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 18:44:13 2022

@author: hellraiser
"""


# PyTorch libraries and modules
import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d 
from torch.nn import Module, Softmax, BatchNorm2d, Dropout
# PyTorch CNN architecture

class NeuralNet(Module):
    def __init__(self, num_class):
        super(NeuralNet, self).__init__()
        
        self.num_class = num_class
        
        self.cnn_model = Sequential(
            Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(0.4))
            
        self.fc_model = Sequential(
            Linear(2048,500),
            ReLU())
            
        self.classifier = Linear(500, num_class)

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        x = self.classifier(x)
        
        return x