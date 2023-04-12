# https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch/blob/main/Facial_Identity_Classification_Test_with_CelebA_HQ.ipynb

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt

import time
import os

class BaseClassifier():
    def __int__(self, ):
        pass 
    
    def forward(self, x):
        return x 
    
def get_classifier(name, num_labels, path = '/root/data' ):    
    if name == 'resnet':
#         path = '/root/pretrained/facial_identity_classification_transfer_learning_with_ResNet18.pth'
        model = models.resnet18(pretrained=True)        
        num_features = model.fc.in_features
        print(num_features)
        print(num_labels)
        model.fc = nn.Linear(num_features, num_labels) # multi-class classification (num_of_class == 307)
        model.load_state_dict(torch.load(path))

    return model
        
