import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

def pgd_attack(model, images, labels, device, eps=0.3, iters=24, alpha=2/255):
    interp = [images.clone().detach().cpu()]

    images = images.unsqueeze(0).to(device)
    labels = torch.tensor([labels]).to(device)
    loss = nn.CrossEntropyLoss()
        
    ori_images = images.data

    
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        
        interp.append(images.squeeze(0).clone().detach().cpu())
    
    interp = torch.stack(interp)
    return interp

def untarget_fgsm(model, image, labels, device, eps=0.3, iter=24, **kwargs):

    interp = [image.clone().detach().cpu()]
    
    images = image.unsqueeze(0).detach().clone().to(device)
    labels = torch.tensor([labels]).to(device)
    model = model.to(device)    
    loss = nn.CrossEntropyLoss()
    
    for i in range(iter) :    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        attack_images = images + eps*images.grad.sign()
        images= attack_images.detach().clone()
        interp.append(attack_images.squeeze(0).clone().detach().cpu())    
    
    interp = torch.stack(interp)
    
    return interp

def simple_untarget_fgsm(model, image, labels, device, eps=0.3, iter=24, **kwargs) :

    interp = [image.clone().detach().cpu()]
    
    images = image.unsqueeze(0).detach().clone().to(device)
    labels = torch.tensor([labels]).to(device)
    model = model.to(device)
    
    loss = nn.CrossEntropyLoss()
    
    images.requires_grad = True
            
    outputs = model(images)
    
    model.zero_grad()
    cost = loss(outputs, labels)
    cost.backward()
    
    for _ in range(iter):
        attack_images = images + eps*images.grad.sign()
        interp.append(attack_images.squeeze(0).clone().detach().cpu())
    interp = torch.stack(interp) # 25 3 32 32
    return interp


# https://github.com/Harry24k/CW-pytorch/blob/master/CW.ipynb

# CW-L2 Attack
# Based on the paper, i.e. not exact same version of the code on https://github.com/carlini/nn_robust_attacks
# (1) Binary search method for c, (2) Optimization on tanh space, (3) Choosing method best l2 adversaries is NOT IN THIS CODE.
def cw_l2_attack(model, images, labels, device, c=1e-4, kappa=0, iters=24, learning_rate=0.01, **kwargs) :
    
    interp =[images.clone().detach().cpu()]
        
    images = images.unsqueeze(0).to(device) 
    
    w = torch.zeros_like(images, requires_grad=True)#.to(device)

    optimizer = optim.Adam([w], lr=learning_rate)
    
    for iter in range(iters) :

        a = 1/2*(nn.Tanh()(w) + 1)
        interp.append(a.squeeze(0).clone().detach().cpu())

        loss1 = nn.MSELoss(reduction='sum')(a, images)
        
        output = model(a)
        one_hot_label = torch.eye(len(output[0]))[labels].to(device)
        i, _ = torch.max((1-one_hot_label)*output, dim=1)
        j = torch.masked_select(output, one_hot_label.bool())
                
        temp = torch.clamp(j-i, min=-kappa)
        
        loss2 = torch.sum(c*temp)
        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    interp1 = torch.stack(interp)
    
    del w, interp, a
    return interp1