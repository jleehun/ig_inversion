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

def pgd_attack(model, images, labels, device, eps=0.3, alpha=2/255, iters=24):
    interp = [images]

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
    # print(1)
    interp = torch.stack(interp)
    return interp

