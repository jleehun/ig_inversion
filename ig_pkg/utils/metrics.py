
import torch 
import torch.nn as nn
import numpy as np


def morf(input, label, attr, model, device, ratio, **kwargs):
    x = mask_MoRF(input, attr, ratio).unsqueeze(0)
    x = x.to(device)
    y_hat = model.forward(x).argmax(dim=-1)
    label = torch.tensor(label)
    score = (y_hat == label).sum().item()
    return  score


def lerf(input, label, attr, model, device, ratio, **kwargs):
    x = mask_LeRF(input, attr, ratio).unsqueeze(0)
    x = x.to(device)
    y_hat = model.forward(x).argmax(dim=-1)
    label = torch.tensor(label)
    score = (y_hat == label).sum().item()
    return  score

def aopc(input, label, attr, model, device, ratio, **kwargs):
    # input 3, 224, 224 / attr 224, 224
        
    logit = model.forward(input.unsqueeze(0))    
    score_orig = nn.functional.softmax(logit, dim = -1)    
    prob_orig = score_orig[0, label].item()
    
    input_hat = mask_MoRF(input, attr, ratio).unsqueeze(0)
    logit_new = model(input_hat)
    score_new = nn.functional.softmax(logit_new, dim = -1)
    prob_new = score_new[0, label].item()
    
    metric_aopc = prob_orig - prob_new  
    
    return metric_aopc

def lodds(input, label, attr, model, device, ratio, **kwargs):
        
    logit = model.forward(input.unsqueeze(0))    
    score_orig = nn.functional.softmax(logit, dim = -1)    
    prob_orig = score_orig[0, label].item()
    
    input_hat = mask_MoRF(input.squeeze(0), attr, ratio).unsqueeze(0)
    logit_new = model(input_hat)
    score_new = nn.functional.softmax(logit_new, dim = -1)
    prob_new = score_new[0, label].item()
    
    metric_lodds = np.log(prob_new / (prob_orig + 1e-5))        
    
    return metric_lodds
# ------------------------------------------------------------------------

def mask_MoRF(x, attr, ratio):
    original_size = x.size() 
    x = x.reshape(3, -1) 
    attr = torch.tensor(attr).flatten() 
    v, index = torch.sort(attr, descending=True, dim=0)    
    index = index[:int(x.size(1)*ratio)]
#     print(x.size(1)*ratio)
    x[:, index] = 0.0 
    x = x.reshape(*original_size)
    return x 

def mask_LeRF(x, attr, ratio):
    original_size = x.size()
    x = x.reshape(3, -1)
    attr = torch.tensor(attr).flatten()
    v, index = torch.sort(attr, descending=True, dim=0)    
    index = index[-int(x.size(1)*ratio):]
    x[:, index] = 0.0 
    x = x.reshape(*original_size)
    return x 