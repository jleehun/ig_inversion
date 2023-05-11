import os
import torch
import numpy as np
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as T
import torch.nn.functional as F

from ig_pkg.datasets import get_datasets

from ig_pkg.models.generator import get_model
from ig_pkg.models.classifier import get_classifier
from ig_pkg.models.pretrained_models import get_pretrained_model

from ig_pkg.inputattribs.ig import make_interpolation, ig
from ig_pkg.inputattribs.baseline_generator import get_baseline_generator

from ig_pkg.misc import process_heatmap, normalize_tensor, convert_to_img, label_to_class, tran, na_imshow

import time
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from ig_pkg.loss.focal_loss import FocalLoss
from ig_pkg.loss.metrics import ArcMarginProduct, AddMarginProduct

import torchvision.models as models
from torch.autograd import Variable

import saliency.core as saliency
import scipy.stats as stats

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

my_cmap=plt.cm.seismic(np.arange(plt.cm.seismic.N))

# for image

def get_rank(attribution, k):
    flat = attribution.flatten()
    num = int(attribution.size(-1) * k / 100)
    val, idx = torch.topk(flat, num)
    idx_list = [[x.item() // 224, x.item() % 224] for x in idx]
    return idx_list
    
def delete_attribution(tensor, attribution, k, device):
    idx = get_rank(attribution, k)
    print(len(idx))
    tensor = tensor.clone().detach()
    for i in idx:
        j, k = i
        tensor[:, j, k] = -1000 # convert to image: Normalize
    tensor = tensor.to(device)
    return tensor

def kendal_tau(tensor, attrib):    
    temp = tensor.mean(dim = 0)
    temp1 = temp.flatten().detach().cpu().numpy()
    temp2 = attri.flatten().detach().cpu().numpy()
    val, p_val = stats.kendalltau(temp1, temp2)
    return val
        
def pipeline(model, tensor, baseline, k, device, name):    
    model = model.to(device)
    logit_orig = model(tensor.unsqueeze(0))
    score_orig = nn.functional.softmax(logit_orig, dim = -1)
    init_pred = torch.argmax(logit_orig).item()
    prob_orig = score_orig[0, init_pred].item()
    attr = ig(model.to(device), tensor.to(device), init_pred, baseline, device)
    
    new_tensor = delete_attribution(tensor, attr, k, device)
    logit_new = model(new_tensor.unsqueeze(0))
    score_new = nn.functional.softmax(logit_new, dim = -1)
    prob_new = score_new[0, init_pred].item()
    
    metric_aopc = prob_orig - prob_new
    metric_lodds = np.log(prob_new / prob_orig)        
#     metric_kendall = 
    return metric_aopc, metric_lodds
    
    

    
