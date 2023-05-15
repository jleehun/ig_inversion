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
from ig_pkg.metrics import pipeline, kendal_correlation

import time
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from ig_pkg.loss.focal_loss import FocalLoss
from ig_pkg.loss.metrics import ArcMarginProduct, AddMarginProduct

import torchvision.models as models
from torch.autograd import Variable

import scipy.stats as stats

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

my_cmap=plt.cm.seismic(np.arange(plt.cm.seismic.N))

# ================================
# def write_npz(samples, path):
#     if samples.ndim == 1:
#         np.save(path, samples)
#     else:
#         np.savez(path, **{str(key): value for key, value in enumerate(samples)}) # 배열 크기만큼 ['0', '1', '2' . ..]로 key값이 저장됨
        
def test():
    model = models.resnet18(pretrained=False)
    eval_mode = model.eval()

    data_path="/root/data/ILSVRC2012_val/"

    _, valid_datasets = get_datasets("imagenet1k", data_path)
    valid_dataloader = torch.utils.data.DataLoader(valid_datasets, batch_size=1, shuffle=False, num_workers=2)

    device = 'cuda:0'

    baseline = torch.zeros((3, 224, 224))
    attribution = []

    for i, (inputs, labels) in enumerate(valid_dataloader):
        attr = ig(eval_mode.to(device), inputs.squeeze(0), labels.item(), baseline, device=device)
        
        attribution.append(attr.detach().cpu())

    output = torch.stack(attribution, dim = 0)
    np.save('/root/data/results/ig_inversion/ig_zero_baseline.npy', output)
    
#     return output

if __name__ == "__main__":
    print('start')
    test()
    print('fin')
