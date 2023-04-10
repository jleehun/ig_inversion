import torch
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.colors import ListedColormap

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def convert_to_img(tensor, means = IMAGENET_MEAN, stds = IMAGENET_STD):
    if tensor.device == 'cpu':
        pass
    else: tensor = tensor.detach().cpu()
    
    means = torch.tensor(means).view(len(means), 1,1)
    stds = torch.tensor(stds).view(len(means), 1,1)
    img = (tensor * stds) + means
    img = img.permute(1,2,0).numpy()
    img = img*255
    img = img.astype(int)
    img = img.clip(0,255)
    return img 

def process_heatmap(R, my_cmap=plt.cm.seismic(np.arange(plt.cm.seismic.N))):
    power = 1.0
    b = 10*((np.abs(R)**power).mean()**(1.0/power))
    my_cmap[:,0:3] *= 0.99
    my_cmap = ListedColormap(my_cmap)
    return (R, {"cmap":my_cmap, "vmin":-b, "vmax":b, "interpolation":'nearest'} )