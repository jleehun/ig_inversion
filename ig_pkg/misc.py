from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
import torch.nn.functional as F
import numpy as np

# for mapping layer to resolution
stylegan2 = [0,1,None,2,None,3]
pggan = [None, 0, None, 1, None, 2, None, 3]
shape = [(512, 4, 4), (512, 8, 8), (512, 16, 16), (512, 32, 32)]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CIFAR100_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR100_STD  = [0.2023, 0.1994, 0.2010] 

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2023, 0.1994, 0.2010]

MNIST_MEAN = [0.1307]
MNIST_STD  = [0.3081] 

def minmax(image):
	return (image - image.min())/(image.max() - image.min())


def image_mask(layer_idx, act_idx, resolution, model):

	shape_idx = stylegan2 if model == 'stylegan2' else pggan
	region_mask = [torch.zeros(shape[shape_idx[l]][0] * shape[shape_idx[l]][1] * shape[shape_idx[l]][2]) for l in layer_idx]

	image_mask = []
	for i in range(len(layer_idx)):
	    region_mask[i][act_idx[i]] = 1
	    temp = region_mask[i].view(shape[i]).mean(dim = 0, keepdim = True).unsqueeze(0)
	    image_mask.append(F.upsample(temp, size = (resolution, resolution), mode = 'bilinear').squeeze())

	image_mask_sum = torch.stack(image_mask).mean(dim = 0)

	return image_mask_sum



def print_two_images(image1, image2, mask, labels, figsize = (18, 5)):

	gs = gridspec.GridSpec(1, 3, wspace = 0.0, hspace = 0.0)

	plt.figure(figsize = figsize)
	plt.tight_layout()

	plt.subplot(gs[0,0])
	plt.axis('off')
	plt.imshow(minmax(image1[0].detach().cpu().permute(1,2,0)))
	plt.title(labels[0])

	plt.subplot(gs[0,1])
	plt.axis('off')
	plt.imshow(minmax(image2[0].detach().cpu().permute(1,2,0)))
	plt.title(labels[1])

	plt.subplot(gs[0,2])
	plt.imshow(minmax(image1[0]).detach().cpu().permute(1,2,0))
	plt.imshow(mask, cmap = 'RdBu_r', vmin = 0.03, vmax = 0.14,alpha = 0.8)
	plt.axis('off')
	plt.title('mask')
	plt.colorbar()

	plt.show()


def print_images(images, title, sample_num = 60):

	assert sample_num % 10 == 0, "sample_num % 10 == 0"

	row = sample_num // 10
	col = 10

	gs = gridspec.GridSpec(row, col, wspace = 0., hspace = 0.0)
	plt.figure(figsize = (col * 4.87, row*5))
	plt.tight_layout()

	for i in range(row):
		for j in range(col):
			plt.subplot(gs[i, j])
			plt.imshow(minmax(images[i*10 + j]))
			plt.axis('off')
	plt.suptitle(title, fontsize=50)

	plt.show()

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
    if R.device == 'cpu':
        R = R.cpu()
    else: R = R.detach().cpu().numpy()
    power = 1.0
    b = 10*((np.abs(R)**power).mean()**(1.0/power))
    new_cmap = my_cmap[:,0:3] * 0.99    
    new_cmap = ListedColormap(new_cmap)
    return (R, {"cmap":new_cmap, "vmin":-b, "vmax":b, "interpolation":'nearest'} )

        
