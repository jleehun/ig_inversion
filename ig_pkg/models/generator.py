import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from ig_pkg.models.stylegan_generator import *
from ig_pkg.models.stylegan2_generator import *
from ig_pkg.models.pggan_generator import *

from ig_pkg.models.utils import make_find_z_fun

def get_model(name_model, path, res, nz = 512):
    if name_model == 'stylegan2':
        G = stylegan2(path, res)

    elif name_model == 'stylegan':
        G = stylegan(path, res)

    elif name_model == 'pggan':
        G = pggan(path, res)
    
    else: raise ValueError("no model")
    
    return G

class Gen_wrapper():
    def __init__(self, name_model, path, res, nz = 512):        
        if name_model == 'stylegan2':
            G = stylegan2(path, res)
            
        elif name_model == 'stylegan':
            G = stylegan(path, res)
            
        elif name_model == 'pggan':
            G = pggan(path, res)
                        
        self.gen = G
        self.nz = nz
        
        self.find_z = make_find_z_fun()
        
    def decode(self, z): # image
        return self.gen(z) 
    
    def sample(self, batchsize):
        return torch.randn(batchsize, self.nz) 
    
    def encode(self, x):
        z = self.sample(x.size(0)).to(x.device)        
        return self.find_z(self.gen, z, x)




def stylegan2(path, res):

	G = StyleGAN2Generator(resolution = res)
	
	weight = torch.load(path)

	G.load_state_dict(weight['generator_smooth'])
	G.eval()

	return G

def stylegan(path, res):

	G = StyleGANGenerator(resolution = res)
	
	weight = torch.load(path)

	G.load_state_dict(weight)
	
	G.eval()
	
	return G

def pggan(path, res):

	G = PGGANGenerator(resolution = res)
	
	weight = torch.load(path)

	G.load_state_dict(weight['generator_smooth'])
	
	G.eval()
	
	return G
    
    
# def make_find_z_fun(loss_fun = torch.nn.MSELoss(),
#                     max_steps: int = 5000,                    
#                     lr: float = 0.1,
#                     diff: float = 1e-3,
#                     ):
#     """
#     initializes a function with which one can find the GAN latent representations for a given image
#     :param max_steps: maximum number of iterations during optimization
#     :param lr: learning rate
#     :param diff: early stopping loss between image
#     :param loss_fun: loss function to optimize during search process
#     :return: function
#     """

#     def find_z(generator, z, img):
#         """
#         find the latent representation of an image, so that when decoding the latent representation decoded=GAN(latent)
#         the decoded image is very close to the source image
#         :param generator: GAN model
#         :param z: initial latent representation (random)
#         :param img: image
#         :return: latent representation that corresponds to image
#         """
#         z = z.clone()
#         z.requires_grad = True
#         optimizer = optim.Adam([z], lr=lr)

#         print("Optimizing latent representation ...")
#         # optimizer.zero_grad()

#         with tqdm(total=max_steps) as progress_bar:
#             for step in range(max_steps):
#                 optimizer.zero_grad()
#                 x = generator.forward(z)
#                 x = torch.clip(x, min=0, max=1)

#                 loss = loss_fun(x, img)

#                 progress_bar.set_postfix(loss=loss.item(), step=step + 1)
#                 progress_bar.update()

#                 if loss < diff:
#                     break
#                 # optimizer.zero_grad()
#                 loss.backward()

#                 optimizer.step()

#         return z.detach()

#     return find_z