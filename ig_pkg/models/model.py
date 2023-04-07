import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stylegan_generator import *
from .stylegan2_generator import *
from .pggan_generator import *

# from models.stylegan_generator import *
# from models.stylegan2_generator import *
# from models.pggan_generator import *

# from models.stylegan2_discriminator import *
# from models.pggan_discriminator import *

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