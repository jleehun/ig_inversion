import torch 
import torch.nn as nn 
import torchvision 
from ig_pkg.bottleneck_ae.autoencoder_modules.ae import AutoEncoder
from ig_pkg.bottleneck_ae.autoencoder_modules.latents import ReductionBottleNeck
from ig_pkg.bottleneck_ae.autoencoder_modules.basic_blocks import EncoderModule, DecoderModule 

class BottleneckAE(AutoEncoder):
    def __init__(self, in_size, hidden_dim, **kwargs):
        
        hidden_dims = [32, 64, 128, 256]
        encoder = EncoderModule(in_size[0], hidden_dims)
        decoder = DecoderModule(in_size[0], hidden_dims[::-1])
        bottleneck = ReductionBottleNeck((256, 2,2), hidden_dim)
        
        super().__init__(encoder, decoder, bottleneck, in_size)
    
    
    