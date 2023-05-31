import torch
import torch.nn as nn 
import torchvision 

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, bottleneck, in_size, **kwargs) -> None:
        super().__init__()
        assert len(in_size) == 3 
        self.in_size = in_size
        self.resize_fn = torchvision.transforms.Resize(in_size[1:])
        
        self.encoder = encoder
        self.decoder= decoder
        self.bottleneck = bottleneck
        self.mse_loss = torch.nn.MSELoss()
        
        self.in_size = tuple(in_size)
        self.out_size = None
        self.bottleneck_size = None
        self.bottleneck_in_size = None
        self.bottleneck_out_size = None
        
        self._set_in_and_out_sizes()

            
    def compute_loss_dict(self, x, x_hat):
        # compute reconstruction loss 
        mse_loss = self.mse_loss(x_hat, x)
        return {
            "reconstruction_loss" : mse_loss,
            "total_loss" : mse_loss
        }
        
    def forward(self, x, y=None):
        x = self.resize_fn(x)
        z, decocder_input = self.encode(x)
        x_hat = self.decode(decocder_input)
        loss_dict = self.compute_loss_dict(x, x_hat)
        return x_hat, loss_dict, {'bottleneck' : z}
    
    def encode(self, x):
        # project and encode witht the latent 
        x = self.encoder(x)
        z = self.bottleneck.encode(x)
        return z # return tuple
    
    def decode(self, z):
        x_hat = self.bottleneck.decode(z)
        x_hat = self.decoder(x_hat)
        return x_hat
    
    
    # @torch.jit.script
    def sample(self, num_samples=1, sigma=1.0):
        x = torch.randn(num_samples, *self.bottleneck_size) 
        x = x * sigma
        return x 
    
    def _set_in_and_out_sizes(self):
        with torch.no_grad():
            x = torch.randn( 1, *self.in_size)
            x = self.encoder(x)
            self.bottleneck_in_size =  tuple(x.shape[1:])
            z, decoder_input = self.bottleneck.encode(x)
            self.bottleneck_size =  tuple(z.shape[1:])
            x_hat = self.bottleneck.decode(decoder_input)
            self.bottleneck_out_size = tuple(x_hat.shape[1:])
            x_hat = self.decoder(x)
            self.out_size = tuple(x_hat.shape[1:])
        print(f"[INFO] {self.__class__.__name__}", "{0:>15}".format("-->in-size     :"), self.in_size)
        print(f"[INFO] {self.__class__.__name__}", "{0:>15}".format("---->b-in-size :"), self.bottleneck_in_size)
        print(f"[INFO] {self.__class__.__name__}", "{0:>15}".format("-------->b-size:"), self.bottleneck_size)
        print(f"[INFO] {self.__class__.__name__}", "{0:>15}".format("--->b-out-size :"), self.bottleneck_out_size)
        print(f"[INFO] {self.__class__.__name__}", "{0:>15}".format("-->out-size    :"), self.out_size)
        
