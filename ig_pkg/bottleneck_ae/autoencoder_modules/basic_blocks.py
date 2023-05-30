import torch 
import torch.nn as nn 


def layer_init(layer, std=2**(1/2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class EncoderModule(nn.Module):
    def __init__(self, in_channels, hidden_dims):
        super().__init__()
        modules = [] 
        self.hidden_dims = hidden_dims
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    layer_init(nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1)),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
    def forward(self, x):
        x = self.encoder(x) 
        return x 

class DecoderModule(nn.Module):
    def __init__(self, out_channels, hidden_dims):
        super().__init__()
        modules = [] 

        self.hidden_dims = hidden_dims
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    layer_init(nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1)),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            layer_init(nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1)),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            layer_init(nn.Conv2d(hidden_dims[-1], out_channels=out_channels,
                                      kernel_size= 3, padding= 1)),
                            nn.Tanh())
    def forward(self, x):
        x = self.decoder(x) 
        x = self.final_layer(x) 
        return x
    
        

if __name__ == "__main__":
    import torchvision
    in_channels = 1
    out_channels = 1
    
    hidden_dims = [64,128,256]
    
    encoder = EncoderModule(in_channels, hidden_dims)
    decoder = DecoderModule(in_channels, hidden_dims)
    
    x = torch.randn(1,1, 32, 32)
    # x = torchvision.transforms.Resize(32)(x)
    x = encoder(x) 
    print(x.size())
    x = decoder(x) 
    print(x.size())