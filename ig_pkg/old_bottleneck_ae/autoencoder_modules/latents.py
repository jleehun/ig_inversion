import torch 
import torch.nn as nn 


def layer_init(layer, std=2**(1/2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class IdentityBottleNeck(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(x)
        return x_hat    
    
    def encode(self, x):
        return x,
    
    def decode(self, z):
        return z
    
    
class ReductionBottleNeck(nn.Module):
    def __init__(self, in_size, hidden_dim, **kwargs):
        super().__init__()
        self.in_size = in_size
        self.proj_dim = torch.prod(torch.tensor(self.in_size)).item()
        self.fc1 = nn.Linear(self.proj_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.proj_dim)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        z, x = self.encode(x)
        x = self.decode(x)
        return  z, x
    
    def encode(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x) 
        z = self.tanh(x) 
        return z, x
    
    def decode(self, z):
        x = self.fc2(z)
        x = x.reshape(x.size(0), *self.in_size)
        return x

class NormalBottleNeck(nn.Module):
    def __init__(self, in_size, hidden_size, latent_size):
        super().__init__()
        self.in_size = in_size
        self.fc = nn.Sequential(
                            layer_init(nn.Linear(in_features=torch.prod(torch.tensor(self.in_size)).item(), out_features=hidden_size)),
                            nn.LeakyReLU(),
                            layer_init(nn.Linear(in_features=hidden_size, out_features=hidden_size)),
                            nn.LeakyReLU(),
                            )
        
        self.fc_mu     = layer_init(nn.Linear(hidden_size, latent_size))
        self.fc_logvar = layer_init(nn.Linear(hidden_size, latent_size))
        self.decode_latent = nn.Sequential(
                                    layer_init(nn.Linear(latent_size, hidden_size)), 
                                    nn.LeakyReLU(),
                                    layer_init(nn.Linear(in_features=hidden_size, out_features=torch.prod(torch.tensor(self.in_size)).item())),
                                    nn.LeakyReLU(),
                            )

    def encode(self, x):
        assert x.shape[1:] == self.in_size, f"x:{x.size()} in_size:{self.in_size}"
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        
        mu     = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar
    
    def decode(self, z):
        x_hat = self.decode_latent(z)
        x_hat = x_hat.reshape(-1, *self.in_size)
        return x_hat         

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z , mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = std * eps + mu
        return z
    
if __name__ == "__main__":
    bottleneck = NormalBottleNeck((512,7,7), 128, 64)
    input = torch.randn(1, 512, 7,7)
    print(bottleneck.encode(input)[0].size())
    z, mu, logvar = bottleneck.encode(input)
    print(bottleneck.decode(z).size())
    print(bottleneck(input)[0].size())
    
    bottleneck = IdentityBottleNeck()
    print(bottleneck.encode(input).size())
    print(bottleneck.decode(input).size())
    