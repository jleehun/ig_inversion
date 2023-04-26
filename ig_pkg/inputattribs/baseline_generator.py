import cv2 as cv

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

from ig_pkg.inputattribs.ig import ig # error?
from ig_pkg.misc import convert_to_img, process_heatmap

# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_baseline_generator(name, **kwargs):
    cls = {
        'zero': ZeroBaselineGenerator,
        'one': OneBaselineGenerator,
        'min': MinBaselineGenerator,
        'max': MaxBaselineGenerator,
        'scalar' :ScalarBaselineGenerator,
        'encoder': EncoderInversionBaselineGenerator,
        'optimizer' : OptimizationInversionBaselineGenerator,
        'hybrid' :HybridInversionBaselineGenerator,
        'gaussian_blur': GaussianBlurredBaselineGenerator,
        'gaussian': GaussianBaselineGenerator,
        'uniform': UniformBaselineGenerator,
        'maximumdistance': MaximumDistanceBaselineGenerator,        
    }
    return cls[name](**kwargs)

class BaselineGenerator():
    def __init__(self, ):
        pass 

    def __call__(self, x, **kwargs):
        return self.generate_baseline(x, **kwargs)
    
    def generate_baseline(self, x, **kwargs):
        pass

class OneBaselineGenerator(BaselineGenerator):
    def __init__(self, **kwargs):
        super().__init__()
        pass 
    
    def generate_baseline(self, x, **kwargs):
        baseline = torch.ones_like(x).to(x.device)
        return baseline

class ZeroBaselineGenerator(BaselineGenerator):
    def __init__(self, **kwargs):
        super().__init__()
        pass 
    
    def generate_baseline(self, x, **kwargs):
        baseline = torch.zeros_like(x).to(x.device)
        return baseline
    
# Gaussian blur filter to the observation. 
# This method requires the assumption of adjacent features, which may not be the case of tabluar dataset,
# https://pytorch.org/vision/main/generated/torchvision.transforms.GaussianBlur.html
class GaussianBlurredBaselineGenerator(BaselineGenerator):
    def __init__(self, kernel_size = 13, sigma = 5, **kwargs):
        super().__init__()
#         self.kernel_size = (kernel_size, kernel_size)
#         self.sigma = (sigma, sigma)
        self.transform = T.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))
                
    def generate_baseline(self, x, **kwargs): 
        baseline = self.transform(x)
        return baseline

class GaussianBaselineGenerator(BaselineGenerator):
    def __init__(self, **kwargs):
        super().__init__()        
        pass 
    
    def generate_baseline(self, x, sigma = 1.0, **kwargs):
#         normal = torch.distributions.normal.Normal(loc = x.mean(), scale = x.std())
#         baseline = normal.sample(x.size())
        normal = torch.distributions.normal.Normal(loc = torch.tensor([0.0]), scale = torch.tensor([sigma]))
        noise = normal.sample(x.size())
        baseline = x + noise.squeeze(-1)
        return baseline
    
# The uniform distributions are defined in the valid range of the original features    
class UniformBaselineGenerator(BaselineGenerator):
    def __init__(self, **kwargs):
        super().__init__()
        pass 
    
    def generate_baseline(self, x, **kwargs):
        uniform = torch.distributions.uniform.Uniform(low = x.min(), high = x.max())
        baseline = uniform.sample(x.size())
        return baseline

class MaximumDistanceBaselineGenerator(BaselineGenerator):
    def __init__(self, **kwargs):
        super().__init__()
        pass 
    
    def generate_baseline(self, x, **kwargs):
        baseline = torch.round(x)
        return baseline

class MinBaselineGenerator(BaselineGenerator):
    def __init__(self, **kwargs):
        super().__init__()
        
        pass 
    
    def generate_baseline(self, x, **kwargs):
        baseline = torch.zeros_like(x).to(x.device)
        baseline.fill_(x.min().item())
        return baseline
    
class MaxBaselineGenerator(BaselineGenerator):
    def __init__(self, **kwargs):
        super().__init__()
        
        pass 
    
    def generate_baseline(self, x, **kwargs):
        baseline = torch.zeros_like(x).to(x.device)
        baseline.fill_(x.max().item())
        
        return baseline

    
class ScalarBaselineGenerator(BaselineGenerator):
    def __init__(self, scalar, **kwargs):
        super().__init__()
        
        self.scalar = scalar
    
    def generate_baseline(self, x, **kwargs):
        baseline = torch.zeros_like(x).to(x.device)
        baseline.fill_(self.scalar)
        return baseline
    
class EncoderInversionBaselineGenerator(BaselineGenerator):
    def __init__(self, encoder, decoder, classifier, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
            
    def generate_baseline(self, x, y, **kwargs):
        """
        given x, 
        z = encoder(x) 
        for loop         
            x' = decoder(z)
            y = classifier(x')
            z -> z - grad 
        baseline = decoder(z)
        """
        baseline = torch.zeros_like(x).to(x.device)
        return baseline

class OptimizationInversionBaselineGenerator(BaselineGenerator):
    def __init__(self, decoder, classifier, **kwargs):
        super().__init__()
        self.decoder = decoder
        self.classifier = classifier
            
    def generate_baseline(self, x, y, **kwargs):
        """
        given x, 
        z = random
        for loop until converge
            x' = decoder(z)
            z = z - grad L(x', x )
        for loop         
            x' = decoder(z)
            y = classifier(x')
            z -> z - grad 
        baseline = decoder(z)
        """
        baseline = torch.zeros_like(x).to(x.device)
    
        return baseline
        
class HybridInversionBaselineGenerator(BaselineGenerator):
    def __init__(self, encoder, decoder, classifier, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
            
    def generate_baseline(self, x, y, **kwargs):
        """
        given x, 
        z = encoder(x) 
        for loop until converge
            x' = decoder(z)
            z = z - grad L(x', x )
        for loop         
            x' = decoder(z)
            y = classifier(x')
            z -> z - grad 
        baseline = decoder(z)
        """
        baseline = torch.zeros_like(x).to(x.device)
    
        return baseline        
            


if __name__ == "__main__":
    x = torch.rand(3, 224, 224)
    y = 1
    print(x.size(), x.min(), x.max(), x.mean())
    fig, axes = plt.subplots(1, 6, figsize = (5, 5))
    axes = axes.flat
    ax = next(axes)
    img = convert_to_img(x)
    ax.imshow(img)
    
    for baseline_name in [ 'zero', 'gaussian_blur', 'gaussian', 'uniform', 'maximumdistance']:
#     for baseline_name in ['zero', 'min', 'max', 'scalar', 'encoder', 'optimizer', 'hybrid']:
        encoder = None
        decoder = None
        classifier = None
        sigma = 1
        b_generator = get_baseline_generator(baseline_name, encoder=encoder, decoder=decoder, classifier=classifier, scalar=3, sigma = 1)
        baseline = b_generator(x, y=y)
        ax = next(axes)
        ax.imshow(convert_to_img(baseline))        
        print(baseline_name)
        print(baseline.size(), baseline.min(), baseline.max(), baseline.mean())
    plt.savefig('fig.png')