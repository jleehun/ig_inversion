import torch
from ig_pkg.inputattribs.ig import ig


def get_baseline_generator(name, **kwargs):
    cls = {
        'zero': ZeroBaselineGenerator,
        'min': MinBaselineGenerator,
        'max': MaxBaselineGenerator,
        'scalar' :ScalarBaselineGenerator,
        'encoder': EncoderInversionBaselineGenerator,
        'optimizer' : OptimizationInversionBaselineGenerator,
        'hybrid' :HybridInversionBaselineGenerator,
    }
    return cls[name](**kwargs)

class BaselineGenerator():
    def __init__(self, ):
        pass 

    def __call__(self, x, **kwargs):
        return self.generate_baseline(x, **kwargs)
    
    def generate_baseline(self, x, **kwargs):
        pass
    
class ZeroBaselineGenerator(BaselineGenerator):
    def __init__(self, **kwargs):
        super().__init__()
        pass 
    
    def generate_baseline(self, x, **kwargs):
        baseline = torch.zeros_like(x).to(x.device)
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
    for baseline_name in ['zero', 'min', 'max', 'scalar', 'encoder', 'optimizer', 'hybrid']:
        encoder = None
        decoder = None
        classifier = None
        b_generator = get_baseline_generator(baseline_name, encoder=encoder, decoder=decoder, classifier=classifier, scalar=3)
        baseline = b_generator(x, y=y)
        print(baseline.size(), baseline.min(), baseline.max(), baseline.mean())
        