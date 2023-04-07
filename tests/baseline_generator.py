from ig_pkg.inputattribs.baseline_generator import get_baseline_generator

import torch 

x = torch.rand(3,224,224)
y=1
 
name = 'zero'
b_generator = get_baseline_generator(name)
baseline = b_generator(x, y=y)
print(baseline.size(), baseline.min(), baseline.max(), baseline.mean())



name = 'optimizer'
decoder = None 
classifier = None 
b_generator = get_baseline_generator(name)
baseline = b_generator(x, y=y)
print(baseline.size(), baseline.min(), baseline.max(), baseline.mean())


