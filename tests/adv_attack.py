from ig_pkg.models.classifier import BaseClassifier 
from ig_pkg.models.generator import BaseGenerator 
from ig_pkg.algorithm.adv_attack import run_adv_attack 
import torch 
from torch.optim import Adam

x = torch.rand(1,3,32,32)
attack_style='z'
g_model = BaseGenerator()

# define parameters that will be optimized
params = []
if attack_style == "z":
    # define z as params for derivative wrt to z
    z = g_model.encode(x)
    z = [z_i.detach() for z_i in z] if isinstance(z, list) else z.detach()
    x_org = x.detach().clone()
    z_org = [z_i.clone() for z_i in z] if isinstance(z, list) else z.clone()

    if type(z) == list:
        for z_part in z:
            z_part.requires_grad = True
            params.append(z_part)
    else:
        z.requires_grad = True
        params.append(z)
else:
    # define x as params for derivative wrt x
    x_org = x.clone()
    x.requires_grad = True
    params.append(x)
    z = None

classifier = BaseClassifier()
target_class = 0 
save_at=10
num_steps=100
maximize = True 
optimizer = Adam()


run_adv_attack(x,
               z,
                optimizer, 
                classifier,
                g_model, 
                target_class,
                attack_style,
                save_at,
                num_steps,
                maximize)