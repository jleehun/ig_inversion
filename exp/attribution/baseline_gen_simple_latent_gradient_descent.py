import random
import os
import torch
import torchvision
import numpy as np 
import argparse
from tqdm import tqdm
from distutils.util import strtobool

from ig_pkg.utils.eval import Cifar10Evaluator
from ig_pkg.utils.metrics import * #morf, lerf, 
from ig_pkg.utils.attribution import *
from ig_pkg.utils.autoencoder_examiner import AutoencoderExaminer

parser =argparse.ArgumentParser()
parser.add_argument("--data-path",  required=True)
# parser.add_argument("--attr-path",  required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--classifier-path", required=True)
# parser.add_argument("--measure",  required=True)
parser.add_argument("--device", required=True)
parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,)

# -----------------------------

args = parser.parse_args()
# print(args)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    
seed_everything(42)

# device = 'cuda:5'

# call dataset, dataloader
import torchvision.transforms as T
CIFAR10_STATS = {
    'mean' : [0.4914, 0.4822, 0.4465],
    'std' : [0.2023, 0.1994, 0.2010]
}

transform = T.Compose([
                T.ToTensor(), 
                T.Normalize(CIFAR10_STATS['mean'], CIFAR10_STATS['std'])
            ])

valid_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, transform=transform)

classifier = torch.load(args.model_path,  map_location='cpu')
ae = torch.load(args.classifier_path, map_location='cpu')
# zero baseline
# baseline = torch.zeros_like(valid_dataset[0][0]).to(device)

pbar = tqdm(range(len(valid_dataset)))
pbar.set_description(f" Generation [👾] | generating attribution | ")

ae.to(args.device)
model = classifier.eval().to(args.device)

interpolation_z = []
interpolation = []
attribution = []

for idx in pbar:
    input, label = valid_dataset[idx]
    input = input.to(args.device)
    

    x_hat, loss_dict, info = AutoencoderExaminer.reconstruct_input(ae, input.unsqueeze(0))    
    z = info['bottleneck']
    
    interp = [input]
    interp_z = [z]
        
    grad = AutoencoderExaminer.get_classifier_latent_direction(ae, model, z, label)
    
    for _ in range(24):        
        z -= grad * 100
        input_hat = AutoencoderExaminer.reconstruct_latent(ae, z)
        
        interp_z.append(z)
        interp.append(input_hat.squeeze(0))
        
    baseline = input_hat.squeeze(0)

    interp_z = torch.stack(interp_z) # 11, 1, 512, 2, 2
    interp = torch.stack(interp) # 11, 3, 32, 32
    attr = integrated_gradient(model, input, label, baseline, interp, args.device) # 32, 32

    interpolation_z.append(interp_z.detach().cpu())
    interpolation.append(interp.detach().cpu())    
    attribution.append(attr.detach().cpu())
    
    if args.debug:
        if idx > 10:
            break

interpolation_z = torch.stack(interpolation_z)
interpolation = torch.stack(interpolation)
attribution = torch.stack(attribution)

print('please')

np.save('/home/dhlee/results/cifar10/latent_simple_gradient_descent_interpolation_z.npy', interpolation_z.numpy())
np.save('/home/dhlee/results/cifar10/latent_simple_gradient_descent_interpolation.npy', interpolation.numpy())
np.save('/home/dhlee/results/cifar10/latent_simple_gradient_descent_attribution.npy', attribution.numpy())

# np.save('/home/dhlee/results/cifar10/latent_linear_interpolation_z.npy', interpolation_z.numpy())
# np.save('/home/dhlee/results/cifar10/latent_linear_interpolation.npy', interpolation.numpy())
# np.save('/home/dhlee/results/cifar10/latent_linear_attribution.npy', attribution.numpy())
# np.save('/home/dhlee/code/ig_inversion/results/cifar10/latent_linear_interpolation_z.npy', interpolation_z.numpy())
# np.save('/home/dhlee/code/ig_inversion/results/cifar10/latent_linear_interpolation.npy', interpolation.numpy())
# np.save('/home/dhlee/code/ig_inversion/results/cifar10/latent_linear_attribution.npy', attribution.numpy())

print('finish')

