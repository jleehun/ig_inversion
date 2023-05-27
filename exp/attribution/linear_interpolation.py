import random
import os
import torch
import torchvision
import numpy as np 
import argparse
from tqdm import tqdm
import json
from distutils.util import strtobool

from ig_pkg.utils.eval import Cifar10Evaluator
from ig_pkg.utils.metrics import * #morf, lerf, 
from ig_pkg.utils.attribution import *

parser =argparse.ArgumentParser()
parser.add_argument("--data-path",  required=True)
# parser.add_argument("--attr-path",  required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--save-dir",  required=True)
# parser.add_argument("--measure",  required=True)
parser.add_argument("--method",  required=True)
parser.add_argument("--device",  required=True)
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

pbar = tqdm(range(len(valid_dataset)))
pbar.set_description(f" Attribtion [👾] | linear & gradient | ")

model = classifier.eval().to(args.device)

interpolation = []
attribution = []

interp_path  = {
    # 'zero': '/home/dhlee/results/cifar10/image_linear_zero_interpolation.npy',
    # 'expected': '/home/dhlee/results/cifar10/image_linear_expected_interpolation.npy',
    
    # 'latent_linear': '/home/dhlee/results/cifar10/latent_linear_interpolation.npy',
    
    'image_gradient_descent': '/home/dhlee/results/cifar10/image_gradient_descent_interpolation.npy', # descent
    'image_gradient_ascent': '/home/dhlee/results/cifar10/image_gradient_ascent_interpolation.npy',            
    'latent_gradient_descent': '/home/dhlee/results/cifar10/latent_gradient_descent_interpolation.npy',
    'latent_gradient_ascent': '/home/dhlee/results/cifar10/latent_gradient_ascent_interpolation.npy',
    
    'image_simple_gradient_descent': '/home/dhlee/results/cifar10/image_simple_gradient_descent_interpolation.npy', # descent
    'image_simple_gradient_ascent': '/home/dhlee/results/cifar10/image_simple_gradient_ascent_interpolation.npy',            
    'latent_simple_gradient_descent': '/home/dhlee/results/cifar10/latent_simple_gradient_descent_interpolation.npy',
    'latent_simple_gradient_ascent': '/home/dhlee/results/cifar10/latent_simple_gradient_ascent_interpolation.npy',
    
    'image_simple_fgsm': '/home/dhlee/results/cifar10/image_simple_fgsm_interpolation.npy',
    'image_fgsm': '/home/dhlee/results/cifar10/image_fgsm_interpolation.npy',
    'image_pgd': '/home/dhlee/results/cifar10/image_pgd_interpolation.npy',
    'image_cw': '/home/dhlee/results/cifar10/image_cw_interpolation.npy',
}[args.method]

interpolation = torch.from_numpy(np.load(interp_path))

if os.path.exists(os.path.join(args.save_dir, 'baseline_score_samples_cifar10.json')):
    sample_result_dict = json.load(open(os.path.join(args.save_dir, 'baseline_score_samples_cifar10.json') ,"r"))
else:
    sample_result_dict = {}
if os.path.exists(os.path.join(args.save_dir, 'baseline_score_average_cifar10.json')):
    average_result_dict = json.load(open(os.path.join(args.save_dir, 'baseline_score_average_cifar10.json') ,"r"))
else:
    average_result_dict = {}

sample_result_dict[f'{args.method}'] = []
average_result_dict[f'{args.method}'] = []

for idx in pbar:    
    input, label = valid_dataset[idx]
    input = input.to(args.device)
    
    baseline = interpolation[idx][-1].to(args.device)
    logit = model(baseline.unsqueeze(0).to(args.device))
    score = torch.softmax(logit, dim=-1)[:, label].item()
    sample_result_dict[f'{args.method}'].append(score)
    
    interp = linear_interpolation(input, 24, baseline).to(args.device) # tensor
    attrib = integrated_gradient(model, input, label, baseline, interp, args.device) # tensor
    
    attribution.append(attrib.detach().cpu())
    
    if args.debug:
        if idx > 10:
            break

attribution = torch.stack(attribution)

print('please')

np.save(f'/home/dhlee/results/cifar10/{args.method}_interpolation_linear_attribution.npy', attribution.numpy())

print('finish')

average_result_dict[f'{args.method}'] = np.mean(sample_result_dict[f'{args.method}'])

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
with open(os.path.join(args.save_dir, f"baseline_score_average_cifar10.json"), 'w') as f:
    json.dump(average_result_dict, f, indent=2, sort_keys=True)
    
with open(os.path.join(args.save_dir, f"baseline_score_samples_cifar10.json"), 'w') as f:
    json.dump(sample_result_dict, f, indent=2, sort_keys=True)  
