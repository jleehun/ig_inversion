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

parser =argparse.ArgumentParser()
parser.add_argument("--data-path",  required=True)
# parser.add_argument("--attr-path",  required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--num",  required=True)
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

device = 'cuda:0'

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

# zero baseline
# baseline = torch.zeros_like(valid_dataset[0][0]).to(device)

pbar = tqdm(range(len(valid_dataset)))
pbar.set_description(f" Evaluation [👾] | generating attribution zero | ")

model = classifier.eval().to(device)

interpolation = []
attribution = []

for idx in pbar:
    ran = np.random.randint(len(valid_dataset))
    if idx == ran:
        ran = np.random.randint(len(valid_dataset))
        if idx == ran:
            ran = np.random.randint(len(valid_dataset))
    
    baseline = valid_dataset[ran][0].to(device)
    
    input, label = valid_dataset[idx]
    input = input.to(device)
    interp = linear_interpolation(input, 24, baseline).to(device) # tensor
    attrib = integrated_gradient(model, input, label, baseline, interp, device='cuda:5') # tensor
    
    interpolation.append(interp.detach().cpu())
    attribution.append(attrib.detach().cpu())
    
    if args.debug:
        if idx > 10:
            break

interpolation = torch.stack(interpolation)
attribution = torch.stack(attribution)

print('please')

np.save('/root/data/case/image_flat_linear_interpolation.npy', interpolation.numpy())
np.save('/root/data/case/image_flat_linear_attribution.npy', attribution.numpy())

# np.save('/home/dhlee/results/cifar10/image_linear_expected_interpolation.npy', interpolation.numpy())
# np.save('/home/dhlee/results/cifar10/image_linear_expected_attribution.npy', attribution.numpy())

print('finish')

