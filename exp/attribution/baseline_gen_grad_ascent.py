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
parser.add_argument("--device",  required=True)
# parser.add_argument("--attr-path",  required=True)
parser.add_argument("--model-path", required=True)
# parser.add_argument("--measure",  required=True)
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

torch.backends.cudnn.enabled = False

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

model = classifier.eval().to(args.device)

interpolation = []
attribution = []


for idx in pbar:
    input, label = valid_dataset[idx]
    input = input.to(args.device)
    interp = image_gradient_ascent_interpolation(model, input, label, 24, args.device).to(args.device)
    # interp = image_gradient_interpolation(model, input, label, 10, args.device).to(args.device)
    baseline = interp[-1]
    attrib = integrated_gradient(model, input, label, baseline, interp, args.device) # tensor
    
    interpolation.append(interp.detach().cpu())
    attribution.append(attrib.detach().cpu())
    
    if args.debug:
        if idx > 10:
            break

interpolation = torch.stack(interpolation)
attribution = torch.stack(attribution)

np.save('/home/dhlee/results/cifar10/image_gradient_ascent_interpolation.npy', interpolation.numpy())
np.save('/home/dhlee/results/cifar10/image_gradient_ascent_attribution.npy', attribution.numpy())


#--------------------
# np.save('/home/dhlee/results/cifar10/image_gradient_interpolation.npy', interpolation.numpy())
# np.save('/home/dhlee/results/cifar10/image_gradient_attribution.npy', attribution.numpy())
# np.save('/home/dhlee/code/ig_inversion/results/cifar10/image_gradient_interpolation.npy', interpolation.numpy())
# np.save('/home/dhlee/code/ig_inversion/results/cifar10/image_gradient_attribution.npy', attribution.numpy())



    
