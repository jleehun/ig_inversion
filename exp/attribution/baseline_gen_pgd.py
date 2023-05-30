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
from ig_pkg.utils.adversarial import pgd_attack, untarget_fgsm, simple_untarget_fgsm, cw_l2_attack

parser =argparse.ArgumentParser()
parser.add_argument("--data-path",  required=True)
# parser.add_argument("--attr-path",  required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--device",  required=True)
parser.add_argument("--measure",  required=True)
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

# zero baseline
# baseline = torch.zeros_like(valid_dataset[0][0]).to(device)

pbar = tqdm(range(len(valid_dataset)))
pbar.set_description(f" Evaluation [👾] | generating attribution {args.measure} | ")

model = classifier.eval().to(args.device)

interpolation = []
attribution = []

if args.measure == "pgd": fn = pgd_attack
elif args.measure == "simple_fgsm": fn = simple_untarget_fgsm
elif args.measure == "fgsm": fn = untarget_fgsm
elif args.measure == "cw": fn = cw_l2_attack

for idx in pbar:
    input, label = valid_dataset[idx]
    input = input.to(args.device)

    adver = fn(model, input, label, args.device, eps=0.3, iters=24, alpha=2/255).to(args.device)
    
    attrib = integrated_gradient(model, input, label, adver[-1], adver, args.device) # tensor
    
    interpolation.append(adver.detach().cpu())    
    attribution.append(attrib.detach().cpu())
    
    if args.debug:
        if idx > 10:
            break

interpolation = torch.stack(interpolation)
attribution = torch.stack(attribution)

print('please')

np.save(f'/root/results/cifar10/image_{args.measure}_interpolation.npy', interpolation.numpy())
np.save(f'/root/results/cifar10/image_{args.measure}_attribution.npy', attribution.numpy())


# np.save(f'/home/dhlee/results/cifar10/image_{args.measure}_interpolation.npy', interpolation.numpy())
# np.save(f'/home/dhlee/results/cifar10/image_{args.measure}_attribution.npy', attribution.numpy())
# np.save(f'/home/dhlee/code/ig_inversion/results/cifar10/image_{args.measure}_interpolation.npy', interpolation.numpy())
# np.save(f'/home/dhlee/code/ig_inversion/results/cifar10/image_{args.measure}_attribution.npy', attribution.numpy())

# np.save('/home/dhlee/code/ig_inversion/results/cifar10/image_pgd_interpolation.npy', interpolation.numpy())
# np.save('/home/dhlee/code/ig_inversion/results/cifar10/image_pgd_attribution.npy', attribution.numpy())

print('finish')
     
