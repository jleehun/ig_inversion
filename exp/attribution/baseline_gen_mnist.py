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
# parser.add_argument("--model-path", required=True)
parser.add_argument("--type",  required=True, type=int)
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
MNIST_MEAN = [0.1307]
MNIST_STD  = [0.3081] 

transform = T.Compose([
                T.ToTensor(), 
                T.Normalize(MNIST_MEAN, MNIST_STD)
            ])

# valid_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, transform=transform)
train_data = torchvision.datasets.MNIST(root = args.data_path,
                            train=True,
                            transform=T.Compose([T.ToTensor(), T.Normalize(mean = MNIST_MEAN, std = MNIST_STD)]),
)

# call classifier

import torch 
from ebm_pkg.models import get_model
from ebm_pkg.datasets import CIFAR10_MEAN, CIFAR10_STD, MNIST_MEAN, MNIST_STD

path = f"/home/dhlee/code/paper_code_exercise/prev___ebm_classification/results/train/mnist/baseline/seed_0"

configs = {
    "cnn" : (10, 'relu' , 256, None), 
}
out_features, activation, cnn_dim, last_avg_kernel_size = configs['cnn']
model = get_model('cnn', 
                in_channels=1,
                out_features=out_features,
                activation=activation,
                cnn_dim=cnn_dim,
                dropout_p = 0.5,
                last_avg_kernel_size=last_avg_kernel_size)
model.load_state_dict(torch.load(f"{path}/model_best.pt", map_location='cpu'))


#  =======================================

# call baseline

baseline = torch.load('/data8/donghun/cifar10/tensor.pt', map_location='cpu')
temp = baseline[args.type][0]
temp = T.Resize(28)(temp.unsqueeze(0))
# temp = torch.zeros((1, 28, 28))
#==========================================


pbar = tqdm(range(len(train_data)))
pbar.set_description(f" Generating [👾] | generating attribution | ")

model = model.eval().to(args.device)

interpolation = []
attribution = []

for idx in pbar:
    
    baseline = temp.clone().detach().to(args.device)
    
    input, label = train_data[idx]
    input = input.to(args.device)
    interp = linear_interpolation(input, 24, baseline).to(args.device) # tensor
    attrib = integrated_gradient(model, input, label, baseline, interp, args.device) # tensor
    
    interpolation.append(interp.detach().cpu())
    attribution.append(attrib.detach().cpu())
    
    if args.debug:
        if idx > 10:
            break

interpolation = torch.stack(interpolation)
attribution = torch.stack(attribution)

print('please')

# np.save(f'/home/dhlee/code/ig_inversion/results/cifar10/image_flat_{args.type}_linear_interpolation.npy', interpolation.numpy())
# np.save(f'/home/dhlee/code/ig_inversion/results/cifar10/image_flat_{args.type}_linear_attribution.npy', attribution.numpy())

np.save(f'/home/dhlee/results/mnist/linear_{args.type}_interpolation.npy', interpolation.numpy())
np.save(f'/home/dhlee/results/mnist/linear_{args.type}_attribution.npy', attribution.numpy())

# np.save(f'/root/data/case/image_flat_{args.type}_linear_interpolation.npy', interpolation.numpy())
# np.save(f'/root/data/case/image_flat_{args.type}_linear_attribution.npy', attribution.numpy())

print('finish')

