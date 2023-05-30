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
parser.add_argument("--method", required=True)
# parser.add_argument("--attr-path",  required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--device", required=True)
parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,)

# -----------------------------
parser.add_argument("--ratio",  type=float)
# ------------------------
args = parser.parse_args()

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    
seed_everything(42)

# device = args.device
# print(device)

attr_path  = {
    'zero': '/home/dhlee/results/cifar10/image_linear_zero_attribution.npy',
    'expected': '/home/dhlee/results/cifar10/image_linear_expected_attribution.npy',
    
    'latent_linear': '/home/dhlee/results/cifar10/latent_linear_attribution.npy',    
    'image_gradient_descent': '/home/dhlee/results/cifar10/image_gradient_descent_attribution.npy', # descent
    'image_gradient_ascent': '/home/dhlee/results/cifar10/image_gradient_ascent_attribution.npy',            
    'latent_gradient_descent': '/home/dhlee/results/cifar10/latent_gradient_descent_attribution.npy',
    'latent_gradient_ascent': '/home/dhlee/results/cifar10/latent_gradient_ascent_attribution.npy',
    
    'image_simple_gradient_descent': '/home/dhlee/results/cifar10/image_simple_gradient_descent_attribution.npy', # descent
    'image_simple_gradient_ascent': '/home/dhlee/results/cifar10/image_simple_gradient_ascent_attribution.npy',            
    'latent_simple_gradient_descent': '/home/dhlee/results/cifar10/latent_simple_gradient_descent_attribution.npy',
    'latent_simple_gradient_ascent': '/home/dhlee/results/cifar10/latent_simple_gradient_ascent_attribution.npy',
    
    'image_simple_fgsm': '/home/dhlee/results/cifar10/image_simple_fgsm_attribution.npy',
    'image_fgsm': '/home/dhlee/results/cifar10/image_fgsm_attribution.npy',
    'image_pgd': '/home/dhlee/results/cifar10/image_pgd_attribution.npy',
    'image_cw': '/home/dhlee/results/cifar10/image_cw_attribution.npy',

    'linear_latent_linear': '/home/dhlee/results/cifar10/latent_linear_interpolation_linear_attribution.npy',    
    'linear_image_gradient_descent': '/home/dhlee/results/cifar10/image_gradient_descent_interpolation_linear_attribution.npy', # descent
    'linear_image_gradient_ascent': '/home/dhlee/results/cifar10/image_gradient_ascent_interpolation_linear_attribution.npy',            
    'linear_latent_gradient_descent': '/home/dhlee/results/cifar10/latent_gradient_descent_interpolation_linear_attribution.npy',
    'linear_latent_gradient_ascent': '/home/dhlee/results/cifar10/latent_gradient_ascent_interpolation_linear_attribution.npy',
    
    'linear_image_simple_gradient_descent': '/home/dhlee/results/cifar10/image_simple_gradient_descent_interpolation_linear_attribution.npy', # descent
    'linear_image_simple_gradient_ascent': '/home/dhlee/results/cifar10/image_simple_gradient_ascent_interpolation_linear_attribution.npy',            
    'linear_latent_simple_gradient_descent': '/home/dhlee/results/cifar10/latent_simple_gradient_descent_interpolation_linear_attribution.npy',
    'linear_latent_simple_gradient_ascent': '/home/dhlee/results/cifar10/latent_simple_gradient_ascent_interpolation_linear_attribution.npy',
    
    'linear_image_simple_fgsm': '/home/dhlee/results/cifar10/image_simple_fgsm_interpolation_linear_attribution.npy',
    'linear_image_fgsm': '/home/dhlee/results/cifar10/image_fgsm_interpolation_linear_attribution.npy',
    'linear_image_pgd': '/home/dhlee/results/cifar10/image_pgd_interpolation_linear_attribution.npy',
    'linear_image_cw': '/home/dhlee/results/cifar10/image_cw_interpolation_linear_attribution.npy',
    
    '0': '/home/dhlee/results/cifar10/image_flat_0_linear_attribution.npy',
    '1': '/home/dhlee/results/cifar10/image_flat_1_linear_attribution.npy',
    '2': '/home/dhlee/results/cifar10/image_flat_2_linear_attribution.npy',
    '3': '/home/dhlee/results/cifar10/image_flat_3_linear_attribution.npy',
    '4': '/home/dhlee/results/cifar10/image_flat_4_linear_attribution.npy',
    '5': '/home/dhlee/results/cifar10/image_flat_5_linear_attribution.npy',
    '6': '/home/dhlee/results/cifar10/image_flat_6_linear_attribution.npy',
    '7': '/home/dhlee/results/cifar10/image_flat_7_linear_attribution.npy',
    '8': '/home/dhlee/results/cifar10/image_flat_8_linear_attribution.npy',
    '9': '/home/dhlee/results/cifar10/image_flat_9_linear_attribution.npy',
    '10': '/home/dhlee/results/cifar10/image_flat_10_linear_attribution.npy',
    '11': '/home/dhlee/results/cifar10/image_flat_11_linear_attribution.npy',
    '12': '/home/dhlee/results/cifar10/image_flat_12_linear_attribution.npy',
    '13': '/home/dhlee/results/cifar10/image_flat_13_linear_attribution.npy',
    '14': '/home/dhlee/results/cifar10/image_flat_14_linear_attribution.npy',
    '15': '/home/dhlee/results/cifar10/image_flat_15_linear_attribution.npy',
    '16': '/home/dhlee/results/cifar10/image_flat_16_linear_attribution.npy',
    '17': '/home/dhlee/results/cifar10/image_flat_17_linear_attribution.npy',
    '18': '/home/dhlee/results/cifar10/image_flat_18_linear_attribution.npy',
    '19': '/home/dhlee/results/cifar10/image_flat_19_linear_attribution.npy',
    '20': '/home/dhlee/results/cifar10/image_flat_20_linear_attribution.npy',
    '21': '/home/dhlee/results/cifar10/image_flat_21_linear_attribution.npy',
    '22': '/home/dhlee/results/cifar10/image_flat_22_linear_attribution.npy',
    '23': '/home/dhlee/results/cifar10/image_flat_23_linear_attribution.npy',
    '24': '/home/dhlee/results/cifar10/image_flat_24_linear_attribution.npy',    
    
}[args.method]

attrs = np.load(attr_path)
classifier = torch.load(args.model_path,  map_location='cpu').eval().to(args.device)

evaluator = Cifar10Evaluator(args.data_path, '/home/dhlee/code/ig_inversion/results/', args.method, debug=args.debug)

evaluator.evaluate(attrs, classifier, **vars(args))






