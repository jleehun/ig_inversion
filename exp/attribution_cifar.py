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
# parser.add_argument("--type", required=True, type=int)
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

    # '0': f'/data8/donghun/results/attribution_{args.type}/latent_0_linear_attribution.npy',
    # '1': f'/data8/donghun/results/attribution_{args.type}/latent_1_linear_attribution.npy',
    # '2': f'/data8/donghun/results/attribution_{args.type}/latent_2_linear_attribution.npy',
    # '3': f'/data8/donghun/results/attribution_{args.type}/latent_3_linear_attribution.npy',
    # '4': f'/data8/donghun/results/attribution_{args.type}/latent_4_linear_attribution.npy',
    # '5': f'/data8/donghun/results/attribution_{args.type}/latent_5_linear_attribution.npy',
    # '6': f'/data8/donghun/results/attribution_{args.type}/latent_6_linear_attribution.npy',
    # '7': f'/data8/donghun/results/attribution_{args.type}/latent_7_linear_attribution.npy',
    # '8': f'/data8/donghun/results/attribution_{args.type}/latent_8_linear_attribution.npy',
    # '9': f'/data8/donghun/results/attribution_{args.type}/latent_9_linear_attribution.npy',
    # '10': f'/data8/donghun/results/attribution_{args.type}/latent_10_linear_attribution.npy',
    # '11': f'/data8/donghun/results/attribution_{args.type}/latent_11_linear_attribution.npy',
    # '12': f'/data8/donghun/results/attribution_{args.type}/latent_12_linear_attribution.npy',
    # '13': f'/data8/donghun/results/attribution_{args.type}/latent_13_linear_attribution.npy',
    # '14': f'/data8/donghun/results/attribution_{args.type}/latent_14_linear_attribution.npy',
    # '15': f'/data8/donghun/results/attribution_{args.type}/latent_15_linear_attribution.npy',
    # '16': f'/data8/donghun/results/attribution_{args.type}/latent_16_linear_attribution.npy',
    # '17': f'/data8/donghun/results/attribution_{args.type}/latent_17_linear_attribution.npy',
    # '18': f'/data8/donghun/results/attribution_{args.type}/latent_18_linear_attribution.npy',
    # '19': f'/data8/donghun/results/attribution_{args.type}/latent_19_linear_attribution.npy',
    # '20': f'/data8/donghun/results/attribution_{args.type}/latent_20_linear_attribution.npy',
    # '21': f'/data8/donghun/results/attribution_{args.type}/latent_21_linear_attribution.npy',
    # '22': f'/data8/donghun/results/attribution_{args.type}/latent_22_linear_attribution.npy',
    # '23': f'/data8/donghun/results/attribution_{args.type}/latent_23_linear_attribution.npy',
    # '24': f'/data8/donghun/results/attribution_{args.type}/latent_24_linear_attribution.npy',    
    # 'max': f'/data8/donghun/results/new/attribution/max_linear_attribution.npy',
    # 'min': f'/data8/donghun/results/new/attribution/min_linear_attribution.npy',
    
    'one': f'/data8/donghun/results/new/attribution/1.0_linear_attribution.npy',
    'half': f'/data8/donghun/results/new/attribution/0.5_linear_attribution.npy',
    'minus1': f'/data8/donghun/results/new/attribution/-1.0_linear_attribution.npy',
    'minus5': f'/data8/donghun/results/new/attribution/-0.5_linear_attribution.npy',
    
    'cir_1': f'/data8/donghun/results/new/attribution/circle_0_linear_attribution.npy',
    'cir_2': f'/data8/donghun/results/new/attribution/circle_1_linear_attribution.npy',
    'cir_4': f'/data8/donghun/results/new/attribution/circle_2_linear_attribution.npy',
    'cir_8': f'/data8/donghun/results/new/attribution/circle_3_linear_attribution.npy',
    'cir_16': f'/data8/donghun/results/new/attribution/circle_4_linear_attribution.npy',
    
    'alt_1': f'/data8/donghun/results/new/attribution/alt_0_linear_attribution.npy',
    'alt_2': f'/data8/donghun/results/new/attribution/alt_1_linear_attribution.npy',
    'alt_4': f'/data8/donghun/results/new/attribution/alt_2_linear_attribution.npy',
    'alt_8': f'/data8/donghun/results/new/attribution/alt_3_linear_attribution.npy',
    'alt_16': f'/data8/donghun/results/new/attribution/alt_4_linear_attribution.npy',
    
}[args.method]


attrs = np.load(attr_path)
classifier = torch.load(args.model_path,  map_location='cpu').eval().to(args.device)

evaluator = Cifar10Evaluator(args.data_path, f'/home/dhlee/code/ig_inversion/results/latent/', args.method, debug=args.debug)

evaluator.evaluate(attrs, classifier, **vars(args))





