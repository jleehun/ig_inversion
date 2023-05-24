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

device = 'cuda:7'

attr_path  = {
    'zero': '/home/dhlee/results/cifar10/image_linear_zero_attribution.npy',
    'expected': '/home/dhlee/results/cifar10/image_linear_expected_attribution.npy,'
}[args.method]

attrs = np.load(attr_path)
classifier = torch.load(args.model_path,  map_location='cpu').eval().to(device)

evaluator = Cifar10Evaluator(args.data_path, '/home/dhlee/code/ig_inversion/results/', args.method, debug=args.debug)

evaluator.evaluate(attrs, classifier, device, **vars(args))




