import random
import os
import torch
import torchvision
import numpy as np 
import argparse
from tqdm import tqdm
from distutils.util import strtobool

from ig_pkg.utils.eval import Cifar10Evaluator, mnistEvaluator
from ig_pkg.utils.metrics import * #morf, lerf, 
from ig_pkg.utils.attribution import *

parser =argparse.ArgumentParser()
parser.add_argument("--data-path",  required=True)
# parser.add_argument("--method", required=True)
# parser.add_argument("--attr-path",  required=True)
# parser.add_argument("--model-path", required=True)
parser.add_argument("--type", required=True)
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

attr_path={
    '0': '/home/dhlee/results/mnist/linear_0_attribution.npy',
    '1': '/home/dhlee/results/mnist/linear_1_attribution.npy',
    '2': '/home/dhlee/results/mnist/linear_2_attribution.npy',
    '3': '/home/dhlee/results/mnist/linear_3_attribution.npy',
    '4': '/home/dhlee/results/mnist/linear_4_attribution.npy',
    '5': '/home/dhlee/results/mnist/linear_5_attribution.npy',
    '6': '/home/dhlee/results/mnist/linear_6_attribution.npy',
    '7': '/home/dhlee/results/mnist/linear_7_attribution.npy',
    '8': '/home/dhlee/results/mnist/linear_8_attribution.npy',
    '9': '/home/dhlee/results/mnist/linear_9_attribution.npy',
    '10': '/home/dhlee/results/mnist/linear_10_attribution.npy',
    '11': '/home/dhlee/results/mnist/linear_11_attribution.npy',
    '12': '/home/dhlee/results/mnist/linear_12_attribution.npy',
    '13': '/home/dhlee/results/mnist/linear_13_attribution.npy',
    '14': '/home/dhlee/results/mnist/linear_14_attribution.npy',
    '15': '/home/dhlee/results/mnist/linear_15_attribution.npy',
    '16': '/home/dhlee/results/mnist/linear_16_attribution.npy',
    '17': '/home/dhlee/results/mnist/linear_17_attribution.npy',
    '18': '/home/dhlee/results/mnist/linear_18_attribution.npy',
    '19': '/home/dhlee/results/mnist/linear_19_attribution.npy',
    '20': '/home/dhlee/results/mnist/linear_20_attribution.npy',
    '21': '/home/dhlee/results/mnist/linear_21_attribution.npy',
    '22': '/home/dhlee/results/mnist/linear_22_attribution.npy',
    '23': '/home/dhlee/results/mnist/linear_23_attribution.npy',
    '24': '/home/dhlee/results/mnist/linear_24_attribution.npy',    
}[args.type]

attrs = np.load(attr_path)
# classifier = torch.load(args.model_path,  map_location='cpu').eval().to(args.device)

from ebm_pkg.models import get_model
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
classifier = model.eval().to(args.device)



evaluator = mnistEvaluator(args.data_path, '/home/dhlee/code/ig_inversion/results/mnist', args.type, debug=args.debug)

evaluator.evaluate(attrs, classifier, **vars(args))






