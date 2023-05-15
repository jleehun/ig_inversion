from evaluations import ImageNet1kEvaluator
import numpy as np 
import argparse
from distutils.util import strtobool
from evaluations.morf_lerf import morf, lerf

parser =argparse.ArgumentParser()
parser.add_argument("--data-path",  required=True)
parser.add_argument("--model-name", required=True)
parser.add_argument("--attr-path",  required=True)
parser.add_argument("--measure",  required=True)
parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,)

# -----------------------------
# morf lerf : removing ratio (0 ~ 1.0)
parser.add_argument("--ratio",  type=float)

# -----------------------------

args = parser.parse_args()

from torchvision.models import resnet50, ResNet50_Weights
weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights)
attrs = np.load(args.attr_path)
evaluator = ImageNet1kEvaluator(args.data_path, 'results', debug=args.debug)


fn = {
    'morf': morf,
    'lerf' : lerf
}[args.measure]

evaluator.evaluate(attrs, model, fn, device='cuda:0', **vars(args))