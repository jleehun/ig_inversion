import numpy as np 
import argparse

parser =argparse.ArgumentParser()
parser.add_argument("--rand",  required=True)
parser.add_argument("--ratio",  required=True)

args = parser.parse_args()
# print()
print(args.rand, args.ratio)