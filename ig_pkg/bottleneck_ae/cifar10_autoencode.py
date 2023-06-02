import torch 
import random 
import numpy as np

seed=0 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True 

from image_autoencoder_trainer import ImageAutoencoderTrainer
from bottleneck_ae import BottleneckAE
import torchvision 
import torchvision.transforms as T 


CIFAR10_STATS = {
    'mean' : [0.4914, 0.4822, 0.4465],
    'std' : [0.2023, 0.1994, 0.2010]
}

def get_cifar10_dataset(root, resize, train_transform=None, valid_transform=None):
    if train_transform is None:          
        train_transform = T.Compose([T.Resize(resize), 
                               T.RandomCrop(32, padding=4),
                               T.RandomHorizontalFlip(),
                               T.ToTensor(), 
                               T.Normalize(CIFAR10_STATS['mean'], CIFAR10_STATS['std'])])
    if valid_transform is None:
        valid_transform = T.Compose([T.Resize(resize), 
                               T.ToTensor(), 
                               T.Normalize(CIFAR10_STATS['mean'], CIFAR10_STATS['std'])])
    train_dataset = torchvision.datasets.CIFAR10(root=root, train=True,  transform=train_transform, download=True)
    valid_dataset = torchvision.datasets.CIFAR10(root=root, train=False, transform=valid_transform, download=True) 
    info  = {
        'train_transform' : train_transform,
        'valid_transform' : valid_transform
    }
    return train_dataset, valid_dataset, info 

import os 
import argparse
parser =  argparse.ArgumentParser()
parser.add_argument("--data-path", required=True)
args = parser.parse_args()


train_dataset, valid_dataset, info = get_cifar10_dataset(args.data_path, 32, None)
hidden_dim=8
config = {
        'epochs': 100,
        'batch_size' : 32,
        'lr' : 1e-3,
        'clip_grad' : 1.0,
        'device' : "cuda:0",
        'num_workers' : 2,
        "save_dir" : f"results/cifar10_{hidden_dim}" 
    }


model = BottleneckAE((3, 32, 32), 8)

trainer = ImageAutoencoderTrainer(model, 
                                train_dataset, 
                                valid_dataset,
                                **config)

torch.set_num_threads(config['num_workers'])
trainer.train()
