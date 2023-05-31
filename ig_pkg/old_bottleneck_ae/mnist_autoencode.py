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

MNIST_STATS = {
    'mean' : [0.1307],
    'std' : [0.3081] 
}

def get_mnist_dataset(root, resize, transform=None):
    if transform is None:
        transform = T.Compose([T.Resize(resize), T.ToTensor(), T.Normalize(MNIST_STATS['mean'], MNIST_STATS['std'])])
    
    train_dataset = torchvision.datasets.MNIST(root=root, train=True, transform=transform, download=True)
    valid_dataset = torchvision.datasets.MNIST(root=root, train=False, transform=transform, download=True) 
    info  = {
        'transform'
    }
    return train_dataset, valid_dataset, info 

import os 
import argparse
parser =  argparse.ArgumentParser()
parser.add_argument("--data-path", required=True)
args = parser.parse_args()


train_dataset, valid_dataset, info = get_mnist_dataset(args.data_path, 32, None)
config = {
        'epochs': 30,
        'batch_size' : 32,
        'lr' : 1e-3,
        'clip_grad' : 1.0,
        'device' : "cuda:0",
        'num_workers' : 2,
        "save_dir" : "results" 
    }


model = BottleneckAE((1, 32, 32), 2)

trainer = ImageAutoencoderTrainer(model, 
                                train_dataset, 
                                valid_dataset,
                                **config)

torch.set_num_threads(config['num_workers'])
trainer.train()