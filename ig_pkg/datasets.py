import torch 
import torchvision 
import torchvision.transforms as T
from torch.utils.data import Dataset
import os

from torchvision import datasets, models


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CIFAR100_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR100_STD  = [0.2023, 0.1994, 0.2010] 

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2023, 0.1994, 0.2010]

MNIST_MEAN = [0.1307]
MNIST_STD  = [0.3081] 

class custom_dataset(Dataset):
    def __init__(self, name, root = '/root/data', transform = None):                        
        self.root = root
        
        if name == 'LSUN_bedroom': self.dir = os.path.join(self.root, 'LSUN_bedroom', 'real_10k')        
        elif name == 'FFHQ': self.dir = os.path.join(self.root, name, 'img')        
        else: self.dir = os.path.join(self.root, name)        
        
        self.file_names = sorted(os.listdir(self.dir))
        self.transform = transform
    
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.file_names[idx]))
        
        if self.transform:
            img = self.transform(img)

        return img
    
# ----------- STATIC functions -----------------
def get_datasets(name, data_path, transform=None):
    # ---- Define the wrapper if required -----
    if transform is None:
        if 'cifar' in name or 'mnist' in name:
            mean, std = {
                "cifar10": [CIFAR10_MEAN, CIFAR10_STD],
                "cifar100": [CIFAR100_MEAN, CIFAR100_STD],
                "mnist": [MNIST_MEAN, MNIST_STD],
                "fashion_mnist": [MNIST_MEAN, MNIST_STD], # incorrect!
            }[name]
            transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        elif 'imagenet' in name:
            transform = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD),
            ])

        elif 'celeb' in name:
            print('flip')
            transform = T.Compose([
                            T.Resize(224),
                            T.RandomHorizontalFlip(),
                            T.CenterCrop(224),
                            T.ToTensor(), 
                            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                        ])
        else:
            transform = T.Compose([
                            T.Resize(224),
                            T.CenterCrop(224),
                            T.ToTensor(), 
                            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                        ])
        

    # ------ CIFAR ---------
    if name =="cifar10": # sadsad
        train_dataset  = torchvision.datasets.CIFAR10(root=data_path,  
                                                    train=True,  
                                                    download=True,
                                                    transform=transform) 
        valid_dataset  = torchvision.datasets.CIFAR10(root = data_path,  
                                                    train=False,  
                                                    download=True,
                                                    transform =transform) 

    elif name =="cifar100":
        train_dataset  = torchvision.datasets.CIFAR100(root=data_path,  
                                                    train=True,  
                                                    download=True,
                                                    transform=transform) 
        valid_dataset  = torchvision.datasets.CIFAR100(root = data_path,  
                                                    train=False,  
                                                    download=True,
                                                    transform =transform) 
    # ------ ImageNet ---------
    elif name =="imagenet1k":
        # train_dataset = torchvision.datasets.ImageNet(root=data_path, split="train", transform=transform)
        train_dataset = None  # need to be prepared in the future
        valid_dataset = torchvision.datasets.ImageNet(root=data_path, split="val", transform=transform)
    
    elif name == "mnist":
        train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
        valid_dataset = torchvision.datasets.MNIST(root=data_path, train=False, transform=transform, download=True) 

    elif name == "fashion_mnist":
        train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=transform, download=True)
        valid_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=False, transform=transform, download=True) 
    
    elif name == 'celebAHQ_identity':        
        train_dataset = datasets.ImageFolder(os.path.join(data_path, 'train'), transform)
        valid_dataset = datasets.ImageFolder(os.path.join(data_path, 'test'), transform)
        # https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch/blob/main/Facial_Identity_Classification_Test_with_CelebA_HQ.ipynb
    
    elif name == 'celebAHQ_whole':        
        train_dataset = datasets.ImageFolder(data_path, transform)
        valid_dataset = 1
    
    elif name == 'celebAHQ_5':        
        train_dataset = datasets.ImageFolder(data_path, transform)
        valid_dataset = 1
    else:
        raise ValueError(f"{name} is not implemented data")
    return train_dataset, valid_dataset



def get_imagenet_image_boundary():
    
    a = torch.zeros(3,224,224)
    b = torch.zeros(3,224,224).fill_(1.0)
    
    min_img = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(a)
    max_img = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(b)
    
    return min_img, max_img


if __name__ =="__main__":
#     min_img, max_img = get_imagenet_image_boundary()
#     print(min_img[:,0,0])
#     print(max_img[:,0,0])
    train_dataset, test_dataset = get_datasets(name= 'celebAHQ_identity', data_path = '/root/data/CelebA_HQ_facial_identity_dataset')
    print('hello')

