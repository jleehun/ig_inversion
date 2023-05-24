import torchvision
import torchvision.transforms as T

from tqdm import tqdm 
import json 
import os 
import numpy as np 
from ig_pkg.utils.metrics import morf, lerf, aopc, lodds

IMAGENET1K_STATS = {
    'mean' : [0.485, 0.456, 0.406],
    'std' : [0.229, 0.224, 0.225]
}

CIFAR10_STATS = {
    'mean' : [0.4914, 0.4822, 0.4465],
    'std' : [0.2023, 0.1994, 0.2010]
}

fn = {
    'morf': morf,
    'lerf' : lerf,
    'aopc': aopc,
    'lodds': lodds
}

class Cifar10Evaluator():
    def __init__(self, data_path, save_dir, debug=False):
        transform = T.Compose([
                        T.ToTensor(), 
                        T.Normalize(CIFAR10_STATS['mean'], CIFAR10_STATS['std'])
                    ])
        self.valid_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, transform=transform)

        self.save_dir =save_dir
        if os.path.exists(os.path.join(self.save_dir, 'evaluation_samples_cifar10.json')):
            self.sample_result_dict = json.load(open(os.path.join(self.save_dir, 'evaluation_samples_cifar10.json') ,"r"))
        else:
            self.sample_result_dict = {}
        if os.path.exists(os.path.join(self.save_dir, 'evaluation_average_cifar10.json')):
            self.average_result_dict = json.load(open(os.path.join(self.save_dir, 'evaluation_average_cifar10.json') ,"r"))
        else:
            self.average_result_dict = {}
        
        self.debug = debug
        self.save()
    
    def evaluate(self, attrs, model, measure, device='cuda:0', **kwargs):
        print(f'cifar10_{kwargs["ratio"]}')
        print(measure)
        """
        attribution evaluation function 
        fn : (input, label, attr, model) --> score 
        """
                
        self.sample_result_dict[f'morf_{kwargs["ratio"]}'] = []
        self.sample_result_dict[f'lerf_{kwargs["ratio"]}'] = []
        self.sample_result_dict[f'aopc_{kwargs["ratio"]}'] = []
        self.sample_result_dict[f'lodds_{kwargs["ratio"]}'] = []
        
        model = model.to(device)
        pbar = tqdm(range(len(self.valid_dataset)))
        pbar.set_description(f" Evaluation [👾] | {model.__class__.__name__} | ")
        for idx in pbar:
            input, label = self.valid_dataset[idx]
            input = input.to(device)
            attr = attrs[idx]
            
            score = morf(input, label, attr, model, device, **kwargs)
            self.sample_result_dict[f'morf_{kwargs["ratio"]}'].append(score)
            
            score = lerf(input, label, attr, model, device, **kwargs)
            self.sample_result_dict[f'lerf_{kwargs["ratio"]}'].append(score)
            
            score = aopc(input, label, attr, model, device, **kwargs)
            self.sample_result_dict[f'aopc_{kwargs["ratio"]}'].append(score)
            
            score = lodds(input, label, attr, model, device, **kwargs)
            self.sample_result_dict[f'lodds_{kwargs["ratio"]}'].append(score)
            
            if self.debug:
                if idx > 10:
                    break 
            
        self.average_result_dict[f'morf_{kwargs["ratio"]}'] = np.mean(self.sample_result_dict[f'morf_{kwargs["ratio"]}'])
        self.average_result_dict[f'lerf_{kwargs["ratio"]}'] = np.mean(self.sample_result_dict[f'lerf_{kwargs["ratio"]}'])
        self.average_result_dict[f'aopc_{kwargs["ratio"]}'] = np.mean(self.sample_result_dict[f'aopc{kwargs["ratio"]}'])
        self.average_result_dict[f'lodds_{kwargs["ratio"]}'] = np.mean(self.sample_result_dict[f'lodds_{kwargs["ratio"]}'])
        self.save()
        
    def save(self):        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, f"evaluation_average_cifar10.json"), 'w') as f:
            json.dump(self.average_result_dict, f, indent=2, sort_keys=True)
            
        with open(os.path.join(self.save_dir, f"evaluation_samples_cifar10.json"), 'w') as f:
            json.dump(self.sample_result_dict, f, indent=2, sort_keys=True)            