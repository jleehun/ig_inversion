import torchvision
import torchvision.transforms as T

from tqdm import tqdm 
import json 
import os 
import numpy as np 
from ig_pkg.utils.metrics import morf, lerf, aopc, lodds, mnist_one_stop

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

MNIST_MEAN = [0.1307]
MNIST_STD  = [0.3081] 

class mnistEvaluator():
    def __init__(self, data_path, save_dir, method, debug=False):
        
        self.valid_dataset = torchvision.datasets.MNIST(root = '/data8/donghun',
                            train=True,
                            transform=T.Compose([T.ToTensor(), T.Normalize(mean = MNIST_MEAN, std = MNIST_STD)]),
        )
        
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
        self.method = method
        self.save()
    
    def evaluate(self, attrs, model, device, **kwargs):
        print(f'mnist_{self.method}_{kwargs["ratio"]}')

        """
        attribution evaluation function 
        fn : (input, label, attr, model) --> score 
        """
                
        self.sample_result_dict[f'morf_{self.method}_{kwargs["ratio"]}'] = []
        # self.sample_result_dict[f'lerf_{self.method}_{kwargs["ratio"]}'] = []
        # self.sample_result_dict[f'lodds_{self.method}_{kwargs["ratio"]}'] = []
        self.sample_result_dict[f'aopc_{self.method}_{kwargs["ratio"]}'] = []
        self.sample_result_dict[f'pred_{self.method}_{kwargs["ratio"]}'] = []
        
        model = model.to(device)
        pbar = tqdm(range(len(self.valid_dataset)))
        pbar.set_description(f" Evaluation [👾] | {model.__class__.__name__} | {self.method} | {kwargs['ratio']}")
        for idx in pbar:
            input, label = self.valid_dataset[idx]
            input = input.to(device)
            attr = attrs[idx]

            morf, aopc, y_hat = mnist_one_stop(input, label, attr, model, device, **kwargs)
            self.sample_result_dict[f'morf_{self.method}_{kwargs["ratio"]}'].append(morf)
            self.sample_result_dict[f'aopc_{self.method}_{kwargs["ratio"]}'].append(aopc)
            self.sample_result_dict[f'pred_{self.method}_{kwargs["ratio"]}'].append(y_hat)

            if self.debug:
                if idx > 10:
                    break 
            
        self.average_result_dict[f'morf_{self.method}_{kwargs["ratio"]}'] = np.mean(self.sample_result_dict[f'morf_{self.method}_{kwargs["ratio"]}'])
        self.average_result_dict[f'aopc_{self.method}_{kwargs["ratio"]}'] = np.mean(self.sample_result_dict[f'aopc_{self.method}_{kwargs["ratio"]}'])
        self.average_result_dict[f'pred_{self.method}_{kwargs["ratio"]}'] = np.mean(self.sample_result_dict[f'pred_{self.method}_{kwargs["ratio"]}'])
        self.save()
        
    def save(self):        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, f"evaluation_average_cifar10.json"), 'w') as f:
            json.dump(self.average_result_dict, f, indent=2, sort_keys=True)
            
        with open(os.path.join(self.save_dir, f"evaluation_samples_cifar10.json"), 'w') as f:
            json.dump(self.sample_result_dict, f, indent=2, sort_keys=True)            
            
class Cifar10Evaluator():
    def __init__(self, data_path, save_dir, method, debug=False):
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
        self.method = method
        self.save()
    
    def evaluate(self, attrs, model, device, **kwargs):
        print(f'cifar10_{self.method}_{kwargs["ratio"]}')

        """
        attribution evaluation function 
        fn : (input, label, attr, model) --> score 
        """
                
        self.sample_result_dict[f'morf_{self.method}_{kwargs["ratio"]}'] = []
        # self.sample_result_dict[f'lerf_{self.method}_{kwargs["ratio"]}'] = []
        # self.sample_result_dict[f'lodds_{self.method}_{kwargs["ratio"]}'] = []
        self.sample_result_dict[f'aopc_{self.method}_{kwargs["ratio"]}'] = []
        self.sample_result_dict[f'pred_{self.method}_{kwargs["ratio"]}'] = []
        
        model = model.to(device)
        pbar = tqdm(range(len(self.valid_dataset)))
        pbar.set_description(f" Evaluation [👾] | {model.__class__.__name__} | {self.method} | {kwargs['ratio']}")
        for idx in pbar:
            input, label = self.valid_dataset[idx]
            input = input.to(device)
            attr = attrs[idx]
            # print(1)
            # print(attr.shape)
            # score = morf(input, label, attr, model, device, **kwargs)
            # self.sample_result_dict[f'morf_{self.method}_{kwargs["ratio"]}'].append(score)
            # print(2)
            
            # score = lerf(input, label, attr, model, device, **kwargs)
            # self.sample_result_dict[f'lerf_{self.method}_{kwargs["ratio"]}'].append(score)
            # print(3)
            
            # score = aopc(input, label, attr, model, device, **kwargs)
            # self.sample_result_dict[f'aopc_{self.method}_{kwargs["ratio"]}'].append(score)
            # print(4)
            
            # score = lodds(input, label, attr, model, device, **kwargs)
            # self.sample_result_dict[f'lodds_{self.method}_{kwargs["ratio"]}'].append(score)
            
            morf, aopc, y_hat = mnist_one_stop(input, label, attr, model, device, **kwargs)
            self.sample_result_dict[f'morf_{self.method}_{kwargs["ratio"]}'].append(morf)
            self.sample_result_dict[f'aopc_{self.method}_{kwargs["ratio"]}'].append(aopc)
            self.sample_result_dict[f'pred_{self.method}_{kwargs["ratio"]}'].append(y_hat)

            if self.debug:
                if idx > 10:
                    break 
            
        self.average_result_dict[f'morf_{self.method}_{kwargs["ratio"]}'] = np.mean(self.sample_result_dict[f'morf_{self.method}_{kwargs["ratio"]}'])
        # self.average_result_dict[f'lerf_{self.method}_{kwargs["ratio"]}'] = np.mean(self.sample_result_dict[f'lerf_{self.method}_{kwargs["ratio"]}'])
        self.average_result_dict[f'aopc_{self.method}_{kwargs["ratio"]}'] = np.mean(self.sample_result_dict[f'aopc_{self.method}_{kwargs["ratio"]}'])
        # self.average_result_dict[f'lodds_{self.method}_{kwargs["ratio"]}'] = np.mean(self.sample_result_dict[f'lodds_{self.method}_{kwargs["ratio"]}'])
        self.average_result_dict[f'pred_{self.method}_{kwargs["ratio"]}'] = np.mean(self.sample_result_dict[f'pred_{self.method}_{kwargs["ratio"]}'])
        self.save()
        
    def save(self):        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, f"evaluation_average_cifar10.json"), 'w') as f:
            json.dump(self.average_result_dict, f, indent=2, sort_keys=True)
            
        with open(os.path.join(self.save_dir, f"evaluation_samples_cifar10.json"), 'w') as f:
            json.dump(self.sample_result_dict, f, indent=2, sort_keys=True)            