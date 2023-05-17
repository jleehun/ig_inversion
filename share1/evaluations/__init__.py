import torchvision
import torchvision.transforms as T

from tqdm import tqdm 
import json 
import os 
import numpy as np 

IMAGENET1K_STATS = {
    'mean' : [0.485, 0.456, 0.406],
    'std' : [0.229, 0.224, 0.225]
}

class ImageNet1kEvaluator():
    def __init__(self, data_path, save_dir, debug=False):
        transform = T.Compose([
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(), 
                        T.Normalize(IMAGENET1K_STATS['mean'], IMAGENET1K_STATS['std'])
                    ])
        self.valid_dataset = torchvision.datasets.ImageNet(root=data_path, split="val", transform=transform)
        self.save_dir =save_dir
        if os.path.exists(os.path.join(self.save_dir, 'evaluation_samples.json')):
            self.sample_result_dict = json.load(open(os.path.join(self.save_dir, 'evaluation_samples.json') ,"r"))
        else:
            self.sample_result_dict = {}
        if os.path.exists(os.path.join(self.save_dir, 'evaluation_average.json')):
            self.average_result_dict = json.load(open(os.path.join(self.save_dir, 'evaluation_average.json') ,"r"))
        else:
            self.average_result_dict = {}
            
        self.debug = debug
        self.save()
    
    def evaluate(self, attrs, model, fn, device='cuda:0', **kwargs):
        print(f'{fn.__name__}_{kwargs["ratio"]}')
        """
        attribution evaluation function 
        fn : (input, label, attr, model) --> score 
        """
        
        self.sample_result_dict[f'{fn.__name__}_{kwargs["ratio"]}'] = []
        model = model.to(device)
        pbar = tqdm(range(len(self.valid_dataset)))
        pbar.set_description(f" Evaluation [👾] | {model.__class__.__name__} - {fn.__name__} | ")
        for idx in pbar:
            input, label = self.valid_dataset[idx]
            input = input.to(device)
            attr = attrs[idx]
            score = fn(input, label, attr, model, device, **kwargs)
            self.sample_result_dict[f'{fn.__name__}_{kwargs["ratio"]}'].append(score)
            
            if self.debug:
                if idx > 10:
                    break 
            
        self.average_result_dict[f'{fn.__name__}_{kwargs["ratio"]}'] = np.mean(self.sample_result_dict[f'{fn.__name__}_{kwargs["ratio"]}'])
        self.save()
        
    def save(self):        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, f"evaluation_average.json"), 'w') as f:
            json.dump(self.average_result_dict, f, indent=2, sort_keys=True)
            
        with open(os.path.join(self.save_dir, f"evaluation_samples.json"), 'w') as f:
            json.dump(self.sample_result_dict, f, indent=2, sort_keys=True)
            
            
    def generate_random(self):
        if not os.path.exists("results"):
            os.makedirs("results")
        attrs = np.random.random(size=(50000,224,224))
        np.save('results/attrs.npy', attrs)