
import os 
import torch 
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm 
import time 
from utils import make_optimizer, make_lr_scheduler
from torch.utils.tensorboard import SummaryWriter

class ImageAutoencoderTrainer:
    def __init__(self, model, 
                 train_dataset, 
                 valid_dataset, 
                 optimizer_type='sgd',
                 lr_scheudler_type='cosine',
                 **kwargs):
        self.flags = OmegaConf.create(kwargs)
        
        self.model = model 
        self.model.to(self.flags.device)

        self.valid_dataset = valid_dataset        
        self.train_dataloader  = DataLoader(train_dataset, 
                                            batch_size=self.flags.batch_size, 
                                            shuffle=True, 
                                            num_workers=self.flags.num_workers)
    
        self.optimizer    = make_optimizer(self.model.parameters(), optimizer_type,  **kwargs)
        self.lr_scheduler = make_lr_scheduler(self.optimizer, lr_scheudler_type,  **kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        

        if not os.path.exists(self.flags.save_dir):
            os.makedirs(self.flags.save_dir)
        
        
        self.flags.train_sgd_steps =  len(self.train_dataloader.dataset)//self.flags.batch_size
        self.flags.best_performance = 1e8
        self.flags.results ={
                'eval/reconstruction_loss':[],
                'last_lr' :[],
                'train/reconstruction_loss':[],
            }
        OmegaConf.save(self.flags, os.path.join(self.flags.save_dir, 'config.yaml'))
        self.writer = SummaryWriter(os.path.join(self.flags.save_dir, 'runs'))

        
    def train(self):
        start_time = time.time()
        
        for epoch in range(self.flags.epochs):
            running_loss_dict = {}
            
            self.model.train()
            pbar = tqdm(enumerate(self.train_dataloader), total=self.flags.train_sgd_steps)
            for i, (x,y) in pbar:
                x = x.to(self.flags.device)
                # map the zeros to zeros 
                x = torch.cat([x, torch.zeros_like(x).to(self.flags.device)], dim=0)
                y = y.to(self.flags.device)
                
                x_hat, loss_dict, info = self.model(x)
                loss = loss_dict['total_loss']

                self.optimizer.zero_grad()                                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.flags.clip_grad)
                self.optimizer.step()
            
                for k, v in loss_dict.items():
                    if k not in running_loss_dict.keys():
                        running_loss_dict[k] = 0.0
                    running_loss_dict[k] += v.item()
                    
                duration = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))         
                pbar.set_description(f"🧪:{self.flags.save_dir} [E:({epoch/(self.flags.epochs):.2f}) D:({duration})] Loss:{running_loss_dict['total_loss']/(i+1):.3E}")  

        
            self.flags.results['last_lr'].append(self.lr_scheduler.get_last_lr()[0])
            self.writer.add_scalar("train/lr", self.lr_scheduler.get_last_lr()[0], epoch)
            for k, v in running_loss_dict.items():
                if f'train/{k}' in  self.flags.results.keys():
                    self.flags.results[f'train/{k}'].append(v/self.flags.train_sgd_steps)
                else:
                    self.flags.results[f'train/{k}'] = [(v/self.flags.train_sgd_steps)]
            
            self.lr_scheduler.step()
            if self.valid_dataset is not None:
                eval_loss_dict = ImageAutoencoderTrainer.evaluate(self.model, 
                                                     self.valid_dataset, 
                                                     self.flags.batch_size, 
                                                     self.flags.device,
                                                     num_workers=self.flags.num_workers)
                for k, v in eval_loss_dict.items():
                    if f'evaluate/{k}' in  self.flags.results.keys():
                        self.flags.results[f'evaluate/{k}'].append(v/self.flags.train_sgd_steps)
                    else:
                        self.flags.results[f'evaluate/{k}'] = [(v/self.flags.train_sgd_steps)]
                        
                print(f"🌹:{self.flags.save_dir}|Eval Resonstruction: {(eval_loss_dict['reconstruction_loss']):.3f}")
        
                if eval_loss_dict['reconstruction_loss'] <= self.flags.best_performance:
                    torch.save(self.model, os.path.join(self.flags.save_dir, f'model_best.pt'))
                    self.flags.best_performance = eval_loss_dict['reconstruction_loss']
            OmegaConf.save(self.flags, os.path.join(self.flags.save_dir, 'config.yaml'))
            torch.save(self.model, os.path.join(self.flags.save_dir, f'model_last.pt'))
            
            
    @staticmethod
    def evaluate(model, dataset, batch_size, device, num_workers):
        data_loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        is_train = model.training 
        model.eval()
        running_loss_dict = {}
        
        count = 0 
        pbar = tqdm((enumerate(data_loader)), total=(len(dataset)//batch_size))
        for k, (x, y) in pbar:
            count +=1
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                x_hat, loss_dict, info =model(x, y=y)
                for k, v in loss_dict.items():
                    if k not in running_loss_dict.keys():
                        running_loss_dict[k] = 0.0
                    running_loss_dict[k] += v.item()
                    
        # average
        for k, v in running_loss_dict.items():
            running_loss_dict[k] = v/count 
        if is_train:
            model.train()
        return running_loss_dict