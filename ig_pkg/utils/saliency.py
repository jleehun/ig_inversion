import torch 
from torch.autograd import Variable
import numpy as np 
from tqdm import tqdm 
import os 

class SaliencyGenerator():
    @staticmethod
    def ig(input, label, model, device, num_samples, baseline_type, **kwargs):
        # print(kwargs)
        model.zero_grad()
        
        baseline = SaliencyGenerator.make_baseline(input, baseline_type)
        X = SaliencyGenerator.make_interpolation(input, num_samples, baseline)

        X = Variable(X, requires_grad=True).to(device)
        X.retain_grad()
        
        input = input.to(device)
        baseline = baseline.to(device)
        
        output = model(X)
        score = torch.softmax(output, dim=-1)
        class_score = torch.FloatTensor(X.size(0), output.size()[-1]).zero_().to(device).type(X.dtype)
        class_score[:,label] = score[:,label]
        output.backward(gradient=class_score)

        gradient = X.grad  #A pproximate the integral using the trapezoidal rule
        gradient = (gradient[:-1] + gradient[1:]) / 2.0
        output = (input - baseline) * gradient.mean(axis=0)
        output = output.abs()
        output = output.mean(dim=0) # RGB mean
        return output, baseline
    
    @staticmethod
    def make_baseline(input, baseline_type):
        B = torch.zeros_like(input).to(input.device)
        if baseline_type == 'zeros':
            pass
        elif baseline_type == "max":
            for channel in range(input.shape[0]):   
                B[channel].fill_(input[channel].max())
        elif baseline_type == "min":
            for channel in range(input.shape[0]):   
                B[channel].fill_(input[channel].min())
        elif baseline_type == "mean":
            for channel in range(input.shape[0]):   
                B[channel].fill_(input[channel].mean())
        return B 
    
    @staticmethod
    def make_interpolation(x, M, baseline):
        lst = [] 
        for i in range(M+1):
            alpha = float(i/M)  
            interpolated =x * (alpha) + baseline * (1-alpha)
            lst.append(interpolated.clone())
        return torch.stack(lst)