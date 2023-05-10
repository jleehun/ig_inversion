
import torch 
from torch.autograd import Variable

def make_interpolation(x, M, baseline):
    lst = [] 
    for i in range(M+1):
        alpha = float(i/M)  
        interpolated =x * (alpha) + baseline * (1-alpha)
        lst.append(interpolated.clone())
    return torch.stack(lst)

def ig(model, x, y, baseline, device='cuda:0', M=25, **kwrags):
    x = x.to(device)
    baseline = baseline.to(device)
    model.zero_grad()
    
    X = make_interpolation(x, M, baseline)
    X = Variable(X, requires_grad=True).to(device)
    X.retain_grad()
    
    output = model(X,)
    score = torch.softmax(output, dim=-1)
    class_score = torch.FloatTensor(X.size(0), output.size()[-1]).zero_().to(device).type(X.dtype)
    class_score[:,y] = score[:,y]
    output.backward(gradient=class_score)

    gradient = X.grad  #Approximate the integral using the trapezoidal rule
    gradient = (gradient[:-1] + gradient[1:]) / 2.0
    output = (x - baseline) * gradient.mean(axis=0)
    output = output.mean(dim=0) # RGB mean
    output = output.abs()
    return output

def ig_one(model, x, y, baseline, device='cuda:0', M=25, **kwrags):
    x = x.to(device)
    baseline = baseline.to(device)
    model.zero_grad()
    
    X = make_interpolation(x, M, baseline)
    X = Variable(X, requires_grad=True).to(device)
    X.retain_grad()
    
    output = model(X,)
    score = torch.softmax(output, dim=-1)
    class_score = torch.FloatTensor(X.size(0), output.size()[-1]).zero_().to(device).type(X.dtype)
    class_score[:,y] = score[:,y]
    output.backward(gradient=class_score)

    gradient = X.grad  #Approximate the integral using the trapezoidal rule
    gradient = (gradient[:-1] + gradient[1:]) / 2.0
#     output = (x - baseline) * gradient.mean(axis=0)
    output = gradient.mean(axis=0)
    output = output.mean(dim=0) # RGB mean
    output = output.abs()
    return output