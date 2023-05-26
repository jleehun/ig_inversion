import torch 
from torch.autograd import Variable


def get_gradient(model, x, y, device):
    temp = Variable(x, requires_grad=True).to(device)
    temp = temp.unsqueeze(0)
    temp.retain_grad()
    model.zero_grad()

    output = model(temp)
    score = torch.softmax(output, dim=-1)
    class_score = torch.FloatTensor(temp.size(0), output.size()[-1]).zero_().to(device).type(temp.dtype)
    class_score[:,y] = score[:,y]
    output.backward(gradient=class_score)

    gradient = temp.grad
    return gradient

def image_gradient_ascent_interpolation(model, x, y, M, device): # last - baseline
    model = model.to(device)
    x = x.to(device)
    
    interp = []
    interp.append(x)
    
    for i in range(M):    
        grad = get_gradient(model, x, y, device)
        x = x + grad.squeeze(0)        
        interp.append(x)
    interp = torch.stack(interp)
    return interp

def image_gradient_interpolation(model, x, y, M, device): # last - baseline
    model = model.to(device)
    x = x.to(device)
    
    interp = []
    interp.append(x)
    
    for i in range(M):    
        grad = get_gradient(model, x, y, device)
        x = x - grad.squeeze(0)        
        interp.append(x)
    interp = torch.stack(interp)
    return interp

def linear_interpolation(x, M, baseline): # first baseline # last image
    lst = [] 
    for i in range(M+1):
        alpha = float(i/M)  
        interpolated =x * (alpha) + baseline * (1-alpha)
        lst.append(interpolated.clone())
    return torch.stack(lst)

def integrated_gradient(model, x, y, baseline, interpolation, device, **kwrags): 
#     x = x.to(device)
#     baseline = baseline.to(device)
#     interpolation = interpolation.to(device)
    model.zero_grad()
    
    X = Variable(interpolation, requires_grad=True).to(device)
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
    # print(5, output.shape)
    output = output.abs()
    # print(6, output.shape)
    return output
