
import torch 

def morf(input, label, attr, model, device, ratio, **kwargs):
    x = mask_MoRF(input, attr, ratio).unsqueeze(0)
    x = x.to(device)
    y_hat = model.forward(x).argmax(dim=-1)
    label = torch.tensor(label)
    score = (y_hat == label).sum().item()
    return  score


def lerf(input, label, attr, model, device, ratio, **kwargs):
    x = mask_LeRF(input, attr, ratio).unsqueeze(0)
    x = x.to(device)
    y_hat = model.forward(x).argmax(dim=-1)
    label = torch.tensor(label)
    score = (y_hat == label).sum().item()
    return  score

# ------------------------------------------------------------------------

def mask_MoRF(x, attr, ratio):
    original_size = x.size()
    x = x.reshape(3, -1)
    attr = torch.tensor(attr).flatten()
    v, index = torch.sort(attr, descending=True, dim=0)    
    index = index[:int(x.size(1)*ratio)]
    x[:, index] = 0.0 
    x = x.reshape(*original_size)
    return x 

def mask_LeRF(x, attr, ratio):
    original_size = x.size()
    x = x.reshape(3, -1)
    attr = torch.tensor(attr).flatten()
    v, index = torch.sort(attr, descending=True, dim=0)    
    index = index[-int(x.size(1)*ratio):]
    x[:, index] = 0.0 
    x = x.reshape(*original_size)
    return x 
