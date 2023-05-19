# https://github.com/ymerkli/fgsm-attack/blob/master/fgsm_attack.py

import torch
import torch.nn as nn
        
def perturb(x, eps, grad, target = None):
    x_prime = None
    if target:
        x_prime = x - eps * grad.sign()
    else:
        x_prime = x + eps * grad.sign()

    # keep image data in the [0,1] range
    x_prime = torch.clamp(x_prime, 0, 1)
    return x_prime

def FGSMAttack(tenser, model, epsilons=0.05, loss=nn.CrossEntropyLoss(), target=None, device=None):
#     model.eval()
#     image_prime = None
#     if device:
#         image, model = image.to(device), model.to(device)

    # FGSM attack requires gradients w.r.t. the data
    image = tensor.detach().clone()
    print(1)
    image.requires_grad = True

    output = model(image)
    init_pred = output.argmax(dim=1, keepdim=True)
    
    if target:
        # in a target attack, we take the loss w.r.t. the target label
        L = loss(output, torch.tensor([self.target], dtype=torch.long))
    else:
        L = loss(output, torch.tensor([init_pred.item()], dtype=torch.long))

    # zero out all existing gradients
    model.zero_grad()
    # calculate gradients
    loss.backward()
    data_grad = image.grad

    perturbed_data = perturb(image, eps, data_grad, target)
    return perturbed_data

def untarget_fgsm(model, loss, image, labels, eps, device) :
    
    images = image.detach().clone().to(device)
    labels = labels.to(device)
    model = model.to(device)
    
    images.requires_grad = True
            
    outputs = model(images)
    
    model.zero_grad()
    cost = loss(outputs, labels)
    cost.backward()
    
    attack_images = images + eps*images.grad.sign()
    
    return attack_images