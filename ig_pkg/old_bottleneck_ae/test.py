import torch 

model = torch.load("results/model_best.pt", map_location='cpu')
inputs = torch.randn(1, 1, 32, 32)
outputs = model(inputs)
x_hat, loss_dict, info = outputs 

print("input: ", inputs.size())
print("bottlenekc:", (info['bottleneck']))
print("decode:",  model.decode(info['bottleneck']).size())