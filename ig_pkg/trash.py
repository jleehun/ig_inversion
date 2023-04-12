import time
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from ig_pkg.loss.focal_loss import FocalLoss
from ig_pkg.loss.metrics import ArcMarginProduct, AddMarginProduct
import torchvision.models as models

train_dataset, valid_dataset = get_datasets(name= 'celebAHQ_whole', data_path = '/root/data/identity_celebahq')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features # 512
model.fc = nn.Linear(num_features, 307) # multi-class classification (num_of_class == 307)
model = model.to(device)

# lr = 1e-1  # initial learning rate
# lr_step = 10
# lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
# weight_decay = 5e-4

criterion = FocalLoss(gamma=2)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
metric_fc = ArcMarginProduct(1000, 307, s=20, m=0.3).to(device)
# metric_fc =  AddMarginProduct(1000, 307, s=30, m=0.5).to(device)

optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=1e-1, weight_decay=5e-4)
# optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
#                                      lr=1e-1, weight_decay=5e-4)

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 30
start_time = time.time()

for epoch in range(num_epochs):
    scheduler.step()
    """ Training Phase """
    model.train()

    running_loss = 0.
    running_corrects = 0

    # load a batch data of images
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward inputs and get output
        outputs = model(inputs)
        outputs = metric_fc(outputs, labels)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # get loss value and update the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset) * 100.
    print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))

    """ Test Phase """
    model.eval()

    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0

        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = metric_fc(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(valid_dataset)
        epoch_acc = running_corrects / len(valid_dataset) * 100.        
        print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))
        
        if epoch_acc > 90:
            save_path = f'/root/pretrained/facial_identity_classification_Arcface_with_ResNet18_resolution_244_{epoch_acc}.pth'
            torch.save(model.state_dict(), save_path)