import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm  # Progress Bar 출력


dataset = ImageFolder(root=r"C:\Users\KETI\PycharmProjects\pythonProject1\tiny-imagenet-200\train",
                      transform=transforms.Compose([
                          transforms.Resize((224,224)),
                          transforms.ToTensor(), 
                      ]))

data_loader = DataLoader(dataset, 
                         batch_size=32, 
                         shuffle=True,
                         num_workers=0
                        )

images, labels = next(iter(data_loader))

labels_map = {v:k for k, v in dataset.class_to_idx.items()}

figure = plt.figure(figsize=(12, 8))
cols, rows = 8, 4

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(images), size=(1,)).item()
    img, label = images[sample_idx], labels[sample_idx].item()
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(torch.permute(img, (1, 2, 0)))

ratio = 0.8 

train_size = int(ratio * len(dataset))
test_size = len(dataset) - train_size
print(f'total: {len(dataset)}\ntrain_size: {train_size}\ntest_size: {test_size}')

train_data, test_data = random_split(dataset, [train_size, test_size])

batch_size = 32 
num_workers = 8 

train_loader = DataLoader(train_data, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0
                         )
test_loader = DataLoader(test_data, 
                         batch_size=batch_size,
                         shuffle=False, 
                         num_workers=0
                        )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name())

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(7*7*128, 200)
    
    def forward(self, x):
        x = self.sequential(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x      

model = CNNModel() 
model.to(device)   

optimizer = optim.Adam(model.parameters(), lr=0.0005)

loss_fn = nn.CrossEntropyLoss()

def model_train(model, data_loader, loss_fn, optimizer, device):
    model.train()
    
    running_loss = 0
    corr = 0
    
    prograss_bar = tqdm(data_loader)
    
    for img, lbl in prograss_bar:
        img, lbl = img.to(device), lbl.to(device)
        
        optimizer.zero_grad()
        
        output = model(img)
        
        loss = loss_fn(output, lbl)

        loss.backward()
        
        optimizer.step()
        _, pred = output.max(dim=1)
        
        corr += pred.eq(lbl).sum().item()
        
        running_loss += loss.item() * img.size(0)
        
    acc = corr / len(data_loader.dataset)
    
    return running_loss / len(data_loader.dataset), acc

def model_evaluate(model, data_loader, loss_fn, device):
    model.eval()
    
    with torch.no_grad():
        corr = 0
        running_loss = 0
        
        for img, lbl in data_loader:
            img, lbl = img.to(device), lbl.to(device)
            
            output = model(img)
            
            _, pred = output.max(dim=1)
            
            corr += torch.sum(pred.eq(lbl)).item()
            
            running_loss += loss_fn(output, lbl).item() * img.size(0)
        
        acc = corr / len(data_loader.dataset)
        
        return running_loss / len(data_loader.dataset), acc

num_epochs = 10

min_loss = np.inf

for epoch in range(num_epochs):
    train_loss, train_acc = model_train(model, train_loader, loss_fn, optimizer, device)

    val_loss, val_acc = model_evaluate(model, test_loader, loss_fn, device)   
    
    if val_loss < min_loss:
        print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
        min_loss = val_loss
        torch.save(model.state_dict(), 'DNNModel10.pth')
    
    print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')

model.load_state_dict(torch.load('DNNModel10.pth'))

final_loss, final_acc = model_evaluate(model, test_loader, loss_fn, device)
print(f'evaluation loss: {final_loss:.5f}, evaluation accuracy: {final_acc:.5f}')