import torch.nn as nn
import torch
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys
import matplotlib.pyplot as plt
from torchvision import models

###### Origin Dataset #####
PATH = (r'C:\Users\KETI\PycharmProjects\pythonProject1\DNNModel.pth')
imgPath = r"C:\Users\KETI\PycharmProjects\pythonProject1\tiny-imagenet-200\val"
imgList = os.listdir(imgPath)
imgList = [os.path.join(imgPath, file) for file in imgList]
imgList = [file for file in imgList if file.endswith(".jpg") or file.endswith(".png")]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        x = self.sequential(x)      # (7, 7, 128)
        x = torch.flatten(x, 1)     # (7 x 7 x 128, 1)
        x = self.fc(x)              # (200 x 1) 
        return x  

fc = nn.Sequential(
    nn.Linear(7*7*512, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

model = models.vgg16(pretrained=True)
model.classifier = fc
model.to(device)

dataset = ImageFolder(root=imgPath,
                        transform=transforms.Compose([
                        transforms.Resize((224,224)),  
                        transforms.ToTensor()])
                        )

data_loader = DataLoader(dataset, 
                         #batch_size=32, 
                         batch_size=4, 
                         shuffle=False,
                         num_workers=0
                        )

from tqdm import tqdm

class_names = dataset.classes

model.load_state_dict(torch.load(PATH))
model.to(device)

def model_evaluate(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        corr = 0
        for cnt, (img, lbl) in enumerate(tqdm(data_loader)):
            img, lbl = img.to(device), lbl.to(device)
            output = model(img)
            _, pred = output.max(dim=1)
            corr += torch.sum(pred.eq(lbl)).item()

        acc = corr / len(data_loader.dataset)
        f = open('test-cifar10-da.txt', 'w')
#        f.write(str(acc).zfill(5))
        f.write(str(acc))
        f.close()
        #print(acc)
        sys.stdout.close()
        return acc

test_acc = model_evaluate(model, data_loader, device)
#print(f"test_acc: {test_acc}")
#print("test_acc: {" + str(test_acc) + "}")
print(test_acc)