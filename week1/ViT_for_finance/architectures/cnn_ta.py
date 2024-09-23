import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=65, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(256 * 65, 128) 
        self.fc2 = nn.Linear(128, 10)

        self.batch_norm1 = nn.BatchNorm2d(128)
        self.batch_norm2 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        
        if x.size(2) > 1 and x.size(3) > 1: 
            x = self.pool(x)
        
        x = F.relu(self.batch_norm2(self.conv2(x)))
        
        if x.size(2) > 1 and x.size(3) > 1:
            x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x