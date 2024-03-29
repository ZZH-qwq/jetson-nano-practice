
# This File is only for reference!!!!

import torch
from torch import nn

MODELRESH=32
MODELRESW=32

# #########################
# TODO: Read it for reference!
# This HzxModel is only for reference! 
# If you copy it, you will get no points!!
# #########################
class HzxModel(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.layers = []
        self.layers.append(nn.Conv2d(3, 32, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.BatchNorm2d(32))
        self.layers.append(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.BatchNorm2d(32))
        # [N, 32, 16, 16]
        self.layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.BatchNorm2d(64))
        # [N, 64, 8, 8]
        self.layers.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.BatchNorm2d(128))
        # [N, 64, 4, 4]
        self.layers.append(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.BatchNorm2d(256))
        # [N, 128, 2, 2]
        self.layers.append(nn.Conv2d(256, 1024, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Flatten())
        # [N, 256]
        self.layers.append(nn.Linear(1024, num_classes))
        self.layers.append(nn.Softmax(dim=-1))
        self.layers = nn.ModuleList(self.layers)

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
