import torch
from torch import nn

MODELRESH = 32
MODELRESW = 32


# #########################
# TODO: Build your own model here!!!
# #########################
class YourModel(nn.Module):
    def __init__(self, num_classes=100) -> None:
        super().__init__()
        self.layers = []
        # 1
        self.layers.append(nn.Conv2d(3, 32, kernel_size=5))
        self.layers.append(nn.ReLU(inplace=True))

        # 2
        self.layers.append(nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=1))
        self.layers.append(nn.BatchNorm2d(32))
        self.layers.append(nn.ReLU(inplace=True))

        # 3
        self.layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.BatchNorm2d(64))
        self.layers.append(nn.ReLU(inplace=True))

        # 4
        self.layers.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.BatchNorm2d(128))
        self.layers.append(nn.LeakyReLU(inplace=True))

        # 5 flatten and liner out
        self.layers.append(nn.MaxPool2d(kernel_size=2))
        self.layers.append(nn.BatchNorm2d(128))
        self.layers.append(nn.Flatten())

        self.layers.append(nn.Linear(512, num_classes))
        self.layers.append(nn.Softmax(dim=-1))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # print(x.shape)
        return x