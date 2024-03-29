
import torch
from torch import nn

MODELRESH = 32
MODELRESW = 32

# #########################
# TODO: Build your own model here!!!
# #########################


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class YourModel(nn.Module):
    def __init__(self, num_classes=100) -> None:  # origin num_classes=100
        super().__init__()
        self.layers = []
        # TODO: Build your own layers here!!!

        # ResNet architecture
        self.layers.append(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3))
        self.layers.append(nn.BatchNorm2d(64))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layers.append(ResidualBlock(64, 64))
        self.layers.append(ResidualBlock(64, 128, stride=2))
        # self.layers.append(ResidualBlock(128, 128))
        # self.layers.append(ResidualBlock(128, 256, stride=2))
        # self.layers.append(ResidualBlock(256, 256))

        self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(128, num_classes))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # print(x.shape)
        return x
