import torch.nn as nn
import torch
from torch import nn
import torch.nn.functional as F

MODELRESH = 32
MODELRESW = 32

# #########################
# TODO: Build your own model here!!!
# #########################


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class YourModel(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.layers = []
        self._make_layers(cfg['VGG11'])
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(512, num_classes))
        self.layers = nn.ModuleList(self.layers)

    def _make_layers(self, cfg):
        in_channels = 3
        for x in cfg:
            if x == 'M':
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                self.layers.append(
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1))
                self.layers.append(nn.BatchNorm2d(x))
                self.layers.append(nn.ReLU(inplace=True))
                in_channels = x
        self.layers.append(nn.AvgPool2d(kernel_size=1, stride=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def fuse_model(self):
        fuse_modules = torch.ao.quantization.fuse_modules
        for idx in range(len(self.layers)):
            if isinstance(self.layers[idx], nn.Conv2d):
                if idx+1 < len(self.layers) \
                    and (isinstance(self.layers[idx+1], nn.ReLU)
                         or isinstance(self.layers[idx+1], nn.BatchNorm2d)):
                    if idx+2 < len(self.layers) \
                        and (isinstance(self.layers[idx+2], nn.ReLU)
                             or isinstance(self.layers[idx+2], nn.BatchNorm2d)):
                        fuse_modules(self.layers, [str(idx), str(
                            idx+1), str(idx+2)], inplace=True)
                    else:
                        fuse_modules(
                            self.layers, [str(idx), str(idx+1)], inplace=True)
