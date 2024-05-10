
import torch
from torch import nn

MODELRESH = 32
MODELRESW = 32

# #########################
# TODO: Build your own model here!!!
# #########################


class YourModel(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.layers = []
        self.layers.append(nn.Conv2d(3, 32, kernel_size=3, padding=1))
        self.layers.append(nn.BatchNorm2d(32))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.BatchNorm2d(32))
        self.layers.append(nn.ReLU(inplace=True))
        # [N, 32, 16, 16]
        self.layers.append(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.BatchNorm2d(64))
        self.layers.append(nn.ReLU(inplace=True))
        # [N, 64, 8, 8]
        self.layers.append(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.BatchNorm2d(128))
        self.layers.append(nn.ReLU(inplace=True))
        # [N, 64, 4, 4]
        self.layers.append(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.BatchNorm2d(256))
        self.layers.append(nn.ReLU(inplace=True))
        # [N, 128, 2, 2]
        self.layers.append(
            nn.Conv2d(256, 1024, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Flatten())
        # [N, 256]
        self.layers.append(nn.Linear(1024, num_classes))
        self.layers.append(nn.Softmax(dim=-1))
        self.layers = nn.ModuleList(self.layers)

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
