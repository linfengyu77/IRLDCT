import sys
sys.path.append('./dip-ct-benchmark')
from torch import nn
import torch.nn.functional as F
from layer import *


class CNN(nn.Module):
    def __init__(self, layer_num, ks):
        super(CNN, self).__init__()
        kernel_size = 3
        padding  = 'same'
        features = ks
        layers   = []
        channels = 1
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))
        for _ in range(layer_num-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.cnn = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        y = self.cnn(x)
        return y

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, learning_type="plain", norm="bnorm", nblk=8):
        super(ResNet, self).__init__()

        self.learning_type = learning_type

        self.enc = CBR2d(in_channels, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=None, relu=0.1)

        res = []
        for i in range(nblk):
            res += [ResBlock(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.1)]
        self.res = nn.Sequential(*res)

        self.dec = CBR2d(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.1)

        self.fc = CBR2d(nker, out_channels, kernel_size=1, stride=1, padding=0, bias=True, norm=None, relu=None)

    def forward(self, x):
        x0 = x

        x = self.enc(x)
        x = self.res(x)
        x = self.dec(x)

        if self.learning_type == "plain":
            x = self.fc(x)
        elif self.learning_type == "residual":
            x = x0 + self.fc(x)

        return x


