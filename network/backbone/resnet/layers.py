
import math
import torch.nn as nn
from utils import getPadding

class Conv2D_BN(nn.Module):
    def __init__(self, inplanes, activation, planes, kernel_size, stride=1, padding='same', groups=1, dilation=1, bias=False):
        super().__init__()
        self.padding = getPadding(kernel_size, padding)
        self.activation = activation
        inplanes = math.floor(inplanes)
        planes = math.floor(planes)
        self.conv_layer = nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=self.padding, dilation=dilation, groups=groups, bias=bias)
        self.batchNorm_layer = nn.BatchNorm2d(planes)

    def forward(self, input):
        output = self.conv_layer(input)
        output = self.batchNorm_layer(output)
        if self.activation != None:
            output = self.activation(output)
        return output