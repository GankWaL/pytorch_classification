import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from utils import conv3x3, conv1x1, getPadding

class Conv2D_BN(nn.Module):
    def __init__(
        self,
        inplanes,
        activation,
        planes,
        kernel_size,
        stride = 1,
        padding = 'same',
        groups = 1,
        dilation = 1,
        bias = False
    ):
        super(Conv2D_BN, self).__init()
        self.padding = getPadding(kernel_size, padding)
        self.activation = activation
        inplanes = math.floor(inplanes)
        planes = math.floor(planes)
        self.conv_layer = nn.Conv2d(inplanes, planes, kernel_size, stride = stride, padding = self.padding, dilation = dilation, groups = groups, bias = bias)
        self.bn_layer = nn.BatchNorm2d(planes)
        
    def forward(self, input):
        output = self.conv_layer(input)
        output = self.bn_layer(output)
        if self.activation != None:
            output = self.activation(output)
        return output


class BasicBlock(nn.Module):
    expansion: int = 1
    
    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("기본 블럭은 오직 groups=1 과 base_width=64을 지원합니다.")
        if dilation > 1:
            raise NotImplementedError("dilation이 1보다 크면 기본 블럭에서 지원하지 않습니다.")
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, input):
        identity = input
        
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)
        
        output = self.conv2(output)
        output = self.bn2(output)
        
        if self.downsample is not None:
            identity = self.downsample(input)
            
        output += identity
        output = self.relu(output)
        
        return output
    
class BottleNeck(nn.Module):
    expansion: int = 4
    
    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 1,
        dilation = 1,
        norm_layer = None,
    ):
        super(BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, input):
        identity = input
        
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)
        
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)
        
        output = self.conv3(output)
        output = self.bn3(output)
        
        if self.downsample is not None:
            identity = self.downsample(input)
            
        out += identity
        out = self.relu(output)
        
        return output