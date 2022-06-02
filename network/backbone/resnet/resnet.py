import torch
import torch.nn as nn

from torch.utils import load_state_dict_from_url

def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size = 3,
        stride = stride,
        padding = dilation,
        groups = groups,
        bias = False,
        dilation = dilation,
    )

def conv1x1(in_planes, out_planes, stride = 1):
    return nn.Conv2d(
        in_planes, 
        out_planes, 
        kernel_size = 1, 
        stride = stride, 
        bias = False
    )

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
        self.bn1 = norm_layer(inplanes, planes, stride)
        self.relu = nn.ReLU(inplane=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out
        
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
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3{out}
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out
    
class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classed = 1000,
        zero_init_resifual = False,
        groups = 1,
        width_per_group = 64,
        replace_stride_with_dilation = None,
        norm_layer = None,
    ):
        super(ResNet, self).__init__()
        
