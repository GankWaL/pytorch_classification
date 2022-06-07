import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
def getPadding(kernel_size, mode = 'same'):
    if mode == 'same':
        return (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))