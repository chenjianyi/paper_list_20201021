"""
create by: chenjianyi
create time: 2020.10.14 10:51
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class convolution2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, \
                 groups=1, act_fun='relu', sn=False, bn=False, bias=True, conv_first=True, transpose=False, **kwargs):
        super(convolution2d, self).__init__()
        self.conv_first = conv_first
        if not transpose:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, \
                                  dilation=dilation, groups=groups, bias=bias)
        else:
            output_padding = kwargs.get('output_padding') if kwargs.get('output_padding') else 0
            self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, \
                                           output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)

        eps = kwargs.get('eps') if kwargs.get('eps') else 1e-5
        momentum = kwargs.get('momentum') if kwargs.get('momentum') else 0.1
        affine = kwargs.get('affine') if kwargs.get('affine') else True
        track_running_stats = kwargs.get('track_running_stats') if kwargs.get('track_running_stats') else True
        out_planes = out_planes if conv_first else in_planes
        self.bn = nn.BatchNorm2d(out_planes, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats) if bn else nn.Sequential()

        self.act_f = ActiveFun(act_fun, **kwargs)

    def forward(self, x):
        if self.conv_first:
            x = self.conv(x)
            x = self.bn(x)
            x = self.act_f(x)
        else:
            x = self.bn(x)
            x = self.act_f(x)
            x = self.conv(x)
        return x

class ActiveFun(nn.Module):
    def __init__(self, act_fun, concat=False, **kwargs):
        super(ActiveFun, self).__init__()
        self.concat = concat
        if act_fun.lower() == 'relu':
            inplace = kwargs.get('inpace') if kwargs.get('inpace') else True
            self.act_f = nn.ReLU(inplace)
        elif act_fun.lower() == 'leakyrelu':
            negative_slope = kwargs.get('negative_slope') if kwargs.get('negative_slope') else 0.01
            inplace = kwargs.get('inpace') if kwargs.get('inpace') else True
            self.act_f = nn.LeakyReLU(negative_slope, inplace)
        elif act_fun.lower() == 'relu6':
            inplace = kwargs.get('inpace') if kwargs.get('inpace') else True
            self.act_f = nn.ReLU6(inplace)
        elif act_fun.lower() == 'rrelu':
            inplace = kwargs.get('inpace') if kwargs.get('inpace') else True
            lower = kwargs.get('lower') if kwargs.get('lower') else 0.125
            upper = kwargs.get('upper') if kwargs.get('upper') else 0.3333333333333333
            self.act_f = nn.RReLU(lower=lower, upper=upper, inplace=inplace)
        elif act_fun.lower() == 'elu':
            inplace = kwargs.get('inpace') if kwargs.get('inpace') else True
            alpha = kwargs.get('alpha') if kwargs.get('alpha') else 1.0
            self.act_f = nn.ELU(alpha=alpha, inplace=inplace)
        elif act_fun.lower() == 'selu':
            inplace = kwargs.get('inpace') if kwargs.get('inpace') else True
            self.act_f = nn.SELU(inplace=inplace)
        elif act_fun.lower() == 'celu':
            inplace = kwargs.get('inpace') if kwargs.get('inpace') else True
            alpha = kwargs.get('alpha') if kwargs.get('alpha') else 1.0
            self.act_f = nn.CELU(alpha=alpha, inplace=inplace)
        elif act_fun.lower() == 'sigmoid':
            self.act_f = nn.Sigmoid()
        elif act_fun.lower() == 'softplus':
            beta = kwargs.get('beta') if kwargs.get('beta') else 1
            threshold = kwargs.get('threshold') if kwargs.get('threshold') else 20
            self.act_f = nn.Softplus(beta=beta, threshold=threshold)
        elif act_fun.lower() == 'softmax':
            dim = kwargs.get('dim') if kwargs.get('dim') else None
            self.act_f = nn.Softmax(dim=dim)
        elif act_fun.lower() == 'none':
            self.act_f = nn.Sequential()
        else:
            raise ValueError()

    def forward(self, x):
        if self.concat:
            x = torch.cat([x, -x], dim=1)
        x = self.act_f(x)
        return x

class ResBlock(nn.Module):
    expansion = 1
    BN_MOMENTUM = 0.1
    def __init__(self, inplanes, planes, stride=1, downsample=None, bias=False, sn=False, bn=True, act_fun='relu', **kwargs):
        super(ResBlock, self).__init__()
        self.conv1 = convolution2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=bias, sn=sn, bn=bn, act_fun=act_fun, **kwargs)
        self.conv2 = convolution2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun, **kwargs)
        self.conv3 = convolution2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=bias, sn=sn, bn=bn, act_fun='none', **kwargs)

        self.downsample = downsample
        if stride != 1 or inplanes != self.expansion * planes:
            self.downsample = convolution2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, padding=0, bias=bias, sn=sn, bn=bn, act_fun='none', **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #out = self.relu(out)
        return out

class Interpolation(nn.Module):
    def __init__(self, **kwargs):
        super(Interpolation, self).__init__()
        self.kwargs = kwargs

    def forward(self, x):
        return F.interpolate(x, **self.kwargs)
