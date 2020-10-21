"""
create by: chenjianyi
create time: 2020.10.14 14:34
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import convolution2d, ResBlock, Interpolation

class VirtualShadowGenerator(nn.Module):
    def __init__(self, in_channels=3):
        super(VirtualShadowGenerator, self).__init__()
        self.DS1 = nn.Sequential(
            ResBlock(in_channels, planes=64, stride=1, downsample=None, bias=False, bn=True, act_fun='leakyrelu', negative_slope=0.2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.AvgPool2d(kernel_size=2),
        )
        self.DS2 = nn.Sequential(
            ResBlock(64, planes=128, stride=1, downsample=None, bias=False, bn=True, act_fun='leakyrelu', negative_slope=0.2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.AvgPool2d(kernel_size=2),
        )
        self.DS3 = nn.Sequential(
            ResBlock(128, planes=256, stride=1, downsample=None, bias=False, bn=True, act_fun='leakyrelu', negative_slope=0.2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.AvgPool2d(kernel_size=2),
        )
        self.DS4 = nn.Sequential(
            ResBlock(256, planes=512, stride=1, downsample=None, bias=False, bn=True, act_fun='leakyrelu', negative_slope=0.2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.AvgPool2d(kernel_size=2),
        )

        self.DS5 = nn.Sequential(
            ResBlock(512, planes=1024, stride=1, downsample=None, bias=False, bn=True, act_fun='leakyrelu', negative_slope=0.2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.AvgPool2d(kernel_size=2),
        )

        self.US1 = nn.Sequential(
            Interpolation(scale_factor=2),
            convolution2d(1024, 512, kernel_size=3, stride=1, padding=1, dilation=1, act_fun='leakyrelu', bn=True, negative_slope=0.2),
        )
        self.US2 = nn.Sequential(
            Interpolation(scale_factor=2),
            convolution2d(512, 256, kernel_size=3, stride=1, padding=1, dilation=1, act_fun='leakyrelu', bn=True, negative_slope=0.2),
        )
        self.US3 = nn.Sequential(
            Interpolation(scale_factor=2),
            convolution2d(256, 128, kernel_size=3, stride=1, padding=1, dilation=1, act_fun='leakyrelu', bn=True, negative_slope=0.2),
        )
        self.US4 = nn.Sequential(
            Interpolation(scale_factor=2),
            convolution2d(128, 64, kernel_size=3, stride=1, padding=1, dilation=1, act_fun='leakyrelu', bn=True, negative_slope=0.2),
        )
        self.US5 = nn.Sequential(
            Interpolation(scale_factor=2),
            convolution2d(64, 3, kernel_size=3, stride=1, padding=1, dilation=1, act_fun='sigmoid', bn=True, negative_slope=0.2),
        )

        self.Refinement = nn.Sequential(
            convolution2d(3, 64, kernel_size=3, stride=1, padding=1, dilation=1, act_fun='leakyrelu', bn=True, negative_slope=0.2),
            convolution2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, act_fun='leakyrelu', bn=True, negative_slope=0.2),
            convolution2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, act_fun='leakyrelu', bn=True, negative_slope=0.2),
            convolution2d(64, 3, kernel_size=3, stride=1, padding=1, dilation=1, act_fun='leakyrelu', bn=True, negative_slope=0.2),
        )

    def forward(self, x):
        ds1 = self.DS1(x)  # (bs, 64, 128, 128)
        ds2 = self.DS2(ds1) # (bs, 128, 64, 64)
        ds3 = self.DS3(ds2)  # (bs, 256, 32, 32)
        ds4 = self.DS4(ds3)  # (bs, 512, 16, 16)
        ds5 = self.DS5(ds4)  # (bs, 1024, 8, 8)

        us1 = self.US1(ds5)  #(bs, 512, 16, 16)
        us2 = self.US2(ds4 + us1)  # (bs, 256, 32, 32)
        us3 = self.US3(us2 + ds3)  # (bs, 128, 64, 64)
        us4 = self.US4(us3 + ds2)  # (bs, 64, 128, 128)
        coarse_residual_shadow = self.US5(us4 + ds1)  # (bs, 1, 256, 256)

        refined_residual_shadow = self.Refinement(coarse_residual_shadow)

        return coarse_residual_shadow, refined_residual_shadow

if __name__ == '__main__':
    net = VirtualShadowGenerator(3)
    x = torch.randn((4, 3, 256, 256))
    a, b = net(x)
    print(a.size(), b.size())
