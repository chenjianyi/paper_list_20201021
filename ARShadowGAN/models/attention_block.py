"""
create by: chenjianyi
create time: 2020.10.14 10:34
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import convolution2d, ResBlock, Interpolation

class AttentionBlock(nn.Module):
    def __init__(self, in_channels=3):
        super(AttentionBlock, self).__init__()
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

        self.rshadow_US1 = nn.Sequential(
            Interpolation(scale_factor=2),
            convolution2d(512, 256, kernel_size=3, stride=1, padding=2, dilation=2, act_fun='leakyrelu', bn=True, negative_slope=0.2),
        )
        self.rshadow_US2 = nn.Sequential(
            Interpolation(scale_factor=2),
            convolution2d(256, 128, kernel_size=3, stride=1, padding=2, dilation=2, act_fun='leakyrelu', bn=True, negative_slope=0.2),
        )
        self.rshadow_US3 = nn.Sequential(
            Interpolation(scale_factor=2),
            convolution2d(128, 64, kernel_size=3, stride=1, padding=2, dilation=2, act_fun='leakyrelu', bn=True, negative_slope=0.2),
        )
        self.rshadow_US4 = nn.Sequential(
            Interpolation(scale_factor=2),
            convolution2d(64, 1, kernel_size=3, stride=1, padding=2, dilation=2, act_fun='sigmoid', bn=True, negative_slope=0.2),
        )

        self.occluder_US1 = nn.Sequential(
            Interpolation(scale_factor=2),
            convolution2d(512, 256, kernel_size=3, stride=1, padding=2, dilation=2, act_fun='leakyrelu', bn=True, negative_slope=0.2),
        )
        self.occluder_US2 = nn.Sequential(
            Interpolation(scale_factor=2),
            convolution2d(256, 128, kernel_size=3, stride=1, padding=2, dilation=2, act_fun='leakyrelu', bn=True, negative_slope=0.2),
        )
        self.occluder_US3 = nn.Sequential(
            Interpolation(scale_factor=2),
            convolution2d(128, 64, kernel_size=3, stride=1, padding=2, dilation=2, act_fun='leakyrelu', bn=True, negative_slope=0.2),
        )
        self.occluder_US4 = nn.Sequential(
            Interpolation(scale_factor=2),
            convolution2d(64, 1, kernel_size=3, stride=1, padding=2, dilation=2, act_fun='sigmoid', bn=True, negative_slope=0.2),
        )

    def forward(self, x):
        ds1 = self.DS1(x)  # (bs, 64, 128, 128)
        ds2 = self.DS2(ds1) # (bs, 128, 64, 64)
        ds3 = self.DS3(ds2)  # (bs, 256, 32, 32)
        ds4 = self.DS4(ds3)  # (bs, 512, 16, 16)

        r_us1 = self.rshadow_US1(ds4)  # (bs, 256, 32, 32)
        r_us2 = self.rshadow_US2(r_us1 + ds3)  # (bs, 128, 64, 64)
        r_us3 = self.rshadow_US3(r_us2 + ds2)  # (bs, 64, 128, 128)
        r_attention = self.rshadow_US4(r_us3 + ds1)  # (bs, 1, 256, 256)

        o_us1 = self.occluder_US1(ds4)  # (bs, 256, 32, 32)
        o_us2 = self.occluder_US2(o_us1 + ds3)  # (bs, 128, 64, 64)
        o_us3 = self.occluder_US3(o_us2 + ds2)  # (bs, 64, 128, 128)
        o_attention = self.occluder_US4(o_us3 + ds1)  # (bs, 1, 256, 256)

        return r_attention, o_attention

if __name__ == '__main__':
    net = AttentionBlock(3)
    x = torch.randn((4, 3, 256, 256))
    net(x)
