"""
create by: chenjianyi
create time: 2020.10.14 15:06
"""

import torch
import torch.nn as nn

class PixelDiscriminator(nn.Module):
    def __init__(self, in_channels, ndf=64, norm_layer=nn.InstanceNorm2d):
        super(PixelDiscriminator, self).__init__()
        use_bias = True

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=1, stride=1, padding=0),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 4, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(1),
            nn.Sigmoid(),
        )

        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, input):
        return self.pooling(self.net(input))

if __name__ == '__main__':
    net = PixelDiscriminator(in_channels=3)
    x = torch.randn((4, 3, 256, 256))
    print(net(x).size())
