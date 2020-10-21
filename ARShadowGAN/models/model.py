"""
create by: chenjianyi
create time: 2020.10.14 14:53
"""

import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))

import torch
import torch.nn as nn

from attention_block import AttentionBlock
from virtual_shadow_generator import VirtualShadowGenerator
from discriminator import PixelDiscriminator
from vgg import vgg16_bn

class ARShadowGAN(nn.Module):
    def __init__(self):
        super(ARShadowGAN, self).__init__()
        self.attention = AttentionBlock(in_channels=4)
        self.generator = VirtualShadowGenerator(in_channels=6)
        self.discriminator = PixelDiscriminator(in_channels=7)
        self.vgg = vgg16_bn(pretrained=False)

        self.mse_loss = nn.MSELoss()

    def forward(self, x, m, M_rshadow=None, M_obj=None, y=None, training_first_stage=True, is_training_d=False, \
               beta1=10, beta2=1, beta3=0.01):
        att_shadow, att_obj = self.attention(torch.cat((x, m), dim=1))
        if self.training:  ### training stage
            if training_first_stage:  ## training first stage, only attention loss
                assert M_rshadow is not None and M_obj is not None
                L_attn = self.mse_loss(att_shadow, M_rshadow) + self.mse_loss(att_obj, M_obj)
                return L_attn
            else:   ## training second stage
                assert y is not None
                coarse_shadow, refined_shadow = self.generator(torch.cat((x, m, att_shadow, att_obj), dim=1))
                d_real = self.discriminator(torch.cat((x, m, y), dim=1))
                d_fake = self.discriminator(torch.cat((x, m, refined_shadow + x), dim=1))
                if is_training_d:  # training discrinator
                    L_adv = torch.log(d_real + 1e-16) + torch.log(1 - d_fake + 1e-16)
                    return L_adv
                else:   # training generator
                    L_2 = self.mse_loss(coarse_shadow + x, y) + self.mse_loss(refined_shadow + x, y)
                    L_per = self.mse_loss(self.vgg(coarse_shadow + x), self.vgg(y)) + self.mse_loss(self.vgg(refined_shadow + x), self.vgg(y))
                    L_adv = torch.log(d_fake + 1e-16)  # fake should labeled as real
                    Loss = beta1 * L_2 + beta2 * L_per + beta3 * L_adv
                    return Loss

        else:  ### test stage
            coarse_shadow, refined_shadow = self.generator(torch.cat((x, m, att_shadow, att_obj), dim=1))
            return refined_shadow + x
