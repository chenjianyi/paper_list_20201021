"""
create by: chenjianyi
create time: 2020.10.12
"""

import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))

import torch
import torch.nn as nn

from tiramisu import FCDenseNet67, FCDenseNet57, FCDenseNet_Tiny
from discriminator import Discriminator
from vgg import vgg19_bn

FCDenseNet = FCDenseNet_Tiny

class RIS_GAN(nn.Module):
    def __init__(self):
        super(RIS_GAN, self).__init__()
        self.residual_generator = FCDenseNet(in_channels=3, n_classes=3)
        self.removal_generator = FCDenseNet(in_channels=3, n_classes=3)
        self.illumination_generator = FCDenseNet(in_channels=3, n_classes=3)
        self.refinement_generator = FCDenseNet(in_channels=9, n_classes=3)

        self.discriminator = Discriminator(in_channel=3, out_channel=1)
        self.Loss = Loss()

    def forward(self, x, label=None, is_training_d=False):
        negative_residual = self.residual_generator(x)  # negative residual map, (bs, 3, h, w)
        coarse_result = self.removal_generator(x)   # coarse result, (bs, 3, h, w)
        inverse_illumination = self.illumination_generator(x)  # inverse illumination map, (bs, 3, h, w)
        x1 = torch.cat(((negative_residual + x), coarse_result, (inverse_illumination * x)), dim=1)  # bs * 9 * h * w
        refined_result = self.refinement_generator(x1)  # refined result, (bs, 3, h, w)

        if self.training and label is not None:
            ## discriminator output
            gt_negative_residual = label - x
            gt_inverse_illumination = label / (x + 1e-16)
            gt = label

            d_gt_negative_residual = self.discriminator(gt_negative_residual)
            d_negative_residual = self.discriminator(negative_residual)

            d_gt_inverse_illumination = self.discriminator(gt_inverse_illumination)
            d_inverse_illumination = self.discriminator(inverse_illumination)

            d_gt = self.discriminator(gt)
            d_refined_result = self.discriminator(refined_result)

            loss = self.Loss(x, negative_residual, inverse_illumination, coarse_result, refined_result, label, \
                             d_gt_negative_residual, d_negative_residual, d_gt_inverse_illumination, d_inverse_illumination, d_gt, d_refined_result, \
                             is_training_d=is_training_d)
            return loss
        else:
            return refined_result

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.vgg = vgg19_bn(pretrained=False)

    def forward(self, x, negative_residual, inverse_illumination, coarse_result, refined_result, label, \
                d_gt_negative_residual, d_negative_residual, d_gt_inverse_illumination, d_inverse_illumination, d_gt, d_refined_result, \
                lambda1=10, lambda2=100, lambda3=1, lambda4=1, beta1=0.1, beta2=0.2, \
                is_training_d=False):
        t_negative_residual = label - x
        t_inverse_illumination = label / (x + 1e-16)
        t = label

        if not is_training_d:
            ## Shadow removal loss
            L_vis = self.l1_loss(coarse_result, t) + self.l1_loss(refined_result, t)
            L_percept = self.l2_loss(self.vgg(coarse_result), self.vgg(t)) + self.l2_loss(self.vgg(refined_result), self.vgg(t))
            L_rem = L_vis + beta1 * L_percept

            ## Residual loss
            L_res = self.l1_loss(negative_residual, t_negative_residual)

            ## Illumination loss
            L_illum = self.l1_loss(inverse_illumination, t_inverse_illumination)

            ## Cross loss
            L_cross = self.l1_loss(negative_residual + x, t) + beta2 * self.l1_loss(t_inverse_illumination * x, t)

        ## Adversarial loss
        if is_training_d:
            L_adv = torch.log(d_gt) + torch.log(1 - d_refined_result) + \
                    torch.log(d_gt_negative_residual) + torch.log(1 - d_negative_residual) + \
                    torch.log(d_gt_inverse_illumination) + torch.log(1 - d_inverse_illumination)
            return L_adv
        else:
            L_adv = torch.log(d_refined_result) + torch.log(d_negative_residual) + torch.log(d_inverse_illumination)  # training generator
            Loss = lambda1 * L_rem + lambda2 * L_res + lambda3 * L_illum + lambda4 * L_cross + L_adv
            return Loss
