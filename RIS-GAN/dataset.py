"""
create by: chenjianyi
create time: 2020.10.14 19:21
"""

import os
import torch

class VirtualDataset():
    def __init__(self, train_file=None, _type='train', img_size=256):
        self.files = [0] * 1000
        pass

    def __getitem__(self, index):
        x = torch.randn(3, 256, 256)
        y = torch.randn(3, 256, 256)
        return x, y
        

    def __len__(self):
        return len(self.files)
