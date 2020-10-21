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
        m = torch.randn(1, 256, 256)
        M_rshadow = torch.randn(1, 256, 256)
        M_obj = torch.randn(1, 256, 256)
        y = torch.randn(3, 256, 256)
        return x, m, M_rshadow, M_obj, y
        

    def __len__(self):
        return len(self.files)
