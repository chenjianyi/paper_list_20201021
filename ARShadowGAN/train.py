"""
create by: chenjianyi
create time: 2020.10.14 17:01
"""

import os
import argparse
import warnings

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist

from models.model import ARShadowGAN
from dataset import VirtualDataset

def main_worker(local_rank, ngpus_per_node, args):
    ### some initial
    warnings.filterwarnings('ignore')
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:22365', world_size=ngpus_per_node, rank=local_rank)
    torch.cuda.set_device(local_rank)

    ### Dataset
    train_dataset = VirtualDataset(args.train_file, _type='train', img_size=args.img_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, pin_memory=True, num_workers=8)

    ### Model
    net = ARShadowGAN()
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net).to(local_rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    net.train()

    ### Optimizer
    att_optimizer = torch.optim.Adam(net.module.attention.parameters(), lr=1e-15, betas=(0.9, 0.99))
    
    gen_d_optimizer = torch.optim.Adam(net.module.discriminator.parameters(), lr=1e-15, betas=(0.9, 0.99))
    gen_g_optimizer = torch.optim.Adam(net.module.generator.parameters(), lr=1e-15, betas=(0.9, 0.99))

    ### Train
    for epoch in range(args.epochs):
        for i, (x, m, M_rshadow, M_obj, y) in enumerate(train_loader):
            x = x.to(local_rank, non_blocking=True)
            m = m.to(local_rank, non_blocking=True)
            M_rshadow = M_rshadow.to(local_rank, non_blocking=True)
            M_obj = M_obj.to(local_rank, non_blocking=True)
            att_optimizer.zero_grad()
            y = y.cuda()
            if args.training_first_stage:  # fisrt stage
                loss = net(x, m, M_rshadow, M_obj, training_first_stage=True)
                loss.backward()
                att_optimizer.step()
                print('[epoch: %d] [step: %d/%d] [loss: %f]' % (epoch, i, len(train_loader), loss.item()))
            else:  # sencond stage
                gen_g_optimizer.zero_grad()
                loss_g = net(x, m, M_rshadow, M_obj, y, training_first_stage=False, is_training_d=False)
                gen_g_optimizer.step()
                gen_d_optimizer.zero_grad()
                loss_d = net(x, m, M_rshadow, M_obj, y, training_first_stage=False, is_training_d=True)
                gen_d_optimizer.step()
                print('[epoch: %d] [step: %d/%d] [g_loss: %f, d_loss: %f]' % (epoch, i, len(train_loader), loss_g.item(), loss_d.item()))
        torch.save('ARShadowGan_%d.pth' % epoch, net.module.state_dict())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0,1')
    parser.add_argument('--train_file', type=str, default='train.txt')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--training_first_stage', type=int, default=1)

    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    ngpus_per_node = len(opt.gpu.split(','))

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
