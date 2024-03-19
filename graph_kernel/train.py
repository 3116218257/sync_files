import argparse
import os
import numpy as np
import math

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from tools import LARS, adjust_learning_rate, gather_from_all
from get_matrix import *
from video_sampler import *
from dataset import video_dataset



class NeuralEFCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.steps = 0

        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        if args.proj_dim[0] == 0:
            self.projector = nn.Identity()
        else:
            if len(args.proj_dim) > 1:
                sizes = [2048, ] + args.proj_dim
                layers = []
                for i in range(len(sizes) - 2):
                    layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                    if args.proj_bn:
                        layers.append(nn.BatchNorm1d(sizes[i + 1]))
                    layers.append(nn.ReLU(inplace=True))

                layers.append(nn.Linear(sizes[-2], sizes[-1], bias=True))

                self.projector = nn.Sequential(*layers)
            elif len(args.proj_dim) == 1:
                self.projector = nn.Linear(2048, args.proj_dim[0], bias=True)

        # if args.rank == 0 and hasattr(self, 'projector'):
        #     print(self.projector)


        self.training = args.is_train
        self.momentum = args.momentum
        self.register_buffer('eigennorm_sqr', torch.zeros(args.k))
        self.register_buffer('num_calls', torch.Tensor([0]))
        self.register_buffer('eigenvalues', None)


    def forward(self, y1, y2, ind1, ind2):
        self.steps += 1

        batch_size, frames, channels, row, col = y1.shape
        y1 = y1.view(batch_size * frames, channels, row, col)
        y2 = y2.view(batch_size * frames, channels, row, col)


        r1, r2 = self.backbone(torch.cat([y1, y2], 0)).chunk(2, dim=0)

        z1, z2 = self.projector(torch.cat([r1, r2], 0)).chunk(2, dim=0)
        
        psi1 = gather_from_all(z1)
        psi2 = gather_from_all(z2)
        
        if self.training:
            norm_ = (psi1.norm(dim=0) ** 2 + psi2.norm(dim=0) ** 2) / (psi1.shape[0] + psi2.shape[0])
            with torch.no_grad():
                if self.num_calls == 0:
                    self.eigennorm_sqr.copy_(norm_.data)
                else:
                    self.eigennorm_sqr.mul_(self.momentum).add_(norm_.data, alpha = 1-self.momentum)
                self.num_calls += 1
            norm_ = norm_.sqrt()
        else:
            norm_ = self.eigennorm_sqr.sqrt()

        psi1 = z1
        psi2 = z2

        # psi1 = psi1.div(psi1.norm(dim=0).clamp(min=1e-6)) * math.sqrt(self.args.t)
        # psi2 = psi2.div(psi2.norm(dim=0).clamp(min=1e-6)) * math.sqrt(self.args.t)
        psi1 = psi1 / norm_ * math.sqrt(self.args.t)
        psi2 = psi2 / norm_ * math.sqrt(self.args.t)

        psi1 = psi1.view(batch_size, frames, args.proj_dim[-1])
        psi2 = psi2.view(batch_size, frames, args.proj_dim[-1])
        
        K = get_K(ind1, ind2, args.coeff)
        K = K.to(args.device)

        '''
        here,
        psi is [batch, n_frames, k]
        K is [batch, n_frames, n_frames]
        so:
        psi_K_psi is [batch, k, k]
        
        '''
        psi_K_psi = torch.transpose(psi2, 1, 2) @ K @ psi1

        psi_K_psi = torch.sum(psi_K_psi, dim=0)

        B = y1.shape[0]
        with torch.no_grad():
            if self.eigenvalues is None:
                self.eigenvalues = psi_K_psi.diag() / (B**2)
            else:
                self.eigenvalues.mul_(0.9).add_(psi_K_psi.diag() / (B**2), alpha = 0.1)

        
        if self.steps % 20 == 1:
            mappp = psi_K_psi.detach().cpu()
            mappp = mappp.numpy()
            plt.imshow(np.array(mappp), cmap='hot')
            plt.colorbar()
            plt.savefig(f'./hot.png')
            plt.clf()
            
            mappp = K[0].cpu()
            mappp = mappp.numpy()
            plt.imshow(np.array(mappp), cmap='hot')
            plt.colorbar()
            plt.savefig(f'./origin_k.png')
            plt.clf()
            
            mappp = recover_k(psi1=psi1.view(frames, args.proj_dim[-1]),
                              psi2=psi2.view(frames, args.proj_dim[-1]),
                                             eigen_value=self.eigenvalues)
            mappp = mappp.numpy()
            plt.imshow(np.array(mappp), cmap='hot')
            plt.colorbar()
            plt.savefig(f'./recover_k.png')
            plt.clf()
        
        psi_K_psi_diag = torch.diagonal(psi_K_psi)
        
        psi2_d_K_psi1 = torch.transpose(psi2.detach(), 1, 2) @ K @ psi1
        psi1_d_K_psi2 = torch.transpose(psi1.detach(), 1, 2) @ K @ psi2

        psi2_d_K_psi1 = torch.sum(psi2_d_K_psi1, dim=0)
        psi1_d_K_psi2 = torch.sum(psi1_d_K_psi2, dim=0)

        loss = - psi_K_psi_diag.sum() * 2
        reg = ((psi2_d_K_psi1) ** 2).triu(1).sum() \
                + ((psi1_d_K_psi2) ** 2).triu(1).sum()

        loss /= psi_K_psi_diag.numel()
        reg /= psi_K_psi_diag.numel()

        return loss, reg,


ckpt_path = './ckpt/'

if __name__ == '__main__':
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    parser = argparse.ArgumentParser(description='Learn eigenfunctions on UCF50')
    parser.add_argument('--device', default=2)

    # opt configs
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.06)
    parser.add_argument('--min_lr', type=float, default=None, metavar='LR')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=1000, metavar='N', help='iterations to warmup LR')
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--n_frames', type=int, default=64)

    # for neuralef
    parser.add_argument('--alpha', default=0.4, type=float)
    # parser.add_argument('--k', default=64, type=int)
    parser.add_argument('--momentum', default=.9, type=float)
    parser.add_argument('--not_all_together', default=True, action='store_true')
    parser.add_argument('--proj_dim', default=[2048, 64], type=int, nargs='+')
    parser.add_argument('--no_proj_bn', default=False, action='store_true')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--t', default=10, type=float)
    parser.add_argument('--k', default=64, type=float)
    parser.add_argument('--coeff', default=1.0, type=float)
    parser.add_argument('--is_train', default=True, type=bool)

    args = parser.parse_args()

    args.proj_bn = not args.no_proj_bn
    if args.min_lr is None:
        args.min_lr = args.lr * 0.001

    device = f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    device = torch.device(device)
    torch.backends.cudnn.benchmark = True

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    video_data = video_dataset('./data/test_data', args.n_frames)
    data_loader = DataLoader(video_data, batch_size=args.batch_size, shuffle=True)


    model = NeuralEFCLR(args)
    model.to(device)

    # optimizer = LARS(model.parameters(),
    #                  lr=0, weight_decay=args.weight_decay,
    #                  weight_decay_filter=exclude_bias_and_norm,
    #                  lars_adaptation_filter=exclude_bias_and_norm)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    losses, regs, n_steps = 0, 0, 0

    loss_his = []
    reg_his = []

    pretrian_log = open('pretrain_log.txt', 'w')

    for epoch in range(args.num_epochs):
        model.train()
        for video1, video2, ind1, ind2 in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", unit="batch"):
            n_steps += 1
            # lr = args.lr
            lr = adjust_learning_rate(args, optimizer, data_loader, n_steps)

            optimizer.zero_grad()
            video1 = video1.to(device)
            video2 = video2.to(device)
            ind1 = ind1.to(device)
            ind2 = ind2.to(device)

            loss, reg = model.forward(video1, video2, ind1, ind2)

            (loss + reg * args.alpha).backward()

            optimizer.step()

            losses += loss.item()
            regs += reg.item()

            if n_steps % 20 == 19:
                print(f'epoch={epoch + 1}, loss={loss.item():.6f},'
                      f' reg={reg.item():.6f},'
                      f' learning_rate={lr:.4f},')
                print(f'epoch={epoch + 1}, loss={loss.item():.6f},'
                      f' reg={reg.item():.6f},'
                      f' learning_rate={lr:.4f},', file=pretrian_log)

        loss_his.append(1.0 * losses / n_steps)
        reg_his.append(1.0 * regs / n_steps)
        regs = 0
        losses = 0
        n_steps = 0

        if epoch % 200 == 1:
            #######save backbone#########
            torch.save(dict(model=model.state_dict(),
                            eigenvalues=model.eigenvalues,
                            eigennorm=model.eigennorm_sqr),
                              ckpt_path + f'ckpt{epoch}.pth')

  

    if not os.path.exists("./figs"):
        os.mkdir("./figs")

    plt.plot(loss_his, color='blue')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.savefig('./figs/loss.png')
    plt.clf()

    plt.plot(reg_his, color='green')
    plt.xlabel('step')
    plt.ylabel('reg')
    plt.savefig('./figs/reg.png')
    plt.clf()

