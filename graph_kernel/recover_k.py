import argparse
import os
import numpy as np
import math

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from tools import LARS, adjust_learning_rate
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
        self.register_buffer('eigennorm', torch.zeros(args.k))
        self.register_buffer('num_calls', torch.Tensor([0]))
        self.register_buffer('eigenvalues', None)


    def forward(self, y1, y2, ind1, ind2):
        self.steps += 1

        batch_size, frames, channels, row, col = y1.shape
        y1 = y1.view(batch_size * frames, channels, row, col)
        y2 = y2.view(batch_size * frames, channels, row, col)

        r1 = self.backbone(y1)
        r2 = self.backbone(y2)

        z1 = self.projector(r1)
        z2 = self.projector(r2)
        
        if self.training:
            norm_ = z1.norm(dim=0) / math.sqrt(np.prod([z1.shape[0]]))
            with torch.no_grad():
                if self.num_calls == 0:
                    self.eigennorm.copy_(norm_.data)
                else:
                    self.eigennorm.mul_(self.momentum).add_(norm_.data, alpha = 1-self.momentum)
                self.num_calls += 1
        else:
            norm_ = self.eigennorm

        norm_ = norm_.to(y1.device)
        psi1 = z1
        psi2 = z2

        # psi1 = psi1.div(psi1.norm(dim=0).clamp(min=1e-6)) * math.sqrt(self.args.t)
        # psi2 = psi2.div(psi2.norm(dim=0).clamp(min=1e-6)) * math.sqrt(self.args.t)
        psi1 = psi1 / norm_ * math.sqrt(self.args.t)
        psi2 = psi2 / norm_ * math.sqrt(self.args.t)

        # psi1 = psi1.view(batch_size, frames, args.proj_dim[-1])
        # psi2 = psi2.view(batch_size, frames, args.proj_dim[-1])
        
        K = get_K(ind1, ind2, args.coeff)
        K = K.to(args.device)

        '''
        here,
        psi is [batch, n_frames, k]
        K is [batch, n_frames, n_frames]
        so:
        psi_K_psi is [batch, k, k]
        
        '''
        psi1 = psi1.view(frames, args.proj_dim[-1])
        psi2 = psi2.view(frames, args.proj_dim[-1])
        K = K.view(frames, frames)
 
        k_recover = recover_k(psi1=psi1, psi2=psi2, eigen_value=self.eigenvalues.to(y1.device),)
        return k_recover


ckpt_path = './ckpt/'

if __name__ == '__main__':
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    parser = argparse.ArgumentParser(description='Learn eigenfunctions on UCF50')
    parser.add_argument('--device', default=1)

    # opt configs
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--n_frames', type=int, default=64)

    # for neuralef
    parser.add_argument('--alpha', default=0.5, type=float)
    # parser.add_argument('--k', default=64, type=int)
    parser.add_argument('--momentum', default=.9, type=float)
    parser.add_argument('--not_all_together', default=True, action='store_true')
    parser.add_argument('--proj_dim', default=[2048, 1024], type=int, nargs='+')
    parser.add_argument('--no_proj_bn', default=False, action='store_true')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--t', default=10, type=float)
    parser.add_argument('--k', default=1024, type=float)
    parser.add_argument('--coeff', default=1.0, type=float)
    parser.add_argument('--is_train', default=False, type=bool)

    args = parser.parse_args()

    args.proj_bn = not args.no_proj_bn

    device = f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    device = torch.device(device)
    torch.backends.cudnn.benchmark = True

    video_data = video_dataset('./data/test_data', args.n_frames)
    data_loader = DataLoader(video_data, batch_size=args.batch_size, shuffle=True)
    

    model = NeuralEFCLR(args)
    model.to(device)

    if not args.is_train:
        state_dict = torch.load('./ckpt/ckpt1001.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.register_buffer('eigenvalues', state_dict['eigenvalues'])
        model.register_buffer('eigennorm', state_dict['eigennorm'])

    for epoch in range(args.num_epochs):
        model.eval()
        for video1, video2, ind1, ind2 in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", unit="batch"):
            video1 = video1.to(device)
            video2 = video2.to(device)
            ind1 = ind1.to(device)
            ind2 = ind2.to(device)

            K = model.forward(video1, video2, ind1, ind2)
            print(K, "\n", torch.abs(ind1 - ind2))
            x_data = torch.abs(ind1 - ind2).squeeze(0).cpu().detach().numpy()
            y_data = K.cpu().detach().numpy()
            plt.scatter(x_data, y_data)
            plt.savefig("ind-k_recover.jpg")







