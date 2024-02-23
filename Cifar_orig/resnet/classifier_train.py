import faulthandler

faulthandler.enable()

from pathlib import Path
import argparse
import os
import sys
import random
import subprocess
import time
import json
import numpy as np
import math

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from utils import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from dataset import CIFAR10Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import resnet50
# from sklearn.svm import SVC
import diffusion_process

class NeuralEFCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone = resnet50(num_classes=10)
        #self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()

        self.online_head = nn.Linear(2048, 10)

        if args.proj_dim[0] == 0:
            self.projector = nn.Identity()
        else:
            if len(args.proj_dim) > 1:
                sizes = [2048,] + args.proj_dim
                layers = []
                for i in range(len(sizes) - 2):
                    layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                    if args.proj_bn:
                        layers.append(nn.BatchNorm1d(sizes[i+1]))
                    layers.append(nn.ReLU(inplace=True))

                layers.append(nn.Linear(sizes[-2], sizes[-1], bias=True))

                self.projector = nn.Sequential(*layers)
            elif len(args.proj_dim) == 1:
                self.projector = nn.Linear(2048, args.proj_dim[0], bias=True)

        if args.rank == 0 and hasattr(self, 'projector'):
            print(self.projector)

    def forward(self, y1, y2, t, index, labels=None):
        if self.args.not_all_together:
            r1 = self.backbone(y1, t)
            r2 = self.backbone(y2, t)

            z1 = self.projector(r1)
            z2 = self.projector(r2)
        else:
            r1, r2 = self.backbone(torch.cat([y1, y2], 0), t).chunk(2, dim=0)
            z1, z2 = self.projector(torch.cat([r1, r2], 0)).chunk(2, dim=0)

        psi1 = gather_from_all(z1)
        psi2 = gather_from_all(z2)

        if self.args.l2_normalize:
            psi1 = F.normalize(psi1, dim=1) * math.sqrt(self.args.t)
            psi2 = F.normalize(psi2, dim=1) * math.sqrt(self.args.t)
        else:
            if self.args.not_all_together:
                psi1 = psi1.div(psi1.norm(dim=0).clamp(min=1e-6)) * math.sqrt(self.args.t)
                psi2 = psi2.div(psi2.norm(dim=0).clamp(min=1e-6)) * math.sqrt(self.args.t)
            else:
                norm_ = (psi1.norm(dim=0) ** 2 + psi2.norm(dim=0) ** 2).sqrt().clamp(min=1e-6)
                psi1 = psi1.div(norm_) * math.sqrt(2 * self.args.t)
                psi2 = psi2.div(norm_) * math.sqrt(2 * self.args.t)

        if self.args.positive_def:
            psi1 /= math.sqrt(2)
            psi2 /= math.sqrt(2)

            psi_K_psi_diag = (psi1 * psi2 * 2 + psi1 * psi1 + psi2 * psi2).sum(0).view(-1, 1)
            if self.args.no_stop_grad:
                psi_K_psi = (psi1.T + psi2.T) @ (psi1 + psi2)
            else:
                psi_K_psi = (psi1.detach().T + psi2.detach().T) @ (psi1 + psi2)

            loss = - psi_K_psi_diag.sum()
            reg = ((psi_K_psi) ** 2).triu(1).sum()
        else:
            psi_K_psi_diag = (psi1 * psi2).sum(0).view(-1, 1)
            if self.args.no_stop_grad:
                psi2_d_K_psi1 = psi2.T @ psi1
                psi1_d_K_psi2 = psi1.T @ psi2
            else:
                psi2_d_K_psi1 = psi2.detach().T @ psi1
                psi1_d_K_psi2 = psi1.detach().T @ psi2

            loss = - psi_K_psi_diag.sum() * 2
            reg = ((psi2_d_K_psi1) ** 2).triu(1).sum() \
                + ((psi1_d_K_psi2) ** 2).triu(1).sum()

        loss /= psi_K_psi_diag.numel()
        reg /= psi_K_psi_diag.numel()

        # if index < 4:
        logits = self.online_head(r1.detach())
        cls_loss = F.cross_entropy(logits, labels)
        acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)
        # else:
        #cls_loss, acc = torch.Tensor([0.]).to(device), torch.Tensor([0.]).to(device)

        return loss, reg, cls_loss, acc


# noise_table = []

# def generate_gaussian_noise(steps=2000):
#     # noise = np.random.normal(mean, std, size)
#     noise = torch.randn(3, 32, 32) * 1.0 / 128
#     for _ in range(int(steps) - 1):
#         noise += torch.randn(3, 32, 32) * 1.0 / 128
#         noise_table.append(noise)
#     # noise = noise * std.view(std.shape[0], 1, 1, 1)

# def process_input(images, steps, device):
#     y = images
#     y = y.to(device)
#     n = noise_table[int(steps)]
#     n = n.to(device)
#     return (y + n).float()

# def generate_gaussian_noise(steps):
#     # noise = np.random.normal(mean, std, size)
#     noise = torch.randn((3, 32, 32)) * 1.0 / 128 * float(steps) / 10.0  + 0.0
#     # noise = noise * std.view(std.shape[0], 1, 1, 1)
#     return noise

# def process_input(images, step, device):
#     y = images
#     y = y.to(device)
#     n = generate_gaussian_noise(step)
#     n = n.to(device)
#     res =(y + n).clamp(-1, 1)
#     return res.float()


num_sigmas = 1000
P_mean = 0
P_std = 1.5
rnd_normal = torch.randn([num_sigmas, 1, 1, 1])
sigmas = (rnd_normal * P_std + P_mean).exp()
sigmas, sorted_indices = torch.sort(sigmas, dim=0, descending=False)

def generate_gaussian_noise(steps):
    # noise = np.random.normal(mean, std, size)
    sigma_batch = sigmas[steps]
    noise = torch.randn((3, 32, 32)) * 1.0 / 64.0 * sigma_batch
    #noise = torch.randn((3, 32, 32)) * float(sigmas[int(steps)])
    # noise = noise * std.view(std.shape[0], 1, 1, 1)
    return noise

def process_input(images, step, device):
    y = images
    y = y.to(device)
    n = generate_gaussian_noise(step)
    n = n.to(device)
    res =(y + n).clamp(-1, 1)
    return res.float()

ckpt_path = './ckpt/'
        
if __name__ == '__main__':
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    parser = argparse.ArgumentParser(description='Learn diffusion eigenfunctions on ImageNet ')
    parser.add_argument('--device', type=int, default=6)
    # for the size of image
    parser.add_argument('--resolution', type=int, default=256)

    # opt configs
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--min_lr', type=float, default=None, metavar='LR')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=1000, metavar='N', help='iterations to warmup LR')
    parser.add_argument('--num_epochs', type=int, default=10)

    # for neuralef
    # parser.add_argument('--k', default=64, type=int)
    parser.add_argument('--momentum', default=.9, type=float)
    parser.add_argument('--time_dependent', default=False, action='store_true')
    parser.add_argument('--not_all_together', default=True, action='store_true')
    parser.add_argument('--proj_dim', default=[2048, 1024], type=int, nargs='+')
    parser.add_argument('--no_proj_bn', default=False, action='store_true')
    parser.add_argument('--no_stop_grad', default=False, action='store_true')
    parser.add_argument('--l2_normalize', default=False, action='store_true')
    parser.add_argument('--positive_def', default=False, action='store_true')
    parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
    parser.add_argument('--t', default=10, type=float)
    parser.add_argument('--weights', default='freeze', type=str,
                    choices=('finetune', 'freeze'),
                    help='finetune or freeze resnet weights')
    parser.add_argument('--lr-classifier', default=0.1, type=float, metavar='LR',
                    help='classifier base learning rate')
    parser.add_argument('--lr-backbone', default=0.0, type=float, metavar='LR',
                    help='backbone base learning rate')


    args = parser.parse_args() 
    print(args.weights)

    args.proj_bn = not args.no_proj_bn
    if args.min_lr is None:
        args.min_lr = args.lr * 0.001

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    torch.backends.cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    cifar_dataset = CIFAR10Dataset(root_dir='/home/haoyuan/Projects/EDM_Diffusion/data', train=True, transform=transform)
    cifar_dataset_test = CIFAR10Dataset(root_dir='/home/haoyuan/Projects/EDM_Diffusion/data', train=False, transform=transform)
    train_data_loader= DataLoader(cifar_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_data_loader= DataLoader(cifar_dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    # model = ResNet50(num_classes=10)
    # model.fc = nn.Linear(2048, 10)
    model = resnet50(10)
    model.to(device)

#############################change checkpoint path##############################
    state_dict = torch.load('./ckpt/ckpt30.pth', map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict['backbone'], strict=False)
    assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    if args.weights == 'freeze':
        model.requires_grad_(False)
        model.fc.requires_grad_(True)
    classifier_parameters, model_parameters = [], []
    for name, param in model.named_parameters():
        if name in {'fc.weight', 'fc.bias'}:
            classifier_parameters.append(param)
        else:
            model_parameters.append(param)


    param_groups = [dict(params=classifier_parameters, lr=args.lr_classifier)]
    if args.weights == 'finetune':
        param_groups.append(dict(params=model_parameters, lr=args.lr_backbone))
    

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    #optimizer = optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    loss_his = []
    acc_his = []
    test_acc_his = []

    best_acc = -10
    test_log = open("test_log.txt", 'w')
    n_steps = 0
    Schedule = diffusion_process.GaussianDiffusion()
    for epoch in range(args.num_epochs):
        if args.weights == 'finetune':
            model.train()
        elif args.weights == 'freeze':
            model.eval()
        else:
            assert False

        for (image1, image2), labels, classes_name in tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="batch"):
            n_steps += 1
            lr = args.lr
            optimizer.zero_grad()
            image1 = image1.to(device)
            image2 = image2.to(device)
            labels = labels.to(device)

            #rand_index = torch.randint(0, 100, size=(1,))
            rand_index = torch.randint(0, 100, size=(image1.shape[0],), device=image1.device)

            #index = torch.full((image1.shape[0],), int(rand_index))
            index = rand_index

            image1 = image1.permute(0, 3, 1, 2)
            image2 = image2.permute(0, 3, 1, 2)
            
            ypn1 = Schedule.q_sample(image1, rand_index)
            ypn2 = Schedule.q_sample(image2, rand_index)

            # ypn1 = process_input(image1, rand_index, device)
            # ypn2 = process_input(image2, rand_index, device)

            # loss, reg, cls_loss, acc = model.forward(ypn1, ypn2, index.to(device), index, labels)
            
            logits = model.forward(ypn2, index.to(device))
            # logits = model.forward(ypn2)
            cls_loss = F.cross_entropy(logits, labels)
            acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)

            (cls_loss).backward()
            # scaler.scale(cls_loss).backward()
            optimizer.step()

            loss_his.append(cls_loss.item())
            acc_his.append(acc.item())

            if n_steps % 20 == 19:
                print(f'epoch={epoch + 1},'
                          f' cls_loss={cls_loss.item():.4f}'
                          f' acc={acc.item():.4f},'
                          f' learning_rate={lr:.4f},')
        
        n_steps = 0
        if epoch % 2 == 0:
            model.eval()
            tot_acc = 0
            nums = 0
            for (image1, image2), labels, classes_name in tqdm(test_data_loader, desc=f"Testing {epoch+1}", unit="batch"):
                nums += 1
                image1 = image1.to(device)
                image2 = image2.to(device)
                labels = labels.to(device)

                # rand_index = torch.randint(0, 10, (1,))
                # index = torch.full((image1.shape[0],), int(rand_index))
                #rand_index = torch.randint(0, 100, size=(1,))
                rand_index = torch.randint(0, 500, size=(image1.shape[0],))

                #index = torch.full((image1.shape[0],), int(rand_index))
                index = rand_index

                image1 = image1.permute(0, 3, 1, 2)
                image2 = image2.permute(0, 3, 1, 2)

                ypn1 = process_input(image1, rand_index, device)
                ypn2 = process_input(image2, rand_index, device)

                logits = model.forward(ypn2, index.to(device))
                #logits = model.forward(ypn2)
                cls_loss = F.cross_entropy(logits, labels)
                acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)

                tot_acc += acc.item()
                print(f'epoch={epoch + 1},'
                          f' cls_loss={cls_loss.item():.4f}'
                          f' acc={acc.item():.4f},')
            test_acc_his.append(tot_acc / nums)
            print(f"Epoch: {epoch + 1} test_acc: ", tot_acc / nums)
            print(f"Epoch: {epoch + 1} test_acc: ", tot_acc / nums, file=test_log)
            if best_acc < tot_acc / nums:
                torch.save(model, ckpt_path + 'best_acc.pt')
            
    if not os.path.exists("./figs"):
        os.mkdir("./figs")

    plt.plot(loss_his, color='blue')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.savefig('./figs/cls_loss.png')
    plt.clf()

    plt.plot(acc_his, color='red')
    plt.xlabel('step')
    plt.ylabel('Acc')
    plt.savefig(f'./figs/cls_acc.png')
    plt.clf()

    plt.plot(test_acc_his, color='blue')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.savefig('./figs/cls_test_acc.png')