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

class NeuralEFCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        #self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        pretrained = torchvision.models.resnet50(pretrained=True)
        self.backbone = resnet50(1000)

        pretrained_state_dict = pretrained.state_dict()
        custom_state_dict = self.backbone.state_dict()
        print('loading pretrained state dict....')
        for name, param in pretrained_state_dict.items():
            if name in custom_state_dict:
                custom_state_dict[name].copy_(param)
        
        self.backbone.load_state_dict(custom_state_dict)
        print('load pretrained model done!')

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

    def forward(self, y1, y2, t_input, index, labels=None):
        ########claculate Bc size##############just for check whether the kernel is correct
        Bc_nums = []
        for class_id in range(10):
            tmp = torch.full((labels.shape[0],), class_id).to(device)
            Bc_nums.append(torch.sum(torch.eq(tmp, labels)))

        ##########calculate new kernel############
        #temporily use 'for' loop to create a kernel#####
        new_kernel = torch.empty((labels.shape[0], labels.shape[0])).to(device)
        tmp_cnt = 0
        for i in labels:
            result = torch.where(labels != i, torch.zeros_like(labels), 1.0 / Bc_nums[int(i)])
            new_kernel[tmp_cnt] = result * (1.0 / Bc_nums[int(i)])
            tmp_cnt += 1
            
        # print(labels)
        # print(new_kernel)
        # assert 1==2   

        if self.args.not_all_together:
            r1 = self.backbone(y1, t_input)
            r2 = self.backbone(y2, t_input)

            z1 = self.projector(r1)
            z2 = self.projector(r2)
        else:
            r1, r2 = self.backbone(torch.cat([y1, y2], 0), t_input).chunk(2, dim=0)
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
            psi_K_psi = psi1.T @ new_kernel @ psi2
            psi_K_psi_diag = torch.diagonal(psi_K_psi)
            psi_K_psi_diag = psi_K_psi_diag.unsqueeze(1)

            # psi_K_psi_diag = (psi1 * psi2).sum(0).view(-1, 1)
            
            if self.args.no_stop_grad:
                psi2_d_K_psi1 = psi2.T @ psi1
                psi1_d_K_psi2 = psi1.T @ psi2
            else:
                # psi2_d_K_psi1 = psi2.detach().T @ psi1
                # psi1_d_K_psi2 = psi1.detach().T @ psi2
                psi2_d_K_psi1 = psi2.detach().T @ new_kernel @ psi1
                psi1_d_K_psi2 = psi1.detach().T @ new_kernel @ psi2

            loss = - psi_K_psi_diag.sum() * 2
            reg = ((psi2_d_K_psi1) ** 2).triu(1).sum() \
                + ((psi1_d_K_psi2) ** 2).triu(1).sum()

        loss /= psi_K_psi_diag.numel()
        reg /= psi_K_psi_diag.numel()


        if index[0] < 800:
            logits = self.online_head(r1.detach())
            #logits = self.online_head(r1)
            cls_loss = F.cross_entropy(logits, labels)
            acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)

        else:
            cls_loss, acc = torch.Tensor([0.]).to(device), torch.Tensor([0.]).to(device)
            
        return loss, reg, cls_loss, acc

num_sigmas = 1000
P_mean = 0
P_std = 1.4
rnd_normal = torch.randn([num_sigmas, 1, 1, 1])
sigmas = (rnd_normal * P_std + P_mean).exp()
sigmas, sorted_indices = torch.sort(sigmas, dim=0, descending=False)

def generate_gaussian_noise(steps):
    # noise = np.random.normal(mean, std, size)
    noise = torch.randn((3, 32, 32)) * 1.0 / 128 * float(sigmas[int(steps)])
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
    parser.add_argument('--device', type=int, default=7)

    # opt configs
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=None, metavar='LR')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=1000, metavar='N', help='iterations to warmup LR')
    parser.add_argument('--num_epochs', type=int, default=50)

    # for neuralef
    parser.add_argument('--alpha', default=0.04, type=float)
    # parser.add_argument('--k', default=64, type=int)
    parser.add_argument('--momentum', default=.9, type=float)
    parser.add_argument('--time_dependent', default=False, action='store_true')
    parser.add_argument('--not_all_together', default=True, action='store_true')
    parser.add_argument('--proj_dim', default=[2048, 2048], type=int, nargs='+')
    parser.add_argument('--no_proj_bn', default=False, action='store_true')
    parser.add_argument('--no_stop_grad', default=False, action='store_true')
    parser.add_argument('--l2_normalize', default=False, action='store_true')
    parser.add_argument('--positive_def', default=False, action='store_true')
    parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
    parser.add_argument('--t', default=10, type=float)



    args = parser.parse_args() 

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
    cifar_dataset = CIFAR10Dataset(root_dir='/home/lhy/Projects/EDM_Diffusion/data', train=True, transform=transform)
    cifar_dataset_test = CIFAR10Dataset(root_dir='/home/lhy/Projects/EDM_Diffusion/data', train=False, transform=transform)

    train_data_loader= DataLoader(cifar_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_data_loader= DataLoader(cifar_dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=True)


    model_arch = open('model.txt', 'w')###tembedding has been added into model.py resblock
    model = NeuralEFCLR(args)
    print(model, file=model_arch)
    model.to(device)

    optimizer = LARS(model.parameters(),
                         lr=0, weight_decay=args.weight_decay,
                         weight_decay_filter=exclude_bias_and_norm,
                         lars_adaptation_filter=exclude_bias_and_norm)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    losses, regs, cls_losses, accs, n_steps, n_steps_clf = 0, 0, 0, 0, 0, 1e-8

    loss_his = []
    reg_his = []
    acc_his = []
    test_acc_his = []

    

    pretrian_log = open('pretrain_log.txt', 'w')

    for epoch in range(args.num_epochs):
        model.train()
        for (image1, image2), labels, classes_name in tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="batch"):
            n_steps += 1
            # lr = args.lr
            lr = adjust_learning_rate(args, optimizer, train_data_loader, n_steps)
            #lr = adjust_learning_rate2(args, optimizer, train_data_loader, n_steps)
            optimizer.zero_grad()
            image1 = image1.to(device)
            image2 = image2.to(device)
            labels = labels.to(device)

            rand_index = torch.randint(0, 1000, size=(1,))

            index = torch.full((image1.shape[0],), int(rand_index))

            image1 = image1.permute(0, 3, 1, 2)
            image2 = image2.permute(0, 3, 1, 2)

            ypn1 = process_input(image1, rand_index, device)
            ypn2 = process_input(image2, rand_index, device)

            # t_emb = T_EMB(index)
            # t_emb= t_emb.to(device)
            # if(index[0]<10000):
            #     tmp1 = ypn1
            #     print(tmp1.shape)
            #     tmp = tmp1[0]
            #     print(tmp.shape)
            #     tmp = tmp.clamp(0, 1)
            #     tmp = tmp * 255
            #     tmp = tmp.byte()
            #     image = transforms.ToPILImage()(tmp)
            #     image.save(f"./image_index{index[0]}.jpg")
            # assert index[0] == 400
            
            index = index.to(device)

            loss, reg, cls_loss, acc = model.forward(ypn1, ypn2, index, index, labels.to(device))

            (loss + reg * args.alpha + cls_loss).backward()
            #cls_loss.backward()

            optimizer.step()


            acc_his.append(acc.item())

            losses += loss.item()
            regs += reg.item()

            if n_steps % 20 == 19:

                print(f'epoch={epoch + 1}, loss={loss.item():.6f},'
                          f' reg={reg.item():.6f}, cls_loss={cls_loss.item():.4f}'
                          f' acc={acc.item():.4f},'
                          f' learning_rate={lr:.4f},')
                print(f'epoch={epoch + 1}, loss={loss.item():.6f},'
                          f' reg={reg.item():.6f}, cls_loss={cls_loss.item():.4f}'
                          f' acc={acc.item():.4f},'
                          f' learning_rate={lr:.4f},', file=pretrian_log)
                
        loss_his.append(1.0 * losses / n_steps)
        reg_his.append(1.0 * regs / n_steps)
        regs = 0
        losses = 0
        n_steps = 0
        
        if epoch % 2 == 1:
            #######save backbone#########
            torch.save(dict(backbone=model.backbone.state_dict(),
                        head=model.online_head.state_dict()),  ckpt_path + f'ckpt{epoch+1}.pth')
            
        model.eval()
        test_step = 0
        total_test = 0
        with torch.no_grad():
            for (image1, image2), labels, classes_name in tqdm(test_data_loader, desc=f"Epoch", unit="batch"):
                test_step += 1
                image1 = image1.to(device)
                image2 = image2.to(device)
                labels = labels.to(device)

                rand_index = torch.randint(0, 100, size=(1,))##########use clean data temporary with no embedding

                index = torch.full((image1.shape[0],), int(rand_index))

                image1 = image1.permute(0, 3, 1, 2)
                image2 = image2.permute(0, 3, 1, 2)

                ypn1 = process_input(image1, rand_index, device)
                ypn2 = process_input(image2, rand_index, device)

                index = index.to(device)

                loss, reg, cls_loss, acc = model.forward(ypn1, ypn2, index, index, labels.to(device))
            
                total_test += acc
            
                print(f"Epoch :{epoch + 1}, Test acc :{acc}")
            
            test_acc_his.append(float(1.0 * total_test / test_step))
 
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

    plt.plot(acc_his, color='red')
    plt.xlabel('step')
    plt.ylabel('Acc')
    plt.savefig(f'./figs/Acc.png')
    plt.clf()
    
    plt.plot(test_acc_his, color='red')
    plt.xlabel('step')
    plt.ylabel('Test_Acc')
    plt.savefig(f'./figs/Test_Acc.png')

