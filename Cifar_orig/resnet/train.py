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
from model import ResNet50, TimeEmbedding

class NeuralEFCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone = ResNet50(num_classes=10)
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

    def forward(self, y1, y2, t_emb, index, labels=None):
        if self.args.not_all_together:
            r1 = self.backbone(y1, t_emb)
            r2 = self.backbone(y2, t_emb)

            z1 = self.projector(r1)
            z2 = self.projector(r2)
        else:
            r1, r2 = self.backbone(torch.cat([y1, y2], 0), t_emb).chunk(2, dim=0)
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
        #     logits = self.online_head(r1.detach())
        #     cls_loss = F.cross_entropy(logits, labels)
        #     acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)
        # else:
        cls_loss, acc = torch.Tensor([0.]).to(device), torch.Tensor([0.]).to(device)

        return loss, reg, cls_loss, acc


def generate_gaussian_noise(mean, std, size):
    # noise = np.random.normal(mean, std, size)
    noise = torch.randn(std.shape[0], 3, 32, 32).to(device)
    noise = noise * std.view(std.shape[0], 1, 1, 1)
    return noise

def process_input(images, sigma, device):
    y = images
    y = y.to(device)
    n = generate_gaussian_noise(mean=0, std=sigma, size=(32, 32))
    n = n.to(device)
    return (y + n).float()

ckpt_path = './ckpt/'
        
if __name__ == '__main__':
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    parser = argparse.ArgumentParser(description='Learn diffusion eigenfunctions on ImageNet ')
    parser.add_argument('--device', type=int, default=4)
    # for the size of image
    parser.add_argument('--resolution', type=int, default=256)

    # opt configs
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=None, metavar='LR')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=1000, metavar='N', help='iterations to warmup LR')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--accum_iter', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam')

    # for neuralef
    parser.add_argument('--alpha', default=0.01, type=float)
    parser.add_argument('--k', default=64, type=int)
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



    args = parser.parse_args() 

    args.proj_bn = not args.no_proj_bn
    if args.min_lr is None:
        args.min_lr = args.lr * 0.001

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    torch.backends.cudnn.benchmark = True


    t_tabel = torch.arange(0, 10, 0.01)

        
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    cifar_dataset = CIFAR10Dataset(root_dir='/home/lhy/Projects/EDM_Diffusion/data', train=True, transform=transform)
    cifar_dataset_test = CIFAR10Dataset(root_dir='/home/lhy/Projects/EDM_Diffusion/data', train=False, transform=transform)
    train_data_loader= DataLoader(cifar_dataset, batch_size=args.batch_size, shuffle=True)
    test_data_loader= DataLoader(cifar_dataset_test, batch_size=args.batch_size, shuffle=False)
    
    model_arch = open('model.txt', 'w')
    model = NeuralEFCLR(args)
    print(model, file=model_arch)
    model.to(device)
    # assert 1==2

    optimizer = LARS(model.parameters(),
                         lr=args.lr, weight_decay=args.weight_decay,
                         weight_decay_filter=exclude_bias_and_norm,
                         lars_adaptation_filter=exclude_bias_and_norm)
    
    losses, regs, cls_losses, accs, n_steps, n_steps_clf = 0, 0, 0, 0, 0, 1e-8
    scaler = torch.cuda.amp.GradScaler()

    loss_his = []
    reg_his = []
    acc_his = []

    for epoch in range(args.num_epochs):
        model.train()
        for (image1, image2), labels, classes_name in tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="batch"):
            n_steps += 1
            # cosine decay lr with warmup
            
            # print(image1)
            # assert 1==2
            lr = args.lr
            optimizer.zero_grad()
            image1 = image1.to(device)
            image2 = image2.to(device)
            labels = labels.to(device)

            index = torch.randint(0, 1000, (image1.shape[0],))
            #index = torch.randint(0, 1000, (1,))
            t = t_tabel[index]
            t = t.to(device)

            image1 = image1.permute(0, 3, 1, 2)
            image2 = image2.permute(0, 3, 1, 2)

            ypn1 = process_input(image1, t, device)
            ypn2 = process_input(image2, t, device)

            T_EMB = TimeEmbedding(T=1000, d_model=128, dim=128 * 4)
            t_emb = T_EMB(index)
            t_emb= t_emb.to(device)
            # if(t<100):
            #     tmp1 = ypn1
            #     print(tmp1.shape)
            #     tmp = tmp1[0]
            #     print(tmp.shape)
            #     tmp = tmp.clamp(0, 1)
            #     tmp = tmp * 255
            #     tmp = tmp.byte()
            #     image = transforms.ToPILImage()(tmp)
            #     image.save(f"./image_std{float(t)}.jpg")
            # assert t == 200

            loss, reg, cls_loss, acc = model.forward(ypn1, ypn2, t_emb, index, labels)

            scaler.scale(loss + reg * args.alpha + cls_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_his.append(loss.item())
            reg_his.append(reg.item())
            acc_his.append(acc.item())

            if n_steps % 20 == 19:
                print(f'epoch={epoch + 1}, loss={loss.item():.6f},'
                          f' reg={reg.item():.6f}, cls_loss={cls_loss.item():.4f}'
                          f' acc={acc.item():.4f},'
                          f' learning_rate={lr:.4f},')
        if epoch % 5 == 4:
            torch.save(model, ckpt_path + f'ckpt{epoch+1}.pt')
                
        # if epoch % 2 == 0:
        #     model.eval()
        #     for (image1, image2), labels, classes_name in tqdm(test_data_loader, desc=f"Testing {epoch+1}", unit="batch"):
        #         image1 = image1.to(device)
        #         image2 = image2.to(device)
        #         labels = labels.to(device)

        #         image1 = image1.permute(0, 3, 1, 2)
        #         image2 = image2.permute(0, 3, 1, 2)
                
        #         t = 0
                
        #         ypn1 = process_input(image1, t, device)
        #         ypn2 = process_input(image2, t, device)
                
        #         t = torch.zeros(16)
        #         t[0] = 1
        #         t = t.to(device)
        #         index = 0
        #         loss, reg, cls_loss, acc = model.forward(ypn1, ypn2, t, index, labels)
        #         print(f'epoch={epoch + 1}, loss={loss.item():.6f},'
        #                   f' reg={reg.item():.6f}, cls_loss={cls_loss.item():.4f}'
        #                   f' acc={acc.item():.4f},')
        
            

    plt.plot(loss_his, color='blue')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.savefig('./loss_1.png')
    plt.clf()

    plt.plot(reg_his, color='green')
    plt.xlabel('step')
    plt.ylabel('reg')
    plt.savefig('./reg_1.png')
    plt.clf()

    plt.plot(acc_his, color='red')
    plt.xlabel('step')
    plt.ylabel('Acc')
    plt.savefig(f'./Acc_1.png')





    
    
    # net = ResNetWithTimeEmbedding(time_embedding_dim=10)
    # print(net)
    # features = torch.randn(batch_size, num_features)  # 输入特征向量
    # time_index = torch.tensor([0, 1, 2, 3])  # 对应的时间索引
    # output = net(features, time_index)
    # print(output.shape)
    


# parser = argparse.ArgumentParser(description='Training')
# parser.add_argument('--data', type=str, metavar='DIR',
#                     help='path to dataset')
# parser.add_argument('--workers', default=8, type=int, metavar='N',
#                     help='number of data loader workers')
# parser.add_argument('--epochs', default=100, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--batch-size', default=4096, type=int, metavar='N',
#                     help='mini-batch size')
# parser.add_argument('--learning-rate', default=4.8, type=float, metavar='LR',
#                     help='base learning rate')
# parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
#                     help='weight decay')
# parser.add_argument('--print-freq', default=5, type=int, metavar='N',
#                     help='print frequency')
# parser.add_argument('--checkpoint-dir', type=str, default='./logs/',
#                     metavar='DIR', help='path to checkpoint directory')
# parser.add_argument('--log-dir', type=str, default='./logs/',
#                     metavar='DIR', help='path to log directory')
# parser.add_argument('--dim', default=360, type=int,
#                     help="dimension of subvector sent to infoNCE")
# parser.add_argument('--mode', type=str, default="baseline",
#                     choices=["baseline", "simclr", "directclr", "neuralef", "bt", "spectral"],
#                     help="project type")
# parser.add_argument('--name', type=str, default='default')
# parser.add_argument('--resume', type=str, default=None)

# parser.add_argument('--alpha', default=1, type=float)
# parser.add_argument('--proj_dim', default=[2048, 128], type=int, nargs='+')
# parser.add_argument('--no_proj_bn', default=False, action='store_true')
# parser.add_argument('--t', default=10, type=float)

# # for ablation
# parser.add_argument('--no_stop_grad', default=False, action='store_true')
# parser.add_argument('--l2_normalize', default=False, action='store_true')
# parser.add_argument('--not_all_together', default=False, action='store_true')
# parser.add_argument('--positive_def', default=False, action='store_true')

# # Dist
# parser.add_argument('--world-size', default=1, type=int,
#                     help='number of nodes for distributed training')
# parser.add_argument('--rank', default=0, type=int,
#                     help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://127.0.0.1', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist-port', default='1234', type=str,
#                     help='port used to set up distributed training')
# parser.add_argument('--dist-backend', default='nccl', type=str,
#                     help='distributed backend')
# parser.add_argument('--gpu', default=None, type=int,
#                     help='GPU id to use.')

# # for BarlowTwins
# parser.add_argument('--scale-loss', default=1, type=float,
#                     metavar='S', help='scale the loss')



# def main():
#     args = parser.parse_args()

#     if os.path.exists('/data/LargeData/Large/ImageNet'):
#         args.data = '/data/LargeData/Large/ImageNet'
#     # elif os.path.exists('/home/LargeData/Large/ImageNet'):
#     #     args.data = '/home/LargeData/Large/ImageNet'
#     # elif os.path.exists('/workspace/home/zhijie/ImageNet'):
#     #     args.data = '/workspace/home/zhijie/ImageNet'

#     args.proj_bn = not args.no_proj_bn
#     args.ngpus_per_node = torch.cuda.device_count()
#     args.rank *= args.ngpus_per_node
#     args.world_size *= args.ngpus_per_node
#     args.dist_url = '{}:{}'.format(args.dist_url, args.dist_port)
#     torch.multiprocessing.spawn(main_worker, (args,), nprocs=args.ngpus_per_node)
    
    

# def main_worker(gpu, args):
#     args.rank += gpu
#     print(args.world_size, args.rank, args.dist_url)
#     torch.distributed.init_process_group(
#         backend='nccl', init_method=args.dist_url,
#         world_size=args.world_size, rank=args.rank)

#     args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.name)
#     args.log_dir = os.path.join(args.log_dir, args.name)
#     if args.rank == 0:
#         if not os.path.exists(args.checkpoint_dir):
#             os.makedirs(args.checkpoint_dir)
#         stats_file = open(args.checkpoint_dir + '/stats.txt', 'a', buffering=1)
#         print(' '.join(sys.argv))
#         print(' '.join(sys.argv), file=stats_file)

#     torch.cuda.set_device(gpu)
#     torch.backends.cudnn.benchmark = True

#     if args.mode == 'neuralef':
#         model = NeuralEFCLR(args).cuda(gpu)
#     else:
#         assert "input mode right!" == "wrong"

#     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
#     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

#     optimizer = LARS(model.parameters(),
#                     lr=0, weight_decay=args.weight_decay,
#                     weight_decay_filter=exclude_bias_and_norm,
#                     lars_adaptation_filter=exclude_bias_and_norm)

#     # automatically resume from checkpoint if it exists
#     if args.resume is not None:
#         if args.resume == 'auto':
#             if os.path.exists(os.path.join(args.checkpoint_dir, 'checkpoint.pth')):
#                 args.resume = os.path.join(args.checkpoint_dir, 'checkpoint.pth')
#             else:
#                 assert False
#         ckpt = torch.load(args.resume, map_location='cpu')
#         start_epoch = ckpt['epoch']
#         model.load_state_dict(ckpt['model'])
#         optimizer.load_state_dict(ckpt['optimizer'])
#     else:
#         start_epoch = 0

#     dataset = torchvision.datasets.ImageFolder(os.path.join(args.data, 'train'), Transform(args))
#     sampler = torch.utils.data.distributed.DistributedSampler(dataset, drop_last=True)
#     assert args.batch_size % args.world_size == 0
#     per_device_batch_size = args.batch_size // args.world_size
#     loader = torch.utils.data.DataLoader(
#         dataset, batch_size=per_device_batch_size, num_workers=args.workers,
#         pin_memory=True, sampler=sampler)

#     if args.rank == 0:
#         writer = SummaryWriter(log_dir = args.log_dir)
#     else:
#         writer = None

#     start_time = time.time()
#     scaler = torch.cuda.amp.GradScaler()

#     for epoch in range(start_epoch, args.epochs):
#         sampler.set_epoch(epoch)

#         for step, ((y1, y2), labels) in enumerate(loader, start=epoch * len(loader)):
#             print(f"step: {step}")
#             y1 = y1.cuda(gpu, non_blocking=True)
#             y2 = y2.cuda(gpu, non_blocking=True)
#             lr = adjust_learning_rate(args, optimizer, loader, step)

#             optimizer.zero_grad()
#             with torch.cuda.amp.autocast():
#                 loss, reg, cls_loss, acc = model.forward(y1, y2, labels)
#             scaler.scale(loss + reg * args.alpha + cls_loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             if step % args.print_freq == 0:
#                 torch.distributed.reduce(acc.div_(args.world_size), 0)
#                 if args.rank == 0:
#                     print(f'epoch={epoch}, step={step}, loss={loss.item():.6f},'
#                           f' reg={reg.item():.6f}, cls_loss={cls_loss.item():.4f}'
#                           f' acc={acc.item():.4f},'
#                           f' learning_rate={lr:.4f},')
#                     stats = dict(epoch=epoch, step=step, learning_rate=lr,
#                                  loss=loss.item(), reg=reg.item(),
#                                  cls_loss=cls_loss.item(), acc=acc.item(),
#                                  time=int(time.time() - start_time))
#                     print(json.dumps(stats), file=stats_file)

#                 if writer is not None:
#                     writer.add_scalar('Loss/loss', loss.item(), step)
#                     writer.add_scalar('Loss/reg', reg.item(), step)
#                     writer.add_scalar('Loss/cls_loss', cls_loss.item(), step)
#                     writer.add_scalar('Accuracy/train', acc.item(), step)
#                     writer.add_scalar('Hparams/lr', lr, step)

#         if args.rank == 0 and epoch % 10 == 9:
#             # save checkpoint
#             state = dict(epoch=epoch + 1, model=model.state_dict(),
#                          optimizer=optimizer.state_dict(),
#                          )
#             torch.save(state, os.path.join(args.checkpoint_dir, 'checkpoint.pth'))

#     if args.rank == 0:
#         # save final model
#         torch.save(dict(backbone=model.module.backbone.state_dict(),
#                         head=model.module.online_head.state_dict()),
#                 os.path.join(args.checkpoint_dir, 'final.pth'))
        
        

# if __name__ == '__main__':
#     main()