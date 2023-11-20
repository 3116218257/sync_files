
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import dnnlib
from torch_utils import distributed as dist

from myutilities import TConditionalBatchNorm1d, CustomSequential, DDPMDiffuser, LearnedSinusoidalPosEmb, adjust_learning_rate2

from torchvision import transforms
from torchvision.utils import save_image
# from model import SongUNetWithMLP
from dataset import CIFAR10Dataset
from torch.utils.data import DataLoader
import os
import random
import numpy as np
from PIL import Image

def process_input(images, sigma, labels=None, augment_pipe=None):
    sigma_data=0.5
    y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
    y = y.to(args.device)
    n = torch.randn_like(y) * sigma.to(args.device)
    n = n.to(args.device)
    return y + n

def cal_input(x, sigma, sigma_data=0.5, label_dim=10, class_labels=None, force_fp32=False):
    x = x.to(torch.float32)
    sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
    class_labels = None if label_dim == 0 else torch.zeros([1, label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, label_dim)
    dtype = torch.float32
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
    c_in = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
    c_noise = sigma.log() / 4
    return (c_in * x).to(dtype), c_noise.flatten(), class_labels 


class DiffusionEF(nn.Module):
    def __init__(self, network_pkl, args, image_size, num_sigmas, num_classes, learned_sinusoidal_dim=16, mlp_hidden_dims = 1024, mlp_num_layers = 6, mlp_activation = nn.ReLU()):

        super().__init__()

        dist.print0(f'Loading network from "{network_pkl}"...')
        with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f)['ema']

        self.net = net.model
        for params in self.net.parameters():
            params.requires_grad = True

        self.norm_Silu = self.norm_silu(input_channels = 3)
        
        self.mlp = self.build_mlp(input_dim=3 * (32 ** 2),
                            hidden_dim=mlp_hidden_dims,
                            output_dim=1024,
                            num_layers=mlp_num_layers,
                            activation=mlp_activation)

        self.mlp_c = self.build_mlp(input_dim=args.k,
                            hidden_dim=mlp_hidden_dims,
                            output_dim=args.k,
                            num_layers=mlp_num_layers,
                            activation=mlp_activation)
        
        #self.attention = Attention(input_size=3)

        self.head = nn.Linear(1024, args.k)
        self.online_clf_head = nn.Linear(args.k, num_classes) 
        self.args = args
        self.momentum = args.momentum
        self.training = True

            
        self.register_buffer('eigennorm_sqr', torch.zeros(num_sigmas, args.k))
        self.register_buffer('eigenvalues', torch.zeros(num_sigmas, args.k))
        self.register_buffer('num_calls', torch.Tensor([0]))

    def build_mlp(self, input_dim, hidden_dim, output_dim, num_layers, activation):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation)
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation)
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(activation)
        return nn.Sequential(*layers)
    
    def norm_silu(self, input_channels):
        layers = []
        layers.append(nn.BatchNorm2d(input_channels))
        layers.append(nn.SiLU())
        return nn.Sequential(*layers)
        
    def forward(self, x1, x2, sigma, index, labels=None, return_psi=False):
        cin_x1, noise_flt1, cls_lbs1 = cal_input(x1, sigma) 
        cin_x2, noise_flt2, cls_lbs2 = cal_input(x2, sigma) 
        with torch.no_grad():
            h1 = self.net(cin_x1, noise_flt1, cls_lbs1)
            h2 = self.net(cin_x2, noise_flt2, cls_lbs2)
            h1 = self.norm_Silu(h1)
            h2 = self.norm_Silu(h2)
        assert h1.shape[0] == h2.shape[0]
        

        h1 = h1.to(torch.float32)
        h2 = h2.to(torch.float32)

        h = torch.cat([h1.view(h1.shape[0],-1), h2.view(h1.shape[0],-1)], 0)
        # print(h.shape)
        h = self.mlp(h)
        psi = self.head(h)
        
        if self.training:
            norm_sqr = psi.norm(dim=0) ** 2 / psi.shape[0]
            with torch.no_grad():
                if self.num_calls == 0:
                    self.eigennorm_sqr[index].copy_(norm_sqr.data)
                else:
                    self.eigennorm_sqr[index].mul_(self.momentum).add_(
                        norm_sqr.data, alpha = 1-self.momentum)
            
            norm_ = norm_sqr.clamp(min=0).sqrt().clamp(min=1e-6)
        else:
            norm_ = self.eigennorm_sqr[index].clamp(min=0).sqrt().clamp(min=1e-6)

        psi = psi.div(norm_)
        psi1, psi2 = psi.chunk(2, dim=0)
        if return_psi: return psi1, psi2

        psi_K_psi_diag = (psi1 * psi2).sum(0).view(-1, 1)     
        if self.training:
            with torch.no_grad():
                eigenvalues_ = psi_K_psi_diag.view(-1) / psi1.shape[0]
                if self.num_calls == 0:
                    self.eigenvalues[index].copy_(eigenvalues_.data)
                else:
                    self.eigenvalues[index].mul_(self.momentum).add_(
                        eigenvalues_.data, alpha = 1-self.momentum)
                self.num_calls += 1 

        psi2_d_K_psi1 = psi2.detach().T @ psi1
        psi1_d_K_psi2 = psi1.detach().T @ psi2
        
        scale = 1. / psi.shape[0]
        #scale = 1.0
        loss = - (psi_K_psi_diag * scale).sum() * 2
        reg = ((psi2_d_K_psi1) ** 2 * scale).triu(1).sum() \
            + ((psi1_d_K_psi2) ** 2 * scale).triu(1).sum()

        loss /= psi_K_psi_diag.numel()
        reg /= psi_K_psi_diag.numel()
        

        # train the classification head only on clean data
        if index[0] < 20 :
            #logits = self.online_clf_head(psi1.detach())
            logits = self.online_clf_head(self.mlp_c(psi1.detach()))
            #print(h.shape, psi1.shape, x1.shape)
            cls_loss = F.cross_entropy(logits, labels.to(torch.long))

            acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)
        else:
            cls_loss, acc = torch.Tensor([0.]).to(x1.device), None

            
        # if loss.shape == torch.Size([]):
        #     loss = torch.unsqueeze(loss, 0)
        #     loss = loss.to(args.device)
        # if reg.shape == torch.Size([]):
        #     reg = torch.unsqueeze(reg, 0)
        #     reg = reg.to(args.device)
        # if cls_loss.shape == torch.Size([]):
        #     cls_loss = torch.unsqueeze(cls_loss, 0)  
        #     cls_loss = cls_loss.to(args.device)

        total_loss = None    
        # total_loss = [loss, reg * args.alpha, cls_loss]
        #     # print(loss.shape, reg.shape, cls_loss.shape)
        # total_loss = torch.stack(total_loss)
        # total_loss = self.attention(total_loss)
        
        return loss, reg, cls_loss, acc, total_loss
            
    
    # def test_acc(self, x1, x2, sigma, t, logsnr, labels):
    def test_acc(self, x1, x2, sigma, index, labels):

        with torch.no_grad():
            # psi1, psi2 = self.forward(x1, x2, sigma, t, logsnr, return_psi=True)
            psi1, psi2 = self.forward(x1, x2, sigma, index, labels, return_psi=True)
            logits1 = self.online_clf_head(psi1.detach())
            logits2 = self.online_clf_head(psi2.detach()) 

        acc = (torch.sum(torch.eq(torch.argmax(logits1, dim=1), labels))+ 
                torch.sum(torch.eq(torch.argmax(logits2, dim=1), labels)))/ (logits1.size(0)+logits2.size(0))

        return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learn diffusion eigenfunctions on ImageNet ')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=50)

    # for the size of image
    parser.add_argument('--resolution', type=int, default=256)

    # for specifying MLP
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--norm_type', default='bn', type=str)
    parser.add_argument('--act_type', default='relu', type=str)
    parser.add_argument('--affine_condition_on_time', default=False, action='store_true')

    # opt configs
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=None, metavar='LR')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=1000, metavar='N', help='iterations to warmup LR')
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--accum_iter', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam')

    # for neuralef
    parser.add_argument('--alpha', default=0.02, type=float)
    parser.add_argument('--k', default=64, type=int)
    parser.add_argument('--momentum', default=.9, type=float)
    parser.add_argument('--time_dependent', default=False, action='store_true')

    args = parser.parse_args()
    if args.min_lr is None:
        args.min_lr = args.lr * 0.001
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    torch.backends.cudnn.benchmark = True

    # discretization of noise added
    num_sigmas = 1000
    num_steps = 100
    sigma_max = 80
    sigma_min = 0.002
    rho = 7
    P_mean=-1.2
    P_std=1.2
    rnd_normal = torch.randn([num_sigmas, 1, 1, 1])
    sigmas = (rnd_normal * P_std + P_mean).exp()
    sigmas, sorted_indices = torch.sort(sigmas, dim=0, descending=False)
    
    # print(sigmas)

    # mydiffusionEF implement
    # define model and initialize with pre-trained model
    # diffuser = DDPMDiffuser().to(device) #the noise_schedule is linear
    network_pkl = '/home/lhy/Projects/EDM_Diffusion/edm-cifar10-32x32-cond-vp.pkl' # vp for SongNet DDPM++
    model=DiffusionEF(network_pkl, image_size=(args.batch_size, 3, args.resolution, args.resolution), args=args, num_sigmas=100, num_classes=10).to(device)
    print('loading weights======')
    model.load_state_dict(torch.load('model_70.pt'))
    # download pretrained model with SongNet DDPM++
    print(model)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=args.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])
    t_steps = t_steps[:-1]
    t_steps = t_steps.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    sigmas = t_steps
    sigmas, sorted_indices = torch.sort(sigmas, dim=0, descending=False)
    
    # construct train dataset
    cifar_dataset_train = CIFAR10Dataset(root_dir='/home/lhy/Projects/EDM_Diffusion/data', train=True, transform=None)
    cifar_dataset_test = CIFAR10Dataset(root_dir='/home/lhy/Projects/EDM_Diffusion/data', train=False, transform=None)
    train_data_loader= DataLoader(cifar_dataset_train, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(cifar_dataset_test, batch_size=args.test_batch_size, shuffle=False)

    # train the DiffusionEF model
    model.train()
    losses, regs, cls_losses, accs, n_steps, n_steps_clf = 0, 0, 0, 0, 0, 1e-8
    if args.opt == 'lars':
        optimizer = LARS(model.parameters(),
                         lr=args.lr, weight_decay=args.weight_decay,
                         weight_decay_filter=exclude_bias_and_norm,
                         lars_adaptation_filter=exclude_bias_and_norm)
    elif args.opt == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), 
                                        lr=args.lr, weight_decay=args.weight_decay)
        #optimizer = torch.optim.AdamW(model.parameters(), 
        #                              lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=args.lr, momentum=.9, 
                                    weight_decay=args.weight_decay)
    optimizer.zero_grad()

    cls_loss_f = open('cls_loss.txt', 'w')

    best_loss = 1e8
    for epoch in range(args.num_epochs):
        
        for (image1, image2), labels, classes_name in train_data_loader:
            n_steps += 1
            # cosine decay lr with warmup
            #lr = adjust_learning_rate2(args.lr, optimizer, n_steps, 
             #                       args.num_epochs, args.warmup_steps, args.min_lr)
            lr = args.lr
            
            image1 = image1.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            index = random.sample(range(20), 1)
            print('index: ', index[0])
            sigma = sigmas[index]
            print('sigma: ', sigma)
            y_pls_n1 = process_input(image1, sigma, labels)
            y_pls_n2 = process_input(image2, sigma, labels)
            y_pls_n1 = y_pls_n1.permute(0, 3, 1, 2)
            y_pls_n2 = y_pls_n2.permute(0, 3, 1, 2)

            ##########################################################################
            # loss, reg, cls_loss, acc = model.forward(y_pls_n1, y_pls_n2, sigma, t, diffuser.logsnr(t), labels)
            loss, reg, cls_loss, acc, tot_l  = model.forward(y_pls_n1, y_pls_n2, sigma.to(device), index, labels)

            #print(w1, w2, w3, (loss + reg * args.alpha + cls_loss).shape, (loss + reg * args.alpha + cls_loss).shape == torch.Size([]) )
            for params in model.parameters():
                params.requires_grad = False
            for params in model.mlp_c.parameters():
                params.requires_grad = True
            for params in model.online_clf_head.parameters():
                params.requires_grad = True    

            (cls_loss).div(args.accum_iter).backward() 

            if epoch % args.accum_iter == 0:
               optimizer.step()
               optimizer.zero_grad()

            losses += loss.detach().item()
            regs += reg.detach().item()

            if acc is not None:
                cls_losses += cls_loss.detach().item()
                accs += acc.detach().item()
                n_steps_clf += 1
            if n_steps%30 == 0:
                print('Epoch: {}, sigma: {}, LR: {:.4f}, Loss: {:.4f}, Reg: {:.4f}, CLS_Loss: {:.4f}, Train acc: {:.2f}%'.format(
                    epoch, sigma.item(), lr, losses / n_steps, regs / n_steps, cls_losses / n_steps_clf, 100 * accs / n_steps_clf))
                print('Epoch: {}, sigma: {}, LR: {:.4f}, Loss: {:.4f}, Reg: {:.4f}, CLS_Loss: {:.4f}, Train acc: {:.2f}%'.format(
                    epoch, sigma.item(), lr, losses / n_steps, regs / n_steps, cls_losses / n_steps_clf, 100 * accs / n_steps_clf), file=cls_loss_f)

        if epoch % 2 == 0:
            print('=======> Testing <=======')
            print('=======> Testing <=======', file=cls_loss_f)
            model.eval()
            total_acc = 0
            count_acc = 0
            for (image1, image2), labels, classes_name in test_data_loader:
                image1 = image1.to(device, non_blocking=True)
                image2 = image2.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                index = random.sample(range(20), 1)
                if index[0] < 20:
                    sigma = sigmas[index]
                    y_pls_n1 = process_input(image1, sigma, labels)
                    y_pls_n2 = process_input(image2, sigma, labels)
                    y_pls_n1 = y_pls_n1.permute(0, 3, 1, 2)
                    y_pls_n2 = y_pls_n2.permute(0, 3, 1, 2)
                    total_acc += model.test_acc(y_pls_n1, y_pls_n2, sigma.to(device), index, labels)
                    count_acc += 1.0

            print('Test acc: ', float(1.0 * total_acc / count_acc))
            print('Test acc: ', float(1.0 * total_acc / count_acc), file=cls_loss_f)
            # print('Test acc: ', float(1.0 * total_acc / count_acc), file=log_loss)
            losses, regs, cls_losses, accs, n_steps, n_steps_clf = 0, 0, 0, 0, 0, 1e-8
            model.train()

