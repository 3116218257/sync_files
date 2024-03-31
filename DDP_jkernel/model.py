import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import warnings
import numpy as np


from utils.dist import gather_from_all, is_master

warnings.filterwarnings('ignore')

class NeuralEFCLR(nn.Module):
    def __init__(self, args, logger, device):
        super().__init__()
        self.args = args
        self.logger = logger
        self.device = device
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()
        
        if args.proj_dim[0] == 0:
            self.projector = nn.Identity()
        else:
            if len(args.proj_dim) > 1:
                sizes = [2048,] + args.proj_dim
                layers = []
                for i in range(len(sizes) - 2):
                    layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                    if not args.no_proj_bn:
                        layers.append(nn.BatchNorm1d(sizes[i+1]))
                    layers.append(nn.ReLU(inplace=True))

                layers.append(nn.Linear(sizes[-2], sizes[-1], bias=True))

                self.projector = nn.Sequential(*layers)
            elif len(args.proj_dim) == 1:
                self.projector = nn.Linear(2048, args.proj_dim[0], bias=True)

        self.momentum = args.momentum
        self.register_buffer('eigennorm_sqr', torch.zeros(args.proj_dim[-1]))
        self.register_buffer('eigenvalues', torch.zeros(args.proj_dim[-1]))
        self.register_buffer('num_calls', torch.Tensor([0]))
    
    def get_K(self, coeff, index1, index2):
        if index1.ndim == 1:
            row_vector = index1.unsqueeze(0)
            col_vector = index2.unsqueeze(0)
        else:
            row_vector = index1
            col_vector = index2
        # K = torch.abs(row_vector - col_vector.T)
        K = torch.exp(-coeff * torch.abs(row_vector - col_vector.T))
        return K

    def forward(self, y, index):
        B = y.shape[0]
        r = self.backbone(y)
        z1, z2 = self.projector(r).chunk(2, dim=0)
        if is_master(): self.logger.debug(f'z1.shape: {z1.shape}')
        psi1 = gather_from_all(z1)
        psi2 = gather_from_all(z2)
        index = gather_from_all(index).reshape(-1, B) # (num_device, B)
        index1, index2 = index.chunk(2, dim=1)
        index1 = index1.reshape(1, -1).float()
        index2 = index2.reshape(1, -1).float()
        if is_master(): self.logger.debug(f'index1: {index1}')
        if is_master(): self.logger.debug(f'index2: {index2}')
        if is_master(): self.logger.debug(f'psi1.shape: {psi1.shape}')

        if self.training:
            with torch.no_grad():
                norm_ = (psi1.norm(dim=0) ** 2 + psi2.norm(dim=0) ** 2) / (psi1.shape[0] + psi2.shape[0])
                eigenvalues_ = ((psi1 / norm_.sqrt()) * (psi2 / norm_.sqrt())).mean(0)
                if self.num_calls == 0:
                    self.eigennorm_sqr.copy_(norm_.data)
                    self.eigenvalues.copy_(eigenvalues_.data)
                else:
                    self.eigennorm_sqr.mul_(self.momentum).add_(
                        norm_.data, alpha = 1-self.momentum) 
                    self.eigenvalues.mul_(self.momentum).add_(
                        eigenvalues_.data, alpha = 1-self.momentum)
                self.num_calls += 1

        norm_ = (psi1.norm(dim=0) ** 2 + psi2.norm(dim=0) ** 2).sqrt().clamp(min=1e-6)
        psi1 = psi1.div(norm_)
        psi2 = psi2.div(norm_)

        K = self.get_K(self.args.coeff, index1, index2).to(self.device) # (B * num_GPUs / 2 ) x (B * num_GPUs / 2 )
        if is_master(): self.logger.debug(f'K.shape: {K.shape}.')

        # todo:
        # 1. psi1 and psi2 be independent from the batchsize
        # 2. formulate term1 and term2 in eq.8
        # 3. setup the training

        #################################################
        # psi_K_psi_diag = (psi1 * psi2).sum(0).view(-1, 1)
        psi_K_psi_diag = 0.5 * (torch.diag(psi1.T @ K @ psi2).view(-1, 1) + torch.diag(psi2.T @ K.T @ psi1).view(-1, 1))

        if is_master(): self.logger.debug(f'psi_K_psi_diag.shape: {psi_K_psi_diag.shape}.') # (k, 1)
        psi_K_psi = 0.5 * ((psi1.detach().T) @ K @ (psi2) + (psi2.detach().T) @ K.T @ (psi1))
        loss = - psi_K_psi_diag.sum()
        reg = ((psi_K_psi) ** 2).triu(1).sum()
        #################################################

        # R1 = (psi1.detach().T) @ K @ (psi2) / (2 * K.shape[0])
        # R2 = (psi1.detach().T + psi2.detach().T) @ K @ (psi1 + psi2) / (2 * K.shape[0]) # todo

        # if is_master(): self.logger.debug(f'R1: \n {R1}')
        # if is_master(): self.logger.debug(f'R2: \n {R2}')

        # psi_K_psi_diag = (psi1 * psi2 * 2 + psi1 * psi1 + psi2 * psi2).sum(0).view(-1, 1)
        # psi_K_psi = (psi1.detach().T + psi2.detach().T) @ (psi1 + psi2)

        # loss = - psi_K_psi_diag.sum()
        # reg = ((psi_K_psi) ** 2).triu(1).sum()

        # loss /= psi_K_psi_diag.numel()
        # reg /= psi_K_psi_diag.numel()

        return loss, reg



class NeuralEigenFunctions(nn.Module):
    def __init__(self, args, logger, device, momentum=0.9, normalize_over=[0]):
        super(NeuralEigenFunctions, self).__init__()
        self.momentum = momentum
        self.normalize_over = normalize_over
        # self.fn = ParallelMLP(input_size, output_size, k, num_layers, hidden_size, nonlinearity)
        self.args = args
        self.logger = logger
        self.device = device
        self.k = self.args.k
        self.fn = self._build_fn()
        self.register_buffer('eigennorm', torch.zeros(self.k))
        self.register_buffer('num_calls', torch.Tensor([0]))
        self.eigenvalues = None
    
    def _build_fn(self):
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()
        self.projector = nn.Linear(2048, self.k, bias=True)
        layers = [self.backbone, self.projector]
        return nn.Sequential(*layers)

    def forward(self, x):
        ret_raw = self.fn(x).squeeze()
        if self.training:
            norm_ = ret_raw.norm(dim=self.normalize_over) / math.sqrt(
                np.prod([ret_raw.shape[dim] for dim in self.normalize_over]))
            with torch.no_grad():
                if self.num_calls == 0:
                    self.eigennorm.copy_(norm_.data)
                else:
                    self.eigennorm.mul_(self.momentum).add_(
                        norm_.data, alpha = 1-self.momentum)
                self.num_calls += 1
        else:
            norm_ = self.eigennorm
        return ret_raw / norm_
    
    def get_eigenvalue(self):
        return self.eigenvalues.float()

    def loss(self, y, index):
        B = y.shape[0]
        z1, z2 = self.forward(y).chunk(2, dim=0)
        psi1 = gather_from_all(z1)
        psi2 = gather_from_all(z2)
        index = gather_from_all(index).reshape(-1, B) # (num_device, B)
        index1, index2 = index.chunk(2, dim=1)
        index1 = index1.reshape(1, -1).float()
        index2 = index2.reshape(1, -1).float()
        K = self.get_K(self.args.coeff, index1, index2).to(self.device)

        with torch.no_grad():
            K_psis = K @ psi2
            psis_K_psis = psi1.T @ K_psis
            mask = torch.eye(self.args.k, device=self.device) - \
                (psis_K_psis / psis_K_psis.diag()).tril(diagonal=-1).T
            grad = K_psis @ mask
            if self.eigenvalues is None:
                self.eigenvalues = psis_K_psis.diag() / (int(B/2)**2)
            else:
                self.eigenvalues.mul_(0.9).add_(psis_K_psis.diag() / (int(B/2)**2), alpha=0.1)
            # psi1.backward(-grad)

        return psi1, grad

    def get_K(self, coeff, index1, index2):
        if index1.ndim == 1:
            row_vector = index1.unsqueeze(0)
            col_vector = index2.unsqueeze(0)
        else:
            row_vector = index1
            col_vector = index2
        # K = torch.abs(row_vector - col_vector.T)
        K = torch.exp(-coeff * torch.abs(row_vector - col_vector.T))
        return K
