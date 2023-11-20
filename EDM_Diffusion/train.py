# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pickle

# from myutilities import TConditionalBatchNorm1d, CustomSequential, DDPMDiffuser, LearnedSinusoidalPosEmb, adjust_learning_rate2

# from torchvision import transforms
# from torchvision.utils import save_image
# from model import SongUNetWithMLP
# from dataset import CIFAR10Dataset
# from torch.utils.data import DataLoader
# import os
# import matplotlib.pyplot as plt


# def process_input(images, sigma, labels=None, augment_pipe=None):
#     sigma = sigma.unsqueeze(1).unsqueeze(2).unsqueeze(3)

#     sigma_data=0.5
#     weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
#     y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

#     n = torch.randn_like(y) * sigma
#     return y + n

# def cal_input(x, sigma, sigma_data=0.5, label_dim=10, class_labels=None, force_fp32=False):
#     x = x.to(torch.float32)
#     sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
#     class_labels = None if label_dim == 0 else torch.zeros([1, label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, label_dim)
#     dtype = torch.float32
    
#     c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
#     c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
#     c_in = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
#     c_noise = sigma.log() / 4

#     return (c_in * x).to(dtype), c_noise.flatten(), class_labels 

# class DiffusionEF(nn.Module):
#     def __init__(self, args, image_size, num_timesteps, num_classes, learned_sinusoidal_dim=16):

#         super().__init__()
#         self.net = SongUNetWithMLP(
#                     img_resolution      = image_size,
#                     in_channels         = 3,                        # Number of color channels at input.
#                     out_channels        = 3,                       # Number of color channels at output.
#                     label_dim           = 10,            # Number of class labels, 0 = unconditional.
#                     augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

#                     model_channels      = 128,          # Base multiplier for the number of channels.
#                     channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
#                     channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
#                     num_blocks          = 4,            # Number of residual blocks per resolution.
#                     attn_resolutions    = [16],         # List of resolutions with self-attention.
#                     dropout             = 0.10,         # Dropout probability of intermediate activations.
#                     label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
                    
#                     embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
#                     channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
#                     encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
#                     decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
#                     resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
 
#                     mlp_hidden_dims     = 512,
#                     mlp_num_layers      = 3,
#                     mlp_activation      = nn.ReLU(),
#         )

#         layers = []
#         layers.append(nn.Linear(256, 256))
#         layers.append(nn.BatchNorm1d(256))
#         layers.append(nn.GELU())
        
#         layers.append(nn.Linear(256, 256))
#         layers.append(nn.BatchNorm1d(256))
#         layers.append(nn.GELU())

#         layers.append(nn.Linear(256, 256))
#         layers.append(nn.BatchNorm1d(256))
#         layers.append(nn.GELU())

#         layers.append(nn.Linear(256, 128))
#         layers.append(nn.BatchNorm1d(128))
#         layers.append(nn.GELU())

#         self.layers = CustomSequential(*layers, num_hidden_channels= 512)
#         self.head = nn.Linear(128, args.k)
#         self.online_clf_head = nn.Linear(args.k, num_classes) 
#         self.args = args
#         self.momentum = args.momentum
#         self.training = True
#         ###########build up time dependence###########
#         self.time_dependent = args.time_dependent
#         if self.time_dependent:
#             sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
#             fourier_dim = learned_sinusoidal_dim + 1

#             self.t_embed_fn = nn.Sequential(
#                 sinu_pos_emb,
#                 nn.Linear(fourier_dim, args.hidden_channels),
#                 nn.GELU(),
#                 nn.Linear(args.hidden_channels, args.hidden_channels),
#             )
#         else:
#             self.t_embed_fn = lambda x: 0

#         self.load_weight_u = True
#         #######load wieght#########
#         # if self.load_weight_u == True:
#         #     print('loading pretrained weight')
#             # unet_state_dict = torch.load('edm-cifar10-32x32-cond-vp.pkl')
#             # self.net.load_state_dict(unet_state_dict)  
#         # for param in self.net.parameters():
#         #     param.requires_grad = False
            
#         self.register_buffer('eigennorm_sqr', torch.zeros(num_timesteps, args.k))
#         self.register_buffer('eigenvalues', torch.zeros(num_timesteps, args.k))
#         self.register_buffer('num_calls', torch.Tensor([0]))

#     def forward(self, x1, x2, sigma, t, logsnr, labels=None, return_psi=False):
#         cin_x1, noise_flt1, cls_lbs1 = cal_input(x1, sigma) 
#         cin_x2, noise_flt2, cls_lbs2 = cal_input(x2, sigma) 
#         with torch.no_grad():
#             h1 = self.net(cin_x1, noise_flt1, cls_lbs1)
#             h2 = self.net(cin_x2, noise_flt2, cls_lbs2)
#         assert h1.shape[0] == h2.shape[0]
#         h1 = h1.to(torch.float32)
#         h2 = h2.to(torch.float32)
#         t = t.item()
#         #t = t.item()
#         ##########################################################################
#         t_embedding = self.t_embed_fn(logsnr)
#         #t_embedding = self.t_embed_fn(None)
#         psi = self.head(self.layers(torch.cat([h1, h2], 0), t, t_embedding))
        
        
#         if self.training:
#             norm_sqr = psi.norm(dim=0) ** 2 / psi.shape[0]
#             with torch.no_grad():
#                 if self.num_calls == 0:
#                     self.eigennorm_sqr[t].copy_(norm_sqr.data)
#                 else:
#                     self.eigennorm_sqr[t].mul_(self.momentum).add_(
#                         norm_sqr.data, alpha = 1-self.momentum)
            
#             norm_ = norm_sqr.clamp(min=0).sqrt().clamp(min=1e-6)
#         else:
#             norm_ = self.eigennorm_sqr[t].clamp(min=0).sqrt().clamp(min=1e-6)

#         psi = psi.div(norm_)
#         psi1, psi2 = psi.chunk(2, dim=0)
#         if return_psi: return psi1, psi2

#         psi_K_psi_diag = (psi1 * psi2).sum(0).view(-1, 1)     
#         if self.training:
#             with torch.no_grad():
#                 eigenvalues_ = psi_K_psi_diag.view(-1) / psi1.shape[0]
#                 if self.num_calls == 0:
#                     self.eigenvalues[t].copy_(eigenvalues_.data)
#                 else:
#                     self.eigenvalues[t].mul_(self.momentum).add_(
#                         eigenvalues_.data, alpha = 1-self.momentum)
#                 self.num_calls += 1 

#         psi2_d_K_psi1 = psi2.detach().T @ psi1
#         psi1_d_K_psi2 = psi1.detach().T @ psi2

#         scale = 1. / psi.shape[0]
#         loss = - (psi_K_psi_diag * scale).sum() * 2
#         reg = ((psi2_d_K_psi1) ** 2 * scale).triu(1).sum() \
#             + ((psi1_d_K_psi2) ** 2 * scale).triu(1).sum()

#         loss /= psi_K_psi_diag.numel()
#         reg /= psi_K_psi_diag.numel()

#         # train the classification head only on clean data
#         #P_m = -1.2
#         if t < 100 :
#             logits = self.online_clf_head(psi1.detach())
#             cls_loss = F.cross_entropy(logits, labels.to(torch.long))
#             acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)
#         else:
#             cls_loss, acc = torch.Tensor([0.]).to(x1.device), None

#         return loss, reg, cls_loss, acc
            
    
#     def test_acc(self, x1, x2, sigma, t, logsnr, labels):

#         with torch.no_grad():
#             psi1, psi2 = self.forward(x1, x2, sigma, t, logsnr, return_psi=True)
#             logits1 = self.online_clf_head(psi1.detach())
#             logits2 = self.online_clf_head(psi2.detach()) 

#         acc = (torch.sum(torch.eq(torch.argmax(logits1, dim=1), labels))+ 
#                 torch.sum(torch.eq(torch.argmax(logits2, dim=1), labels)))/ (logits1.size(0)+logits2.size(0))

#         return acc



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Learn diffusion eigenfunctions on ImageNet ')
#     parser.add_argument('--device', type=int, default=0)
#     parser.add_argument('--log_steps', type=int, default=50)

#     # for the size of image
#     parser.add_argument('--resolution', type=int, default=256)

#     # for specifying MLP
#     parser.add_argument('--num_layers', type=int, default=5)
#     parser.add_argument('--hidden_channels', type=int, default=512)
#     parser.add_argument('--norm_type', default='bn', type=str)
#     parser.add_argument('--act_type', default='relu', type=str)
#     parser.add_argument('--affine_condition_on_time', default=False, action='store_true')

#     # opt configs
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--test_batch_size', type=int, default=16)
#     parser.add_argument('--lr', type=float, default=0.001)
#     parser.add_argument('--min_lr', type=float, default=None, metavar='LR')
#     parser.add_argument('--weight_decay', type=float, default=1e-5)
#     parser.add_argument('--warmup_steps', type=int, default=1000, metavar='N', help='iterations to warmup LR')
#     parser.add_argument('--num_epochs', type=int, default=10000)
#     parser.add_argument('--accum_iter', type=int, default=1)
#     parser.add_argument('--opt', type=str, default='adam')

#     # for neuralef
#     parser.add_argument('--alpha', default=1, type=float)
#     parser.add_argument('--k', default=64, type=int)
#     parser.add_argument('--momentum', default=.9, type=float)
#     parser.add_argument('--time_dependent', default=False, action='store_true')

#     args = parser.parse_args()
#     if args.min_lr is None:
#         args.min_lr = args.lr * 0.001
#     print(args)

#     device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
#     device = torch.device(device)
#     torch.backends.cudnn.benchmark = True


#     # mydiffusionEF implement
#     # define model and initialize with pre-trained model
#     diffuser = DDPMDiffuser().to(device) #the noise_schedule is linear
#     model=DiffusionEF(image_size=(args.batch_size, 3, args.resolution, args.resolution), args=args, num_timesteps=1000, num_classes=10).to(device)
#     print(model)
#     # print("# params: {}".format(sum([p.numel() for p in model.parameters() if p.requires_grad])))

#     ckpt_path = './ckpt/'
#     # construct train dataset
#     cifar_dataset_train = CIFAR10Dataset(root_dir='/home/lhy/Projects/EDM_Diffusion/data', train=True, transform=None)
#     cifar_dataset_test = CIFAR10Dataset(root_dir='/home/lhy/Projects/EDM_Diffusion/data', train=False, transform=None)
#     train_data_loader= DataLoader(cifar_dataset_train, batch_size=args.batch_size, shuffle=True)
#     test_data_loader = DataLoader(cifar_dataset_test, batch_size=args.test_batch_size, shuffle=False)

#     # train the DiffusionEF model
#     model.train()
#     losses, regs, cls_losses, accs, n_steps, n_steps_clf = 0, 0, 0, 0, 0, 1e-8
#     if args.opt == 'lars':
#         optimizer = LARS(model.parameters(),
#                          lr=args.lr, weight_decay=args.weight_decay,
#                          weight_decay_filter=exclude_bias_and_norm,
#                          lars_adaptation_filter=exclude_bias_and_norm)
#     elif args.opt == 'adam':
#         optimizer = torch.optim.AdamW(model.parameters(), 
#                                       lr=args.lr, weight_decay=args.weight_decay)
#     else:
#         optimizer = torch.optim.SGD(model.parameters(), 
#                                     lr=args.lr, momentum=.9, 
#                                     weight_decay=args.weight_decay)
#     optimizer.zero_grad()

#     loss_his = []
#     cls_loss_his = []
#     best_acc = -10
#     for epoch in range(args.num_epochs):
        
#         for (image1, image2), labels, classes_name in train_data_loader:
#             n_steps += 1
#             # cosine decay lr with warmup
#             lr = adjust_learning_rate2(args.lr, optimizer, n_steps, 
#                                     args.num_epochs, args.warmup_steps, args.min_lr)
            
#             image1 = image1.to(device, non_blocking=True)
#             image2 = image2.to(device, non_blocking=True)
#             labels = labels.to(device, non_blocking=True)
#             t = diffuser.sample_t().to(device)
            
#             # diffuse the training data
#             #image1 = diffuser(image1, t)
#             #image2 = diffuser(image2, t)

#             P_mean=-1.2
#             P_std=1.2
#             rnd_normal = torch.randn([image1.shape[0], 1, 1, 1], device=image1.device)

#             sigma = (rnd_normal * P_std + P_mean).exp()

#             y_pls_n1 = process_input(image1, labels, sigma)
#             y_pls_n2 = process_input(image2, labels, sigma)
#             y_pls_n1 = y_pls_n1.permute(0, 3, 1, 2)
#             y_pls_n2 = y_pls_n2.permute(0, 3, 1, 2)
#             ##########################################################################
#             loss, reg, cls_loss, acc = model.forward(y_pls_n1, y_pls_n2, sigma, t, diffuser.logsnr(t), labels)
#             (loss + reg * args.alpha + cls_loss).div(args.accum_iter).backward()
#             if epoch % args.accum_iter == 0:
#                 optimizer.step()
#                 optimizer.zero_grad()

#             losses += loss.detach().item()
#             regs += reg.detach().item()

#             if acc is not None:
#                 cls_losses += cls_loss.detach().item()
#                 accs += acc.detach().item()
#                 n_steps_clf += 1
#             if n_steps%20 ==0:
#                 print('Epoch: {}, t: {}, LR: {:.4f}, Loss: {:.4f}, Reg: {:.4f}, CLS_Loss: {:.4f}, Train acc: {:.2f}%'.format(
#                     epoch, t.item(), lr, losses / n_steps, regs / n_steps, cls_losses / n_steps_clf, 100 * accs / n_steps_clf))
#                 loss_his.append(losses / n_steps)
#                 cls_loss_his.append(cls_losses / n_steps_clf)

#         if epoch % 2 == 0:
#             model.eval()
#             # test_data_loader_iter = iter(test_data_loader)
#             # (test_image1, test_image2), test_labels, test_classes_names = next(test_data_loader_iter)
#             # test_image1 = test_image1.to(device, non_blocking=True)
#             # test_image2 = test_image2.to(device, non_blocking=True)
#             # test_labels = test_labels.to(device, non_blocking=True)
#             # t0 = torch.Tensor([0.]).long().to(device)

#             # P_mean=-1.2
#             # P_std=1.2
#             # rnd_normal = torch.randn([image1.shape[0], 1, 1, 1], device=image1.device)

#             # sigma = (rnd_normal * P_std + P_mean).exp()

#             # y_pls_n1 = process_input(image1, labels, sigma)
#             # y_pls_n2 = process_input(image2, labels, sigma)
#             # y_pls_n1 = y_pls_n1.permute(0, 3, 1, 2)
#             # y_pls_n2 = y_pls_n2.permute(0, 3, 1, 2)

#             # test_acc = model.test_acc(y_pls_n1, y_pls_n2, sigma, t0, diffuser.logsnr(t0), test_labels)
#             # n_steps+=1
#             # print('Epoch: {}, Loss: {:.4f}, Reg: {:.4f}, CLS_Loss: {:.4f}, Train acc: {:.2f}%, Test acc: {:.2f}%'.format(
#             #     epoch, losses / n_steps, regs / n_steps, cls_losses / n_steps_clf, 100 * accs / n_steps_clf, 100 * test_acc))
            
#             # if 100 * test_acc > best_acc:
#             #     best_acc = 100 * test_acc
#             #     torch.save(model, ckpt_path + 'best_test_model.pth')

#             losses, regs, cls_losses, accs, n_steps, n_steps_clf = 0, 0, 0, 0, 0, 1e-8
#             model.train()
            
#     plt.plot(loss_his, color='blue')
#     plt.savefig('loss.png')

#     plt.plot(cls_loss_his, color='green')
#     plt.savefig('cls_loss.png')



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
import matplotlib.pyplot as plt
from tqdm import tqdm

P_mean=-1.2
P_std=1.2

def process_input(images, sigma, labels=None, augment_pipe=None):
    # rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)

    sigma_data=0.5
    y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
    y = y.to(args.device)
    n = torch.randn_like(y) * sigma
    n = n.to(args.device)
    return y + n

def cal_input(x, sigma, sigma_data=0.5, label_dim=None, class_labels=None, force_fp32=False, net=None):
    x = x.to(torch.float32)
    sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
    class_labels = None if label_dim == 0 else torch.zeros([1, label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1,label_dim)
    dtype = torch.float32
    
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
    c_in = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
    c_noise = sigma.log() / 4
    
    F_x = net((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels)
    assert F_x.dtype == dtype
    D_x = c_skip * x + c_out * F_x.to(torch.float32)#denoised model
    
    return (c_in * x).to(dtype), c_noise.flatten(), class_labels, D_x

num_test = 1
logf = open(f'./logs/logits_true{num_test}.txt', 'w')
log_loss = open(f'./logs/Loss_log_attention{num_test}.txt', 'w')
model_arch = open(f'./logs/model{num_test}.txt', 'w')
class DiffusionEF(nn.Module):
    def __init__(self, network_pkl, args, image_size, num_sigmas, num_classes, learned_sinusoidal_dim=16, mlp_hidden_dims = 1024, mlp_num_layers = 18, mlp_activation = nn.ReLU()):

        super().__init__()

        dist.print0(f'Loading network from "{network_pkl}"...')
        with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f)['ema']

        self.net = net.model
        for params in self.net.parameters():
            params.requires_grad = False

        self.norm_Silu = self.norm_silu(input_channels = 3)
        ###use resnet as the regressor
        # self.Resnet = torchvision.models.resnet50(pretrained=False)
        # self.Resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # num_features = self.Resnet.fc.in_features
        # self.Resnet.fc = nn.Linear(num_features, 512)

        self.mlp = self.build_mlp(input_dim=3 * (32 ** 2),
                            hidden_dim=mlp_hidden_dims,
                            output_dim=512,
                            num_layers=mlp_num_layers,
                            activation=mlp_activation)

        # self.mlp_c = self.build_mlp(input_dim=args.k,
        #                     hidden_dim=mlp_hidden_dims,
        #                     output_dim=args.k,
        #                     num_layers=mlp_num_layers,
        #                     activation=mlp_activation)
        
        #self.attention = Attention(input_size=3)

        self.head = nn.Linear(512, args.k)
        #self.head = nn.Linear(3*(32 * 32), args.k)
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
        class_labels = torch.zeros(10, labels.shape[0], device=args.device)
        for i in range (labels.shape[0]):
            class_labels[int(labels[i]),i] = 1

        cin_x1, noise_flt1, cls_lbs1, D1 = cal_input(x1, sigma, label_dim=10, class_labels=class_labels, net=self.net) 
        cin_x2, noise_flt2, cls_lbs2, D2 = cal_input(x2, sigma, label_dim=10, class_labels=class_labels, net=self.net) 
        #print(x1.shape, cls_lbs1.shape)
        with torch.no_grad():
            # h1 = self.net(cin_x1, noise_flt1, labels)
            # h2 = self.net(cin_x2, noise_flt2, labels)
            #print(cin_x1.shape, noise_flt1.shape, cls_lbs1.shape)
            h1 = self.net(cin_x1, noise_flt1, class_labels=cls_lbs1)
            h2 = self.net(cin_x2, noise_flt2, class_labels=cls_lbs2)
            z1 = self.norm_Silu(h1)
            z2 = self.norm_Silu(h2)
            # z1 = self.norm_Silu(D1)
            # z2 = self.norm_Silu(D2)
        assert z1.shape[0] == z2.shape[0]
        

        z1 = z1.to(torch.float32)
        z2 = z2.to(torch.float32)

        rsp1 = z1.reshape(z1.shape[0], -1)
        rsp2 = z2.reshape(z2.shape[0], -1)

        # h = torch.cat([z1.view(z1.shape[0],-1), z2.view(z1.shape[0],-1)], 0)
        h = torch.cat([rsp1, rsp2], 0)
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
##################################
            
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
        #repre = psi1.detach()

        # train the classification head only on clean data
        if index[0] <= 20 :
            #logits = self.online_clf_head(psi1.detach())
            
            logits = self.online_clf_head(psi1.detach())
            #logits = self.online_clf_head(psi1)
            #print(h.shape, psi1.shape, x1.shape)
            cls_loss = F.cross_entropy(logits, labels.to(torch.long))

            print('predict class and labels: ',torch.argmax(logits, dim=1), file=logf)
            print(labels.to(torch.long), file=logf)

            acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)
        else:
            cls_loss, acc = torch.Tensor([0.]).to(x1.device), None


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
    parser.add_argument('--alpha', default=0.005, type=float)
    parser.add_argument('--k', default=64, type=int)
    parser.add_argument('--momentum', default=.9, type=float)
    parser.add_argument('--time_dependent', default=False, action='store_true')

    args = parser.parse_args()
    if args.min_lr is None:
        args.min_lr = args.lr * 0.001
    print(args)
    print(args, file=model_arch)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    torch.backends.cudnn.benchmark = True

    # discretization of noise added
    #num_sigmas = 1000
    num_steps = 100
    sigma_max = 80
    sigma_min = 0.002
    rho = 7

    #rnd_normal = torch.randn([num_sigmas, 1, 1, 1])
    #normal * P_std + P_mean).exp()
    #sigmas, sorted_indices = torch.sort(sigmas, dim=0, descending=False)
    # print(sigmas)


    # mydiffusionEF implement
    # define model and initialize with pre-trained model
    # diffuser = DDPMDiffuser().to(device) #the noise_schedule is linear
    network_pkl = '/home/lhy/Projects/EDM_Diffusion/edm-cifar10-32x32-cond-vp.pkl' # vp for SongNet DDPM++
    model=DiffusionEF(network_pkl, image_size=(args.batch_size, 3, args.resolution, args.resolution), args=args, num_sigmas=100, num_classes=10).to(device)
    # download pretrained model with SongNet DDPM++
    print(model, file=model_arch)
    print(model)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=args.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])
    t_steps = t_steps[:-1]
    t_steps = t_steps.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    sigmas = t_steps
    sigmas, sorted_indices = torch.sort(sigmas, dim=0, descending=False)
    
    #assert 1 + 1 == 3

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

    loss_his = []
    reg_his = []
    best_loss = 1e8
    for epoch in range(args.num_epochs):        
        for (image1, image2), labels, classes_name in tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="batch"):
            n_steps += 1
            # cosine decay lr with warmup
            lr = adjust_learning_rate2(args.lr, optimizer, n_steps, 
                                    args.num_epochs, args.warmup_steps, args.min_lr)
            #lr = args.lr
            optimizer.zero_grad()
            image1 = image1.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            #index = random.sample(range(num_sigmas), 1)
            index = random.sample(range(100), 1)

            print('index: ', index[0])
            sigma = sigmas[index]
            print('sigma: ', sigma)
                
            # rnd_normal = torch.randn([image1.shape[0], 1, 1, 1], device=image1.device)
            # sigma = (rnd_normal * P_std + P_mean).exp()
            
            y_pls_n1 = process_input(image1, sigma, labels)
            y_pls_n2 = process_input(image2, sigma, labels)
            y_pls_n1 = y_pls_n1.permute(0, 3, 1, 2)
            y_pls_n2 = y_pls_n2.permute(0, 3, 1, 2)

            # for i in range(100):
            #     index[0] = i
            #     print('index: ', index[0])
            #     sigma = sigmas[index]
            #     print('sigma: ', sigma)
            #     y_pls_n1 = process_input(image1, sigma, labels)
            #     y_pls_n2 = process_input(image2, sigma, labels)
            #     y_pls_n1 = y_pls_n1.permute(0, 3, 1, 2)
            #     y_pls_n2 = y_pls_n2.permute(0, 3, 1, 2)
            #     if(index[0]<100):
            #         tmp1 = y_pls_n1
            #         print(tmp1.shape)
            #         tmp = tmp1[0]
            #         print(tmp.shape)
            #         tmp = tmp.clamp(0, 1)
            #         tmp = tmp * 255
            #         tmp = tmp.byte()
            #         image = transforms.ToPILImage()(tmp)
            #         image.save(f"./noises/image_noise_index{i}and_sigma{float(sigma)}.jpg")
            # assert index[0] == 3
            
            
            ##########################################################################
            # loss, reg, cls_loss, acc = model.forward(y_pls_n1, y_pls_n2, sigma, t, diffuser.logsnr(t), labels)
            loss, reg, cls_loss, acc, tot_l  = model.forward(y_pls_n1, y_pls_n2, sigma, index, labels)

            # for params in model.parameters():
            #     params.requires_grad = True
            # for params in model.mlp_c.parameters():
            #     params.requires_grad = False
            # for params in model.online_clf_head.parameters():
            #     params.requires_grad = 
            #print(w1, w2, w3, (loss + reg * args.alpha + cls_loss).shape, (loss + reg * args.alpha + cls_loss).shape == torch.Size([]) )
            #(loss + reg * args.alpha + cls_loss).div(args.accum_iter).backward() 
            optimizer.zero_grad()
            # if index[0] < 20:
            #(loss + args.alpha * reg + cls_loss).div(args.accum_iter).backward()    
            (loss + args.alpha * reg + cls_loss).div(args.accum_iter).backward()  
            #(cls_loss).div(args.accum_iter).backward()  
            # else: 
            #     (loss + args.alpha * reg).div(args.accum_iter).backward() 
            # if best_loss > float(loss + reg * args.alpha):
            #     best_loss = float(loss + reg * args.alpha)
            #     torch.save(model.state_dict(), './pt/mini_loss.pt')
                
            if epoch % args.accum_iter == 0:
               optimizer.step()
               

            losses += loss.detach().item()
            regs += reg.detach().item()

            if acc is not None:
                cls_losses += cls_loss.detach().item()
                accs += acc.detach().item()
                n_steps_clf += 1
            if n_steps%20 == 0:
                loss_his.append(losses / n_steps)
                reg_his.append(regs / n_steps)
                print('Epoch: {}, sigma: {}, LR: {:.4f}, Loss: {:.4f}, Reg: {:.4f}, CLS_Loss: {:.4f}, Train acc: {:.2f}%'.format(
                    epoch, sigma.item(), lr, losses / n_steps, regs / n_steps, cls_losses / n_steps_clf, 100 * accs / n_steps_clf))
                print('Epoch: {}, sigma: {}, LR: {:.4f}, Loss: {:.4f}, Reg: {:.4f}, CLS_Loss: {:.4f}, Train acc: {:.2f}%'.format(
                    epoch, sigma.item(), lr, losses / n_steps, regs / n_steps, cls_losses / n_steps_clf, 100 * accs / n_steps_clf), file=log_loss)
        
        # if epoch % 5 == 0:
        #     torch.save(model.state_dict(), f'./pt/model_{epoch}.pt')
        losses, regs, cls_losses, accs, n_steps, n_steps_clf = 0, 0, 0, 0, 0, 1e-8
        # if epoch % 2 == 0:
            # print('=======> Testing <=======')
            # model.eval()
            # total_acc = 0
            # count_acc = 0
            # for (image1, image2), labels, classes_name in test_data_loader:
            #     image1 = image1.to(device, non_blocking=True)
            #     image2 = image2.to(device, non_blocking=True)
            #     labels = labels.to(device, non_blocking=True)

            #     index = random.sample(range(num_sigmas), 1)
            #     if index[0] < 100:
            #         sigma = sigmas[index]
            #         y_pls_n1 = process_input(image1, sigma, labels)
            #         y_pls_n2 = process_input(image2, sigma, labels)
            #         y_pls_n1 = y_pls_n1.permute(0, 3, 1, 2)
            #         y_pls_n2 = y_pls_n2.permute(0, 3, 1, 2)
            #         total_acc += model.test_acc(y_pls_n1, y_pls_n2, sigma.to(device), index, labels)
            #         count_acc += 1.0

            # print('Test acc: ', float(1.0 * total_acc / count_acc))
            # print('Test acc: ', float(1.0 * total_acc / count_acc), file=log_loss)
            # losses, regs, cls_losses, accs, n_steps, n_steps_clf = 0, 0, 0, 0, 0, 1e-8
            # model.train()
            
    plt.plot(loss_his, color='blue')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.savefig(f'./figs/loss{num_test}.png')
    plt.clf()

    plt.plot(reg_his, color='green')
    plt.xlabel('steps')
    plt.ylabel('reg')
    plt.savefig(f'./figs/reg{num_test}.png')
