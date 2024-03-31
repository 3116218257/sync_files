import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm, trange

from dataloader import prepare_dataloader, prepare_video_dataset, video_dataloader
from utils.dist import is_master
from model import NeuralEFCLR, NeuralEigenFunctions
import numpy as np

class Trainer_Ibra():
    def __init__(self, args, logger, log_dir, ckpt_dir, sample_dir):
        self.args = args
        self.logger = logger
        self.device = int(os.environ["LOCAL_RANK"])
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.sample_dir = sample_dir
        self.num_iters = 0
        self.history_iters = 0
        if is_master():
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        # for validation use
        if is_master():
            ibra_path = '/home/jiachun/codebase/jkernel/data/walker_tensor'
            self.x_train = torch.load(ibra_path)[:self.args.val_batch_size].to(self.device)
            self.index_train = torch.arange(self.args.val_batch_size).to(self.device)
            self.x_test = torch.load(ibra_path)[1000:1000+self.args.val_batch_size].to(self.device)
            self.index_test = torch.arange(self.args.val_batch_size).to(self.device)
            self.logger.info('Trainer built.')
    
    def _build_dataloader(self):
        self.dataloader = prepare_dataloader('walker', self.args.batch_size_per_gpu, mode='train')
        self.valloader = prepare_dataloader('walker', batch_size=self.args.val_batch_size, mode='val')

    def _build_model(self):
        model = NeuralEFCLR(self.args, self.logger, self.device).cuda(self.device)
        # model = NeuralEigenFunctions(self.args, self.logger, self.device).cuda(self.device)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.device])

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                            lr=0.0001,
                                            weight_decay=0.000,
                                            betas=(0.9, 0.999),
                                            amsgrad=False,
                                            eps=0.00000001)
    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                            lr=0.0001,
                                            weight_decay=0.000,
                                            betas=(0.9, 0.999),
                                            amsgrad=False,
                                            eps=0.00000001)
    @torch.no_grad()
    def validation(self):
        self.model.eval()
        # x, index = next(iter(self.valloader))
        # x = x.to(self.device)
        # index = index.to(self.device)
        # index1, index2 = index.chunk(2)

        # training data
        r = self.model.module.backbone(torch.cat((self.x_train, self.x_train), dim=0))
        psi1, psi2 = self.model.module.projector(r).chunk(2, dim=0)
        trueK_train = self.model.module.get_K(self.args.coeff, self.index_train, self.index_train)
        estiK_train = psi1 @ torch.diag(self.model.module.eigenvalues.data) @ psi2.T
        fig, axes = plt.subplots(2, 2)
        im = axes[0, 0].imshow(trueK_train.squeeze(dim=0).detach().cpu())
        im = axes[0, 1].imshow(estiK_train.squeeze(dim=0).detach().cpu())
        # test data
        r = self.model.module.backbone(torch.cat((self.x_test, self.x_test), dim=0))
        psi1, psi2 = self.model.module.projector(r).chunk(2, dim=0)
        trueK_test = self.model.module.get_K(self.args.coeff, self.index_test, self.index_test)
        estiK_test = psi1 @ torch.diag(self.model.module.eigenvalues.data) @ psi2.T
        im = axes[1, 0].imshow(trueK_test.squeeze(dim=0).detach().cpu())
        im = axes[1, 1].imshow(estiK_test.squeeze(dim=0).detach().cpu())

        fig.colorbar(im, ax=axes.ravel().tolist())
        self.writer.add_figure('trueK and estiK', plt.gcf(), self.num_iters)
    def save(self):
        state = {
            'backbone': self.model.module.backbone,
            'projector': self.model.module.projector,
            'num_iters': self.num_iters
        }
        torch.save(state, f'./ckpt/{self.num_iters}.pt')
        self.logger.info('Save done.')

    def train(self):
        total_iters = self.args.total_iters
        done = False
        epoch = 1
        scaler = torch.cuda.amp.GradScaler()
        with tqdm(total=total_iters) as pbar:
            pbar.update(self.num_iters)
            while not done:
                if torch.cuda.device_count() > 1: self.dataloader.sampler.set_epoch(epoch)
                for x, index in self.dataloader:
                    self.model.train()
                    # self.logger.debug(f'rank: {self.device}, index: {index}')
                    self.optimizer.zero_grad()

                    with torch.cuda.amp.autocast():
                        loss, reg = self.model(x, index)
                        scaler.scale(loss + reg * self.args.alpha).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    if is_master():
                        self.writer.add_scalar('loss', loss.item(), self.num_iters)
                        self.writer.add_scalar('reg', reg.item(), self.num_iters)
                        self.writer.add_scalar('total', loss.item() +  self.args.alpha * reg.item(), self.num_iters)


                    # (loss + self.args.alpha * reg).backward()
                    # self.optimizer.step()
                    pbar.update(1)
                    self.num_iters += 1
                    if self.num_iters % 20 == 0 and is_master():
                        self.validation()
                        # self.save()
                epoch += 1

        if is_master(): self.logger.info('Training done.')
        
        
        
class Trainer_football():
    def __init__(self, args, logger, log_dir, ckpt_dir, sample_dir):
        self.args = args
        self.logger = logger
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.sample_dir = sample_dir
        self.num_iters = 0
        self.history_iters = 0
        self.root_path = args.data_root
        if is_master():
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        self._build_dataset()
        self._build_model()
        self._build_optimizer()
        if is_master():
            self.logger.info('Trainer built.')

    
    def _build_dataset(self):
        # self.dataloader = prepare_dataloader('walker', self.args.batch_size_per_gpu, mode='train')
        # self.valloader = prepare_dataloader('walker', batch_size=self.args.val_batch_size, mode='val')
        self.train_data = prepare_video_dataset(video_dir=self.root_path, mode='train')
        self.val_data = prepare_video_dataset(video_dir=self.root_path, mode='val')

    def _build_model(self):
        # print(self.device)
        model = NeuralEFCLR(self.args, self.logger, self.device).to(self.device)
        # model = NeuralEigenFunctions(self.args, self.logger, self.device).cuda(self.device)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.device])

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                            lr=0.0001,
                                            weight_decay=0.000,
                                            betas=(0.9, 0.999),
                                            amsgrad=False,
                                            eps=0.00000001)

    @torch.no_grad()
    def validation(self):
        self.model.eval()
        # x, index = next(iter(self.valloader))
        # x = x.to(self.device)
        # index = index.to(self.device)
        # index1, index2 = index.chunk(2)
        # for validation use
        rand_idx = np.random.randint(0, len(self.val_data))
        val_video = torch.Tensor(np.array(self.val_data[rand_idx].frame_list))
        val_video = val_video.permute(0, 3, 1, 2)
        # val_video_loader = video_dataloader(val_video, batch_size=self.args.val_batch_size, mode='val')
        
        self.x_train = val_video[:self.args.val_batch_size].to(self.device)
        self.index_train = torch.arange(self.args.val_batch_size).to(self.device)
        self.x_test = val_video[val_video.shape[0] - self.args.val_batch_size:].to(self.device)
        self.index_test = torch.arange(self.args.val_batch_size).to(self.device)

        # training data
        
        r = self.model.module.backbone(torch.cat((self.x_train, self.x_train), dim=0))
        psi1, psi2 = self.model.module.projector(r).chunk(2, dim=0)
        trueK_train = self.model.module.get_K(self.args.coeff, self.index_train, self.index_train)
        estiK_train = psi1 @ torch.diag(self.model.module.eigenvalues.data) @ psi2.T
        fig, axes = plt.subplots(2, 2)
        im = axes[0, 0].imshow(trueK_train.squeeze(dim=0).detach().cpu())
        im = axes[0, 1].imshow(estiK_train.squeeze(dim=0).detach().cpu())
        # test data
        r = self.model.module.backbone(torch.cat((self.x_test, self.x_test), dim=0))
        psi1, psi2 = self.model.module.projector(r).chunk(2, dim=0)
        trueK_test = self.model.module.get_K(self.args.coeff, self.index_test, self.index_test)
        estiK_test = psi1 @ torch.diag(self.model.module.eigenvalues.data) @ psi2.T
        im = axes[1, 0].imshow(trueK_test.squeeze(dim=0).detach().cpu())
        im = axes[1, 1].imshow(estiK_test.squeeze(dim=0).detach().cpu())

        fig.colorbar(im, ax=axes.ravel().tolist())
        self.writer.add_figure('trueK and estiK', plt.gcf(), self.num_iters)
        
    def save(self):
        state = {
            'backbone': self.model.module.backbone,
            'projector': self.model.module.projector,
            'num_iters': self.num_iters
        }
        torch.save(state, f'./ckpt/{self.num_iters}.pt')
        self.logger.info('Save done.')

    def train(self):
        total_iters = self.args.total_iters
        done = False
        epoch = 1
        scaler = torch.cuda.amp.GradScaler()
            # pbar.update(self.num_iters)
        while not done:
            with tqdm(total=len(self.train_data)) as pbar:
                if epoch == self.args.epochs:
                    done = True
                    break
                print("epoch: ", epoch)
                for i, video in enumerate(self.train_data):
                    
                    #set up a single video loader for each video
                    self.dataloader= video_dataloader(video, self.args.batch_size_per_gpu, mode='train')
                    if torch.cuda.device_count() > 1: self.dataloader.sampler.set_epoch(epoch)
                
                    for x, index in self.dataloader:
                        x = x.permute(0, 3, 1, 2)
                        
                        self.model.train()
                        # self.logger.debug(f'rank: {self.device}, index: {index}')
                        self.optimizer.zero_grad()

                        with torch.cuda.amp.autocast():
                            loss, reg = self.model(x, index)
                            scaler.scale(loss + reg * self.args.alpha).backward()
                        scaler.step(self.optimizer)
                        scaler.update()

                        if is_master():
                            self.writer.add_scalar('loss', loss.item(), self.num_iters)
                            self.writer.add_scalar('reg', reg.item(), self.num_iters)
                            self.writer.add_scalar('total', loss.item() +  self.args.alpha * reg.item(), self.num_iters)


                    # (loss + self.args.alpha * reg).backward()
                    # self.optimizer.step()
                        self.num_iters += 1
                        if self.num_iters % 20 == 0 and is_master():
                            self.validation()
                            # self.save()
                    pbar.update(1)
                epoch += 1

        if is_master(): self.logger.info('Training done.')