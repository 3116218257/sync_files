import os
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import logging
from utils.dist import is_master
from utils.misc import create_exp_name
from tensorboard import program
from model import NeuralEFCLR
from dataloader import prepare_video_dataset, video_dataloader
from tqdm import tqdm
import torch.multiprocessing as mp
import argparse
import random


def create_experiment(args):
    log_dir = ckpt_dir = sample_dir = None
    if is_master():# only one gpu master creates directories
        name = create_exp_name(args.exp_name)
        exp_dir = os.path.join('experiments', name)
        log_dir = os.path.join('./', exp_dir, 'logs')
        ckpt_dir = os.path.join(exp_dir, 'checkpoints')
        sample_dir = os.path.join(exp_dir, 'samples')
        os.makedirs(exp_dir)
        os.makedirs(log_dir)
        os.makedirs(ckpt_dir)
        os.makedirs(sample_dir)
    return log_dir, ckpt_dir, sample_dir

def get_logger(level):
    handler1 = logging.StreamHandler()
    # handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    # handler2.setFormatter(formatter)
    my_logger = logging.getLogger('training_logger')
    my_logger.addHandler(handler1)
    # logger.addHandler(handler2)
    my_logger.setLevel(level)
    return my_logger

def setup_tensorboard(args, log_dir):
        tb = program.TensorBoard()
        tb.configure(argv=[None, f'--logdir={log_dir}', f'--port={args.tb_port}', f'--load_fast=false'])
        url = tb.launch()

def setup(global_rank, world_size):
    # 配置Master Node的信息
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8900'
    print(global_rank, world_size)
    # 初始化Process Group
    # 关于init_method, 参数详见https://pytorch.org/docs/stable/distributed.html#initialization
    #dist.init_process_group("nccl", init_method='env://', rank=global_rank, world_size=world_size, timeout=timedelta(seconds=10))
    dist.init_process_group("nccl", init_method='env://', rank=global_rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
  
def train(local_rank, opt):
    
    global_rank = local_rank + opt.node_rank * opt.nproc_per_node
    world_size = opt.nnode * opt.nproc_per_node
    print("local_rank: ", local_rank, "setting up distributed env")
    setup(global_rank=global_rank, world_size=world_size)
    print("local_rank: ", local_rank, "distributed env set")
    
    device = opt.gpu_ids[local_rank]
    torch.manual_seed(opt.seed)
    
    use_amp = opt.use_amp
    logger = get_logger(level = getattr(logging, opt.verbose.upper(), None))
    
    if is_master():
        log_dir, ckpt_dir, sample_dir = create_experiment(opt)
        writer = SummaryWriter(log_dir)
        setup_tensorboard(opt, log_dir)
        logger.info(f'Number of devices: {torch.cuda.device_count()}')
        
    '''setup model and optimizer'''
    print("local_rank: ", local_rank, "loading model")
    model = NeuralEFCLR(opt, logger, device).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    
    optimizer = torch.optim.Adam(model.parameters(),
                                    lr=0.0001,
                                    weight_decay=0.000,
                                    betas=(0.9, 0.999),
                                    amsgrad=False,
                                    eps=0.00000001)
    
    train_data = prepare_video_dataset(video_dir=opt.root_path, mode='train')
    val_data = prepare_video_dataset(video_dir=opt.root_path, mode='val')

    epoch = 1
    scaler = torch.cuda.amp.GradScaler()
    num_iters = 0
    while True:
        with tqdm(total=len(train_data)) as pbar:
            if epoch == opt.epochs:
                break
            print("epoch: ", epoch)
            dist.barrier()
            for i, video in enumerate(train_data):
                    #set up a single video loader for each video
                dataloader= video_dataloader(video, opt.batch_size_per_gpu, mode='train')
                if torch.cuda.device_count() > 1: dataloader.sampler.set_epoch(epoch)
            
                for x, index in dataloader:
                    # print(index)
                    x = x.permute(0, 3, 1, 2).to(device)
                    model.train()

                    optimizer.zero_grad()

                    with torch.cuda.amp.autocast():
                        loss, reg = model(x, index)
                        scaler.scale(loss + reg * opt.alpha).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    if is_master():
                        writer.add_scalar('loss', loss.item(), num_iters)
                        writer.add_scalar('reg', reg.item(), num_iters)
                        writer.add_scalar('total', loss.item() + opt.alpha * reg.item(), num_iters)


                # (loss + self.args.alpha * reg).backward()
                # self.optimizer.step()
                    num_iters += 1
                    if num_iters % 20 == 0 and is_master():
                        with torch.no_grad():
                       # validation(model, val_data, opt, device, writer, num_iters)
                            model.eval()
                            
                            rand_idx = np.random.randint(0, len(val_data))
                            val_video = torch.Tensor(np.array(val_data[rand_idx].frame_list))
                            val_video = val_video.permute(0, 3, 1, 2)

                            x_train = val_video[:opt.val_batch_size].to(device)
                            index_train = torch.arange(opt.val_batch_size).to(device)
                            x_test = val_video[val_video.shape[0] - opt.val_batch_size:].to(device)
                            index_test = torch.arange(opt.val_batch_size).to(device)

                            # training data
                            
                            r = model.module.backbone(torch.cat((x_train, x_train), dim=0))
                            psi1, psi2 = model.module.projector(r).chunk(2, dim=0)
                            trueK_train = model.module.get_K(opt.coeff, index_train, index_train)
                            estiK_train = psi1 @ torch.diag(model.module.eigenvalues.data) @ psi2.T
                            fig, axes = plt.subplots(2, 2)
                            im = axes[0, 0].imshow(trueK_train.squeeze(dim=0).detach().cpu())
                            im = axes[0, 1].imshow(estiK_train.squeeze(dim=0).detach().cpu())
                            # test data
                            r = model.module.backbone(torch.cat((x_test, x_test), dim=0))
                            psi1, psi2 = model.module.projector(r).chunk(2, dim=0)
                            trueK_test = model.module.get_K(opt.coeff, index_test, index_test)
                            estiK_test = psi1 @ torch.diag(model.module.eigenvalues.data) @ psi2.T
                            im = axes[1, 0].imshow(trueK_test.squeeze(dim=0).detach().cpu())
                            im = axes[1, 1].imshow(estiK_test.squeeze(dim=0).detach().cpu())

                            fig.colorbar(im, ax=axes.ravel().tolist())
                            writer.add_figure('trueK and estiK', plt.gcf(), num_iters)
                            # plt.close(fig)
                            # self.save()
                pbar.update(1)
            epoch += 1

    if is_master(): logger.info('Training done.')


def validation(model, val_data, opt, device, writer, num_iters):
    pass

def train1(local_rank, args):
    print("test train1")

def init_seeds(RANDOM_SEED=1337, no=0):
    RANDOM_SEED += no
    print("local_rank = {}, seed = {}".format(no, RANDOM_SEED))
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1233, type=int)
    parser.add_argument('--nproc_per_node', default=1, type=int)
    parser.add_argument('--nnode', default=1 ,type=int)
    parser.add_argument('--node_rank', default=0, type=int)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--use_amp", action='store_true', default=False)
    
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--no_tb', default=False, action='store_true')
    parser.add_argument('--tb_port', type=int, default=9990)
    parser.add_argument('--verbose', type=str, default='info')
    parser.add_argument('--total_iters', type=int, default=10000)
    parser.add_argument('--coeff', type=float, default=1.)
    parser.add_argument('--k', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument("--gpu_ids", default=[2], type=int)
    # model settings
    parser.add_argument('--batch_size_per_gpu', type=int, default=100)
    # direct copy
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=4096, type=int, metavar='N',
                        help='mini-batch size')
    
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--proj_dim', default=[2048, 128], type=int, nargs='+')
    parser.add_argument('--no_proj_bn', default=False, action='store_true')
    parser.add_argument('--momentum', default=0.99, type=float)
    parser.add_argument('--root_path', default='/home/lhy/Projects/sync_files/process_data/processed_data/video1', type=str)


    args = parser.parse_args()
    
    
    print(args)
    args.local_rank= int(os.environ["LOCAL_RANK"])
    
    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)
    
    mp.spawn(train, args=(args,), nprocs=args.nproc_per_node)
       
       
