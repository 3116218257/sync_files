import argparse
import os
import logging

import torch
from tensorboard import program
from torch.distributed import init_process_group, destroy_process_group

# from dataloader import prepare_dataloader
# from utils.dist import gather_from_all
from trainer import Trainer_Ibra, Trainer_football
from utils.dist import is_master
from utils.misc import create_exp_name

def parse_args():
    parser = argparse.ArgumentParser()
    # experimental routines
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--no_tb', default=False, action='store_true')
    parser.add_argument('--tb_port', type=int, default=9990)
    parser.add_argument('--verbose', type=str, default='info')
    parser.add_argument('--total_iters', type=int, default=10000)
    parser.add_argument('--coeff', type=float, default=1.)
    parser.add_argument('--k', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument("--gpu_ids", default=[6], type=int)
    # model settings
    parser.add_argument('--batch_size_per_gpu', type=int, default=100)
    # direct copy
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=4096, type=int, metavar='N',
                        help='mini-batch size')

    # parser.add_argument('--dim', default=360, type=int,
    #                     help="dimension of subvector sent to infoNCE")
    # parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--proj_dim', default=[2048, 128], type=int, nargs='+')
    parser.add_argument('--no_proj_bn', default=False, action='store_true')
    # parser.add_argument('--t', default=10, type=float)
    # for ablation
    # parser.add_argument('--no_stop_grad', default=False, action='store_true')
    # parser.add_argument('--l2_normalize', default=False, action='store_true')
    # parser.add_argument('--not_all_together', default=False, action='store_true')
    # parser.add_argument('--positive_def', default=False, action='store_true')
    # parser.add_argument('--corrector', default=0, type=float)
    parser.add_argument('--momentum', default=0.99, type=float)
    parser.add_argument('--data_root', default='/home/lhy/Projects/sync_files/process_data/processed_data/video1', type=str)
    # parser.add_argument('--adaptive_weighting', default=False, action='store_true')

    args = parser.parse_args()
    return args

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

def ddp_setup():
    # initialize the process group
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # torch.cuda.set_device(6)
    # torch.cuda.set_device(args.gpu_ids)
    # print(int(os.environ["LOCAL_RANK"]))
    # assert 0


if __name__ == '__main__':
    args = parse_args()
    # setup directory
    log_dir, ckpt_dir, sample_dir = create_experiment(args)
    if not args.no_tb and is_master(): setup_tensorboard(args, log_dir)
    # setup ddp
    ddp_setup()
    # setup logger
    logger = get_logger(level = getattr(logging, args.verbose.upper(), None))
    if is_master(): logger.info(f'Number of devices: {torch.cuda.device_count()}')

    # setup trainer
    # trainer = Trainer_Ibra(args, logger, log_dir, ckpt_dir, sample_dir)
    trainer = Trainer_football(args, logger, log_dir, ckpt_dir, sample_dir)

    if is_master(): logger.info('Start to train.')
    # start training
    trainer.train()
    # clean up
    destroy_process_group()


