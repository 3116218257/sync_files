import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .ibra import ibra
from .walker import walker
from .football_video import VideoDataset

def prepare_dataloader(name, batch_size, num_workers=2, mode='train', **kwargs):
    assert mode in ['train', 'val']
    if name == 'ibra':
        train_data, test_data = ibra('train'), ibra('val')
    elif name == 'walker':
        train_data, test_data = walker('train'), walker('val')
    elif name == 'football_video':
        pass
    elif name == 'football_image':
        pass
    
    if mode == 'train':
        train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              pin_memory=True,
                              shuffle=False if torch.cuda.device_count() > 1 else True,
                              sampler=DistributedSampler(train_data) if torch.cuda.device_count() > 1 else None,
                              drop_last=True,
                              num_workers=num_workers)
        return train_loader
    elif mode == 'val':
        # use a single device when operating on test_loader
        test_loader = DataLoader(test_data,
                              batch_size=batch_size,
                              pin_memory=True,
                              shuffle=True,
                              sampler=None,
                              drop_last=False,
                              num_workers=num_workers)

        return test_loader

def data_rescale(X):
    X = 2 * X - 1.0
    return X

def inverse_data_rescale(X):
    X = (X + 1.0) / 2.0
    return torch.clamp(X, 0.0, 1.0)


def prepare_video_dataset(video_dir='/home/lhy/Projects/sync_files/process_data/processed_data/video1', mode='train'):
    video_dataset = VideoDataset(video_dir, mode)
    return video_dataset


def video_dataloader(video, batch_size, num_workers=2, mode='train'):
    if mode == 'train':
        video_loader = DataLoader(video,
                                batch_size=batch_size,
                                pin_memory=True,
                                shuffle=False if torch.cuda.device_count() > 1 else True,
                                sampler=DistributedSampler(video) if torch.cuda.device_count() > 1 else None,
                                drop_last=True,
                                num_workers=num_workers)
    elif mode == 'val':
        video_loader = DataLoader(video,
                                batch_size=batch_size,
                                pin_memory=True,
                                shuffle=True,
                                sampler=None,
                                drop_last=False,
                                num_workers=num_workers)
    else:
        raise ValueError('mode must be either train or val')
    
    return video_loader