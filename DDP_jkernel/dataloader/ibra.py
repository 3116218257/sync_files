import torch


ibra_path = '../data/ibra_tensor' # (3014, 3, 180, 320)

class ibra():
    def __init__(self, mode='train') -> None:
        if mode == 'train':
            self.data = torch.load(ibra_path)[:2000]# first 2000 frames for training
        elif mode == 'val':
            self.data = torch.load(ibra_path)[2000:]# last 1014 frames for validation
        else:
            raise KeyError

    def __getitem__(self, index):
        return self.data[index], index
    
    def __len__(self):
        return len(self.data)
