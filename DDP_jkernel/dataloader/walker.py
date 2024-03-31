import torch


walker_path = '/home/jiachun/codebase/jkernel/data/walker_tensor' # (1332, 3, 480, 270)

class walker():
    def __init__(self, mode='train') -> None:
        if mode == 'train':
            self.data = torch.load(walker_path)[:1000]
        elif mode == 'val':
            self.data = torch.load(walker_path)[1000:]
        else:
            raise KeyError

    def __getitem__(self, index):
        return self.data[index], index
    
    def __len__(self):
        return len(self.data)
