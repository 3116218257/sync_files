import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
from tqdm import tqdm
import numpy as np
# from torchvision.io import read_video

#This is a single video class, batch size = frames
class one_video():
    def __init__(self, video_path) -> None:
        self.video_path = video_path
        video_capture = cv2.VideoCapture(video_path)
        self.frame_list = []

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            frame = cv2.resize(frame, (384, 216))
            self.frame_list.append(frame)
        video_capture.release()
        
        
        # self.frame_list, _, _ = read_video(video_path, output_format='TCHW')
        # print(self.frame_list.shape)

        # if self.mode == 'train':
        #     self.frame_list = self.frame_list[:len(self.frame_list)-200]
        # elif self.mode == 'val':
        #     self.frame_list = self.frame_list[len(self.frame_list)-200:]
        
        # self.frame_list = torch.Tensor(np.array(self.frame_list))
            
        
    def __getitem__(self, index):
        return torch.Tensor(np.array(self.frame_list[index])), index
    
    def __len__(self):
        return len(self.frame_list)

#This is a video set class, just for loading multiple videos,
# each time it will load one_video class
class VideoDataset(Dataset):
    def __init__(self, video_dir, mode='train'):
        self.video_dir = video_dir
        self.video_files = os.listdir(video_dir)
        self.mode = mode
        
        if self.mode == 'train':
            self.video_files = self.video_files[:len(self.video_files)-10]
        elif self.mode == 'val':
            self.video_files = self.video_files[len(self.video_files)-10:]
        else:
            raise ValueError('mode should be train or val')
        

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        rt_data = one_video(video_path)
        return rt_data
            

if __name__ == '__main__':
    video_dir = '/home/lhy/Projects/sync_files/process_data/processed_data/video1'
    video_dataset = VideoDataset(video_dir)
    # print(video_dataset[0].frame_list.shape)
    
    with tqdm(total=len(video_dataset)) as pbar:
        for i, video in enumerate(video_dataset):
            pbar.update(1)
            one_video_loader = DataLoader(video, batch_size=1, shuffle=False)
            for video_tensor, id in one_video_loader:
                print(video_tensor.permute(0, 3, 1, 2).shape)
                # pass


    # for video_tensor, id in video_loader:
    #     print(video_tensor.shape)