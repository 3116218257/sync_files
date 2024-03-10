import os
import cv2
import torch
import numpy as np
from video_sampler import video_sampler, save_selected_frames
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm

class video_dataset(Dataset):
    def __init__(self, data_path='./data', n_frames=10):
        self.data_path = data_path
        self.file_list = os.listdir(data_path)
        self.file_list = [file_name for file_name in self.file_list if not file_name.startswith('.DS')]
        self.selected_frames = n_frames


    def __len__(self):
        return len(self.file_list)

    def read_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame / 127.5 - 1.0)

        frames_np = np.array(frames)
        frames_tensor = torch.from_numpy(frames_np)
        cap.release()

        return frames_tensor

    def __getitem__(self, idx):
        folder_name = self.file_list[idx]

        folder_path = os.path.join(self.data_path, folder_name)

        video_files = os.listdir(folder_path)

        video_path = os.path.join(folder_path, video_files[0])

        video = self.read_video(video_path)

        video1, video2 = video_sampler(video, self.selected_frames)

        return video1.permute(0, 3, 1, 2), video2.permute(0, 3, 1, 2)


if __name__ == '__main__':
    dataset = video_dataset(data_path='./data/UCF50/')

    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    pbar = tqdm(data_loader)
    for video, _ in data_loader:
        for id in range(10):
            save_selected_frames(video[0][id], './frames/', id)
        pbar.update(1)

    print('\ndataset check pass!')
