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
        self.folder_list = os.listdir(data_path)
        self.folder_list = [file_name for file_name in self.folder_list if not file_name.startswith('.DS')]
        self.selected_frames = n_frames

        self.file_list = []
        self.total_files = 0
        for folder in self.folder_list:
            folder_path = os.path.join(data_path, folder)
            files = os.listdir(folder_path)
            files = [os.path.join(folder_path, file) for file in files]
            self.file_list += files
            self.total_files += len(files)


    def __len__(self):
        return self.total_files

    def read_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame / 127.5 - 1)

        frames_tensor = torch.FloatTensor(np.array(frames))
        cap.release()

        return frames_tensor

    def __getitem__(self, idx):
        file_name = self.file_list[idx]

        # folder_path = os.path.join(self.data_path, folder_name)

        # video_files = os.listdir(folder_path)
        # print(video_files)

        # video_path = os.path.join(folder_path, video_files)

        video = self.read_video(file_name)
        # print(video.shape)

        video1, video2, ind1, ind2 = video_sampler(video, self.selected_frames)
        # video1, video2, ind1, ind2 = video_sampler(video, video.shape[0])

        return video1.permute(0, 3, 1, 2), video2.permute(0, 3, 1, 2), ind1, ind2


if __name__ == '__main__':
    dataset = video_dataset(data_path='./data/test_one_video')

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    pbar = tqdm(data_loader)
    for video, _ in data_loader:
        for id in range(1):
            print(video.shape)
            # save_selected_frames(video[0][id], './frames/', id)
        pbar.update(1)

    print('\ndataset check pass!')
