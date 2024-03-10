import torch
from PIL import Image
import os

def video_sampler(data, n):
    num_frames, c, h, w = data.shape
    indices = torch.randperm(num_frames)[:n]
    indices_prim = torch.randperm(num_frames)[:n]

    indices, _ = torch.sort(indices)
    indices_prim, _ = torch.sort(indices_prim)

    selected_frames = data[indices, :, :, :]
    selected_frames_prim = data[indices_prim, :, :, :]
    
    return selected_frames.to(torch.float), selected_frames_prim.to(torch.float)


def save_selected_frames(frame, save_path, index):
    # print(frame.shape)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    image = Image.fromarray(frame.numpy())
    image.save(save_path + f"{index}.png")


if __name__ == '__main__':
    # video = torch.randn((512, 10, 3, 128, 128))
    # # print(video_sampler(video, 5).shape)
    # x = read_video()
    #
    # sampled = video_sampler(x, 5)
    # for id in range(5):
    #     save_selected_frames(sampled[id], './frames/', id)
    print('test finish')
