from networks import VPPrecond, SongUNet
import torch
import torch.nn as nn
import numpy as np
from torch_utils import persistence
from torch.nn.functional import silu
import pickle

class SongUNetWithMLP(nn.Module):
    def __init__(
        self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        mlp_hidden_dims     = 512,
        mlp_num_layers      = 5,
        mlp_activation      = nn.ReLU()
        ):
        super().__init__()
        self.song_unet = SongUNet(img_resolution=32,
                                    in_channels=3, 
                                    out_channels=3, 
                                    label_dim=10, 
                                    augment_dim=0, 
                                    model_channels=128,
                                    channel_mult=[1, 2, 2, 2], 
                                    channel_mult_emb=4, 
                                    num_blocks=4, 
                                    attn_resolutions=[16], 
                                    dropout=0.10,
                                    label_dropout=0, 
                                    embedding_type='positional', 
                                    channel_mult_noise=1, 
                                    encoder_type='standard', 
                                    decoder_type='standard',
                                    resample_filter=[1, 1])

        # with open('edm-cifar10-32x32-cond-vp.pkl', 'rb') as file:
        #     self.song_unet = pickle.load(file)
        # for param in self.song_unet.parameters():
        #     param.requires_grad = False
        
        self.mlp = self.build_mlp(input_dim=out_channels * 32 ** 2,
                                  hidden_dim=mlp_hidden_dims,
                                  output_dim=256,
                                  num_layers=mlp_num_layers,
                                  activation=mlp_activation)

    def build_mlp(self, input_dim, hidden_dim, output_dim, num_layers, activation):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation)
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(activation)
        layers.append(nn.BatchNorm1d(output_dim))
        return nn.Sequential(*layers)

    def forward(self, x, n_label, cls_label=None):
        unet_output = self.song_unet(x, n_label, cls_label)
        unet_output = torch.flatten(unet_output, start_dim=1)
        mlp_output = self.mlp(unet_output)
        return mlp_output

#these two functions just for test the output size
def cal_y_n(images, labels=None, augment_pipe=None):
    rnd_normal = torch.randn([images.shape[0], 1, 1, 1])
    sigma = rnd_normal
    weight = (sigma ** 2) / (sigma + 1) ** 2
    y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
    n = torch.randn_like(y) * sigma
    return y, n

def cal_c(x, sigma, class_labels=None, force_fp32=False):
    x = x.to(torch.float32)
    sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
    c_in = 1 / (sigma ** 2 + 1).sqrt()
    c_noise = sigma.log() / 4
    return c_in, c_noise


if __name__ == '__main__':
    batch_size = 8
    model = SongUNetWithMLP(
        img_resolution      = 32,
        in_channels         = 3,                        # Number of color channels at input.
        out_channels        = 3,                       # Number of color channels at output.
        label_dim           = 10,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        mlp_hidden_dims     = 512,
        mlp_num_layers      = 5,
        mlp_activation      = nn.ReLU()
    )

    x1 = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    # y, n = cal_y_n(x1)
    c_in, c_noise = cal_c(x1, t)
    label_dim = 10
    y = model(x1, c_noise.flatten(), torch.zeros([1, label_dim]))
    print(y.shape)
    #z=torch.flatten(y, start_dim=1)
    #print(z.shape)
