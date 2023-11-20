import torch
import torchvision.models as models
import torch.nn as nn
from einops import rearrange
import math

class decoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.t_embedding = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU()
        )
    
    def forward(self, t):
        return self.t_embedding(t)
    

class LearnedSinusoidalPosEmb(nn.Module): 
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        # fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
    
class time_embedded_resnet(nn.Module):
    def __init__(self, t_dim=32):
        super().__init__()
        self.resnet = models.resnet50(zero_init_residual=True)
        self.resnet.fc = nn.Linear(2048, 256)##fit decoder
        self.fc = nn.Linear(2048, 2048)
        self.encoder = LearnedSinusoidalPosEmb(dim=t_dim)
        self.decoder = decoder(embedding_dim=256)
        
    def forward(self, x, t):
        out = self.encoder(t)
        out = out + x
        out = self.resnet(out)
        out = self.decoder(out)
        out = self.fc(out)
        return out
    
if __name__ == '__main__':
    t = torch.zeros(32)
    t[4] = 1
    x = torch.randn((256, 3, 32, 32))
    model = time_embedded_resnet(t_dim=32)
    model.fc = nn.Identity()
    out = model.forward(x, t)
    print(out.shape)








