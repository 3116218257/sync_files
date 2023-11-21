# import torch
# import torchvision.models as models
# import torch.nn as nn
# from einops import rearrange
# import math

    

# class LearnedSinusoidalPosEmb(nn.Module): 
#     def __init__(self, dim):
#         super().__init__()
#         assert (dim % 2) == 0
#         half_dim = dim // 2
#         self.weights = nn.Parameter(torch.randn(half_dim))

#     def forward(self, x):
#         x = rearrange(x, 'b -> b 1')
#         freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
#         fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
#         #fouriered = torch.cat((x, fouriered), dim = -1)
#         return fouriered
    
# class time_embedded_resnet(nn.Module):
#     def __init__(self, t_dim=32):
#         super().__init__()
#         self.resnet = models.resnet50(zero_init_residual=True)
#         resnet = models.resnet50(pretrained=True)
#         conv1 = list(resnet.children())[0:2]
#         self.conv1 = nn.Sequential(*conv1)
        
#         after_layers = list(resnet.children())[3:9]
#         self.after_conv1 = nn.Sequential(*after_layers)
        
#         # self.resnet.fc = nn.Linear(2048, t_dim * (t_dim + 1))
        
#         # self.recover_size = nn.Linear(t_dim * (t_dim + 1), 2048)
        
#         self.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
        
#         self.encoder = LearnedSinusoidalPosEmb(dim=t_dim)
        
#     def forward(self, x, t):
#         out = self.conv1(x)
#         t_embedding = self.encoder(t)
#         out = out + t_embedding
#         out = self.after_conv1(out)
#         out = self.fc(out)
#         out =  out.squeeze(3).squeeze(2)
        
#         # out = self.resnet(x)
#         # t_embedding = self.encoder(t)
#         # out = out + t_embedding
#         # out = self.recover_size(out)
#         # out = self.fc(out)
        
#         return out




    
# if __name__ == '__main__':
#     t = torch.zeros(16)
#     t[4] = 1
#     x = torch.randn((256, 3, 32, 32))
#     model = time_embedded_resnet(t_dim=16)
#     # model = models.resnet50(pretrained=False)
#     # print(model)
#     model.fc = nn.Identity()
#     out = model.forward(x, t)
#     print(out.shape)
#     # embedding_dim = 64  # Adjust the dimensionality as needed
#     # model = ResNetWithPositionalEmbedding(embedding_dim)
    
    
#     # input_tensor = torch.randn(1, 3, 256, 256)  # Replace with your input tensor shape
#     # position_indices = torch.tensor([0])  # Replace [0] with your desired position indices

#     # output = model(input_tensor, position_indices)
#     # print(model)
    
#     # print(output.shape)





import torch.nn as nn
from torch.nn import init
import torch
import math


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1)
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(128 * 4, mid_channels),
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x, temb):
        identity = x
        out = self.conv1(x)

        out += self.temb_proj(temb)[:, :, None, None]

        out = self.bn1(out)

        out = self.bn2(self.conv2(out))

        out = self.bn3(self.conv3(out))

        identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)

        self.layer1 = self._make_layer(64, 64, 256, blocks=3, stride=1)
        self.layer2 = self._make_layer(256, 128, 512, blocks=4, stride=2)
        self.layer3 = self._make_layer(512, 256, 1024, blocks=6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 2048, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, in_channels, mid_channels, out_channels, blocks, stride):
        layers = nn.ModuleList()
        layers.append(ResBlock(in_channels, mid_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, mid_channels, out_channels))
        return layers

    def forward(self, x, temb):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        for layers in self.layer1:
            x = layers(x, temb)
        for layers in self.layer2:
            x = layers(x, temb)
        for layers in self.layer3:
            x = layers(x, temb)
        for layers in self.layer4:
            x = layers(x, temb)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    x = torch.randn((256, 3, 32, 32))
    t = torch.randint(1000, (256, ))
    t_emb = TimeEmbedding(1000, 128, 128 * 4)
    model = ResNet50(num_classes=10)
    print(model)

    tembbb = t_emb(t)
    output = model(x, tembbb)
    print(output.shape)