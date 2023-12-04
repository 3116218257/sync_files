import torch.nn as nn
from torch.nn import init
import torch
import math
import torchvision

# class Swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)


# class TimeEmbedding(nn.Module):
#     def __init__(self, T, d_model, dim):
#         assert d_model % 2 == 0
#         super().__init__()
#         emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
#         emb = torch.exp(-emb)
#         pos = torch.arange(T).float()
#         emb = pos[:, None] * emb[None, :]
#         assert list(emb.shape) == [T, d_model // 2]
#         emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
#         assert list(emb.shape) == [T, d_model // 2, 2]
#         emb = emb.view(T, d_model)

#         self.timembedding = nn.Sequential(
#             nn.Embedding.from_pretrained(emb),
#             nn.Linear(d_model, dim),
#             Swish(),
#             nn.Linear(dim, dim),
#         )
#         self.initialize()

#     def initialize(self):
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 init.xavier_uniform_(module.weight)
#                 init.zeros_(module.bias)

#     def forward(self, t):
#         emb = self.timembedding(t)
#         return emb
    

# class ResBlock(nn.Module):
#     def __init__(self, in_channels, mid_channels, out_channels, stride=1):
#         super(ResBlock, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1)
#         self.T_EMD = TimeEmbedding(T=1500, d_model=128, dim=128 * 4)
#         self.temb_proj = nn.Sequential(
#             Swish(),
#             nn.Linear(128 * 4, mid_channels),
#         )
#         self.bn1 = nn.BatchNorm2d(mid_channels)

#         self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1)
#         self.bn2 = nn.BatchNorm2d(mid_channels)

#         self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1)
#         self.bn3 = nn.BatchNorm2d(out_channels)

#         self.relu = nn.ReLU(inplace=True)
#         if stride != 1 or in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(out_channels)
#             )
#         else:
#             self.downsample = nn.Identity()

#     def forward(self, x, t):
#         identity = x
#         out = self.conv1(x)

#         # temb = self.T_EMD(t)
#         # out += self.temb_proj(temb)[:, :, None, None]

#         out = self.bn1(out)
#         out = self.relu(out)
        
#         out = self.bn2(self.conv2(out))
#         out = self.relu(out)

#         out = self.bn3(self.conv3(out))

#         # if self.downsample is not nn.Identity():
#         #     identity = self.downsample(x)
#         identity = self.downsample(identity)

#         out += identity
#         out = self.relu(out)
        
#         return out

# class ResNet50(nn.Module):
#     def __init__(self, num_classes):
#         super(ResNet50, self).__init__()

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)

#         self.layer1 = self._make_layer(64, 64, 256, blocks=3, stride=1)
#         self.layer2 = self._make_layer(256, 128, 512, blocks=4, stride=2)
#         self.layer3 = self._make_layer(512, 256, 1024, blocks=6, stride=2)
#         self.layer4 = self._make_layer(1024, 512, 2048, blocks=3, stride=2)

#         #self.T_EMD = TimeEmbedding(T=2000, d_model=256, dim=256 * 4)
#         # self.temb_proj = nn.Sequential(
#         #     Swish(),
#         #     nn.Linear(256 * 4, 2048),
#         # )

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(2048, num_classes)

#     def _make_layer(self, in_channels, mid_channels, out_channels, blocks, stride):
#         layers = nn.ModuleList()
#         layers.append(ResBlock(in_channels, mid_channels, out_channels, stride))
#         for _ in range(1, blocks):
#             layers.append(ResBlock(out_channels, mid_channels, out_channels))
#         return layers

#     def forward(self, x, t):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.maxpool(x)
#         for layers in self.layer1:
#             x = layers(x, t)
#         for layers in self.layer2:
#             x = layers(x, t)
#         for layers in self.layer3:
#             x = layers(x, t)
#         for layers in self.layer4:
#             x = layers(x, t)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)

#         x = self.fc(x)

#         return x


    
# if __name__ == '__main__':
#     # mod = open('resnet_arch.txt', 'w')
#     # model = torchvision.models.resnet50()
#     # print(model, file=mod)
#     x = torch.randn((256, 3, 32, 32))
#     t = torch.randint(500, 501, (256,))
#     print(t)
#     t_emb = TimeEmbedding(2000, 256, 256 * 4)
#     print(t_emb)
#     model = ResNet50(num_classes=10)

#     tembbb = t_emb(t)
#     print(tembbb)
#     print(tembbb.shape)
#     output = model(x, t)
#     print(output.shape)


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
    

class Bottleneck(nn.Module):

    expansion = 4 
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.T_EMD = TimeEmbedding(T=1000, d_model=128, dim=128 * 4)
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(128 * 4, out_channel),
        )
        
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)


    def forward(self, x, t):
        identity = x 
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)

        temb = self.T_EMD(t)
        out += self.temb_proj(temb)[:, :, None, None]

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity 
        out = self.relu(out)

        return out
    

class ResNet(nn.Module):

    def __init__(self, block, block_num, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channel = 64  

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        self.layer1 = self._make_layer(block=block, channel=64, block_num=block_num[0], stride=1)
        self.layer2 = self._make_layer(block=block, channel=128, block_num=block_num[1], stride=2)
        self.layer3 = self._make_layer(block=block, channel=256, block_num=block_num[2], stride=2)
        self.layer4 = self._make_layer(block=block, channel=512, block_num=block_num[3], stride=2) 

        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) 
        self.fc = nn.Linear(in_features=512*block.expansion, out_features=num_classes)

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel*block.expansion: 
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel*block.expansion, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(num_features=channel*block.expansion))
            
        layers = []  
        layers.append(block(in_channel=self.in_channel, out_channel=channel, downsample=downsample, stride=stride)) 
        self.in_channel = channel*block.expansion

        for _ in range(1, block_num): 
            layers.append(block(in_channel=self.in_channel, out_channel=channel))

        return nn.Sequential(*layers) 
    
    def forward(self, x, t):
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layers in self.layer1:
            x = layers(x, t)
        for layers in self.layer2:
            x = layers(x, t)
        for layers in self.layer3:
            x = layers(x, t)
        for layers in self.layer4:
            x = layers(x, t)
        # x = self.layer1(x, t)
        # x = self.layer2(x, t)
        # x = self.layer3(x, t)
        # x = self.layer4(x, t)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet50(num_classes=10):
    return ResNet(block=Bottleneck, block_num=[3, 4, 6, 3], num_classes=num_classes)


if __name__ == '__main__':

    input = torch.randn(256, 3, 32, 32)
    ResNet50 = resnet50(10)
    output = ResNet50.forward(input)
    print(ResNet50)
    print(output.shape)