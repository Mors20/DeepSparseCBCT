
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist



def silu(x):
    return x*torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)



class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_groups=8):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)  # GroupNorm 替代 BatchNorm
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)  # GroupNorm 替代 BatchNorm
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += identity  # 残差连接
        out = self.relu(out)
        return out

class CodeClassifier(nn.Module):
    def __init__(self, latent_size, dim_embd, num_layers, codebook_size, num_groups=8):
        super().__init__()
        self.latent_size = latent_size
        self.dim_embd = dim_embd
        

        self.conv_in = nn.Conv3d(latent_size, dim_embd, kernel_size=3, stride=1, padding=1)


        self.cnn_layers = nn.Sequential(
            *[BasicBlock3D(dim_embd, dim_embd, num_groups=num_groups) for _ in range(num_layers)]
        )


        self.classifier = nn.Conv3d(dim_embd, codebook_size, kernel_size=1)

    def forward(self, x):
        # x: [B, C, 32, 32, 32]
        B, C, D, H, W = x.shape
        
        # 初始特征提取
        x = self.conv_in(x)  # [B, 256, 32, 32, 32]
        
        # 通过 CNN 提取特征
        x = self.cnn_layers(x)  # [B, 256, 32, 32, 32]

        # 输出分类（按体素分类）
        logits = self.classifier(x)  # [B, 1024, 32, 32, 32]

        return logits

