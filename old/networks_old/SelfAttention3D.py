# SelfAttention3D.py
# This is the Self-Attention 3D module that my project partner created. Ori

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, D, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, D*H*W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, D*H*W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(B, -1, D*H*W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, D, H, W)

        out = self.gamma * out + x
        return out
