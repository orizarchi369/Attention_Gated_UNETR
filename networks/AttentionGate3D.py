# AttentionGate3D.py
# New version, for "improved attention gates"

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate3D(nn.Module):
    def __init__(self, gating_channels, skip_channels, out_channels):
        super().__init__()
        self.gating_conv = nn.Conv3d(gating_channels, out_channels, kernel_size=1)
        self.skip_conv = nn.Conv3d(skip_channels, out_channels, kernel_size=1, stride=2)  # Downsample skip
        self.relu = nn.ReLU(inplace=True)
        self.attention_conv = nn.Conv3d(out_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable gamma for residual gating, starts at 0

        # Initialize sigmoid bias to start mask near 1.0
        nn.init.constant_(self.attention_conv.bias, 2.0)  # Positive bias for initial mask ~ sigmoid(2) ≈ 0.88 (close to 1)

    def forward(self, gating, skip):
        # Downsample skip to match gating spatial size
        s = self.skip_conv(skip)
        g = self.gating_conv(gating)
        # Add and apply ReLU
        attention = self.relu(g + s)
        # Compute attention map (1 channel)
        attention = self.attention_conv(attention)
        attention = self.sigmoid(attention)
        # Upsample attention map to match original skip size
        attention = F.interpolate(attention, size=skip.shape[2:], mode='trilinear', align_corners=False)
        # Residual gating: skip * (1 + gamma * mask)
        out = skip * (1 + self.gamma * attention)
        return out