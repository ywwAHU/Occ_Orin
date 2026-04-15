from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvNormAct


class CameraEncoder(nn.Module):
    def __init__(self, base_channels: int, out_channels: int) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 3
        self.net = nn.Sequential(
            ConvNormAct(3, c1, kernel_size=5, stride=2),
            ConvNormAct(c1, c2, stride=2),
            ConvNormAct(c2, c3, stride=2),
            ConvNormAct(c3, out_channels, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, camera_count, channels, height, width = images.shape
        images = images.view(batch_size * camera_count, channels, height, width)
        features = self.net(images)
        _, feat_channels, feat_height, feat_width = features.shape
        return features.view(batch_size, camera_count, feat_channels, feat_height, feat_width)
