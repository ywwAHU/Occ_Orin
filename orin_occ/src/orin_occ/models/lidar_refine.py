from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvNormAct


class LidarRefiner(nn.Module):
    def __init__(self, lidar_channels: int, bev_channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvNormAct(lidar_channels, bev_channels, kernel_size=3),
            ConvNormAct(bev_channels, bev_channels, kernel_size=3),
        )

    def forward(self, bev_features: torch.Tensor, lidar_bev: torch.Tensor | None) -> torch.Tensor:
        if lidar_bev is None:
            return bev_features
        return bev_features + self.encoder(lidar_bev)
