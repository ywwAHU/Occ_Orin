from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvNormAct, DepthwiseResidualBlock


class BevEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.stem = ConvNormAct(in_channels, out_channels, kernel_size=3)
        self.blocks = nn.Sequential(
            DepthwiseResidualBlock(out_channels),
            DepthwiseResidualBlock(out_channels),
            DepthwiseResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.blocks(x)
