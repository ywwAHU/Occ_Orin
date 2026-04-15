from __future__ import annotations

import torch
from torch import nn


class OccupancyHead(nn.Module):
    def __init__(self, bev_channels: int, num_classes: int, z_bins: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.z_bins = z_bins
        self.head = nn.Conv2d(bev_channels, num_classes * z_bins, kernel_size=1)

    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        logits = self.head(bev_features)
        batch_size, _, bev_height, bev_width = logits.shape
        logits = logits.view(batch_size, self.num_classes, self.z_bins, bev_height, bev_width)
        return logits
