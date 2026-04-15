from __future__ import annotations

import torch
from torch import nn

from orin_occ.config import ProjectConfig

from .bev_encoder import BevEncoder
from .camera_encoder import CameraEncoder
from .lidar_refine import LidarRefiner
from .occ_head import OccupancyHead
from .projector import GeometryAwareProjector


class OrinOccNet(nn.Module):
    def __init__(self, config: ProjectConfig) -> None:
        super().__init__()
        model_cfg = config.model
        self.use_lidar = model_cfg.use_lidar
        self.camera_encoder = CameraEncoder(
            base_channels=model_cfg.backbone_base_channels,
            out_channels=model_cfg.feature_channels,
        )
        self.projector = GeometryAwareProjector(config.grid, config.image)
        self.bev_encoder = BevEncoder(
            in_channels=model_cfg.feature_channels,
            out_channels=model_cfg.bev_channels,
        )
        self.lidar_refiner = (
            LidarRefiner(model_cfg.lidar_channels, model_cfg.bev_channels)
            if model_cfg.use_lidar
            else None
        )
        self.occ_head = OccupancyHead(
            bev_channels=model_cfg.bev_channels,
            num_classes=model_cfg.num_classes,
            z_bins=config.grid.occ_zbins,
        )

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        lidar_bev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        camera_features = self.camera_encoder(images)
        bev_features = self.projector(camera_features, intrinsics, extrinsics)
        bev_features = self.bev_encoder(bev_features)
        if self.lidar_refiner is not None:
            bev_features = self.lidar_refiner(bev_features, lidar_bev)
        return self.occ_head(bev_features)
