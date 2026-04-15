from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from orin_occ.config import GridConfig, ImageConfig


class GeometryAwareProjector(nn.Module):
    def __init__(self, grid: GridConfig, image: ImageConfig) -> None:
        super().__init__()
        self.grid = grid
        self.image = image

        x_min, x_max = grid.x_range
        y_min, y_max = grid.y_range
        xs = torch.linspace(x_min, x_max, grid.bev_width)
        ys = torch.linspace(y_min, y_max, grid.bev_height)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        ego_points = torch.stack(
            [
                grid_x.reshape(-1),
                grid_y.reshape(-1),
                torch.zeros(grid.bev_height * grid.bev_width),
                torch.ones(grid.bev_height * grid.bev_width),
            ],
            dim=0,
        )
        self.register_buffer("ego_points", ego_points, persistent=False)

    def forward(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, camera_count, channels, feat_height, feat_width = features.shape
        point_count = self.grid.bev_height * self.grid.bev_width

        world_points = self.ego_points.unsqueeze(0).expand(batch_size, -1, -1)
        feature_sum = torch.zeros(
            batch_size,
            channels,
            self.grid.bev_height,
            self.grid.bev_width,
            device=features.device,
            dtype=features.dtype,
        )
        feature_count = torch.zeros_like(feature_sum[:, :1])

        scale_x = feat_width / float(self.image.width)
        scale_y = feat_height / float(self.image.height)

        for camera_idx in range(camera_count):
            camera_features = features[:, camera_idx]
            ego_to_camera = extrinsics[:, camera_idx]
            camera_intrinsic = intrinsics[:, camera_idx]

            cam_points = torch.matmul(ego_to_camera, world_points)
            xyz = cam_points[:, :3, :]
            uvw = torch.matmul(camera_intrinsic, xyz)

            depth = uvw[:, 2, :].clamp(min=1e-5)
            u = (uvw[:, 0, :] / depth) * scale_x
            v = (uvw[:, 1, :] / depth) * scale_y

            valid = (
                (depth > 1e-3)
                & (u >= 0.0)
                & (u <= feat_width - 1)
                & (v >= 0.0)
                & (v <= feat_height - 1)
            )

            x_norm = (u / max(feat_width - 1, 1)) * 2.0 - 1.0
            y_norm = (v / max(feat_height - 1, 1)) * 2.0 - 1.0
            grid = torch.stack([x_norm, y_norm], dim=-1).view(
                batch_size, self.grid.bev_height, self.grid.bev_width, 2
            )

            sampled = F.grid_sample(
                camera_features,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            valid_mask = valid.view(batch_size, 1, self.grid.bev_height, self.grid.bev_width)
            feature_sum = feature_sum + sampled * valid_mask
            feature_count = feature_count + valid_mask

        return feature_sum / feature_count.clamp(min=1.0)
