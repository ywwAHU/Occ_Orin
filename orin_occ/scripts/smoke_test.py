from __future__ import annotations

import argparse

import torch

from orin_occ.config import load_config
from orin_occ.models import OrinOccNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    model = OrinOccNet(config)
    model.eval()

    batch_size = 1
    camera_count = len(config.camera_names)
    images = torch.randn(batch_size, camera_count, 3, config.image.height, config.image.width)
    intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(batch_size, camera_count, 1, 1)
    extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(batch_size, camera_count, 1, 1)
    lidar_bev = torch.randn(
        batch_size,
        config.model.lidar_channels,
        config.grid.bev_height,
        config.grid.bev_width,
    )

    with torch.no_grad():
        logits = model(images, intrinsics, extrinsics, lidar_bev)

    expected_shape = (
        batch_size,
        config.model.num_classes,
        config.grid.occ_zbins,
        config.grid.bev_height,
        config.grid.bev_width,
    )
    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(f"Unexpected shape: got {tuple(logits.shape)}, expected {expected_shape}")

    print(f"smoke test passed, logits shape={tuple(logits.shape)}")


if __name__ == "__main__":
    main()
