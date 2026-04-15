from __future__ import annotations

from pathlib import Path

import torch

from orin_occ.config import ProjectConfig
from orin_occ.models import OrinOccNet


def export_onnx(config: ProjectConfig, checkpoint: str | None = None) -> Path:
    model = OrinOccNet(config)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
    model.eval()

    batch_size = 1
    camera_count = len(config.camera_names)
    image_h = config.image.height
    image_w = config.image.width

    images = torch.randn(batch_size, camera_count, 3, image_h, image_w, dtype=torch.float32)
    intrinsics = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(batch_size, camera_count, 1, 1)
    extrinsics = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(batch_size, camera_count, 1, 1)
    lidar_bev = torch.randn(
        batch_size,
        config.model.lidar_channels,
        config.grid.bev_height,
        config.grid.bev_width,
        dtype=torch.float32,
    )

    output_path = Path(config.export.onnx_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_names = ["images", "intrinsics", "extrinsics", "lidar_bev"]
    dynamic_axes = None
    if config.export.dynamic_batch:
        dynamic_axes = {name: {0: "batch"} for name in input_names}
        dynamic_axes["logits"] = {0: "batch"}

    torch.onnx.export(
        model,
        (images, intrinsics, extrinsics, lidar_bev),
        str(output_path),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=config.export.opset,
    )
    return output_path
