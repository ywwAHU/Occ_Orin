from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ImageConfig:
    height: int
    width: int


@dataclass
class GridConfig:
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    bev_height: int
    bev_width: int
    occ_zbins: int


@dataclass
class ModelConfig:
    backbone_base_channels: int
    feature_channels: int
    bev_channels: int
    num_classes: int
    use_lidar: bool
    lidar_channels: int


@dataclass
class TrainingConfig:
    manifest_path: str
    synthetic_samples: int
    batch_size: int
    num_workers: int
    epochs: int
    learning_rate: float
    weight_decay: float
    amp: bool
    save_dir: str


@dataclass
class ExportConfig:
    onnx_path: str
    opset: int
    dynamic_batch: bool


@dataclass
class ProjectConfig:
    project_name: str
    camera_names: list[str]
    image: ImageConfig
    grid: GridConfig
    model: ModelConfig
    training: TrainingConfig
    export: ExportConfig


def _as_tuple(values: list[float]) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"Expected length-2 list, got {values!r}")
    return float(values[0]), float(values[1])


def load_config(path: str | Path) -> ProjectConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw: dict[str, Any] = json.load(handle)

    image = ImageConfig(**raw["image"])
    grid = GridConfig(
        x_range=_as_tuple(raw["grid"]["x_range"]),
        y_range=_as_tuple(raw["grid"]["y_range"]),
        z_range=_as_tuple(raw["grid"]["z_range"]),
        bev_height=int(raw["grid"]["bev_height"]),
        bev_width=int(raw["grid"]["bev_width"]),
        occ_zbins=int(raw["grid"]["occ_zbins"]),
    )
    model = ModelConfig(**raw["model"])
    training = TrainingConfig(**raw["training"])
    export = ExportConfig(**raw["export"])

    return ProjectConfig(
        project_name=raw["project_name"],
        camera_names=list(raw["camera_names"]),
        image=image,
        grid=grid,
        model=model,
        training=training,
        export=export,
    )
