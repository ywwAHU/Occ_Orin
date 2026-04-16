from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from orin_occ.config import GridConfig


def _load_point_cloud(path: str | Path, bin_columns: int) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".npy":
        points = np.load(path).astype(np.float32)
    elif path.suffix.lower() == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        if bin_columns <= 0 or raw.size % bin_columns != 0:
            raise ValueError(
                f"Cannot reshape {path} with bin_columns={bin_columns}; "
                f"raw float count={raw.size}"
            )
        points = raw.reshape(-1, bin_columns).astype(np.float32)
    else:
        raise ValueError(f"Unsupported point cloud format: {path.suffix}")

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected point cloud shape [N, >=3], got {points.shape}")
    return points


def _transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([points_xyz, ones], axis=1)
    transformed = hom @ transform.T
    return transformed[:, :3]


def _points_to_ego(points: np.ndarray, lidar_info: dict[str, Any]) -> np.ndarray:
    xyz = points[:, :3]
    if "lidar_to_ego" in lidar_info:
        transform = np.asarray(lidar_info["lidar_to_ego"], dtype=np.float32)
        return _transform_points(xyz, transform)
    if "ego_to_lidar" in lidar_info:
        ego_to_lidar = np.asarray(lidar_info["ego_to_lidar"], dtype=np.float32)
        lidar_to_ego = np.linalg.inv(ego_to_lidar).astype(np.float32)
        return _transform_points(xyz, lidar_to_ego)
    return xyz


def build_lidar_bev(
    points_ego: np.ndarray,
    grid: GridConfig,
    channel_count: int,
    intensities: np.ndarray | None = None,
) -> np.ndarray:
    if points_ego.ndim != 2 or points_ego.shape[1] != 3:
        raise ValueError(f"Expected ego points shape [N, 3], got {points_ego.shape}")

    bev_height = grid.bev_height
    bev_width = grid.bev_width
    base = np.zeros((4, bev_height, bev_width), dtype=np.float32)
    if points_ego.shape[0] == 0:
        if channel_count <= 4:
            return base[:channel_count]
        padded = np.zeros((channel_count, bev_height, bev_width), dtype=np.float32)
        padded[:4] = base
        return padded

    x = points_ego[:, 0]
    y = points_ego[:, 1]
    z = points_ego[:, 2]
    x_min, x_max = grid.x_range
    y_min, y_max = grid.y_range
    z_min, z_max = grid.z_range

    valid = (
        (x >= x_min)
        & (x < x_max)
        & (y >= y_min)
        & (y < y_max)
        & (z >= z_min)
        & (z <= z_max)
    )
    if intensities is None:
        intensities = np.ones((points_ego.shape[0],), dtype=np.float32)
    intensities = intensities.astype(np.float32)

    x = x[valid]
    y = y[valid]
    z = z[valid]
    intensities = intensities[valid]
    if x.size == 0:
        if channel_count <= 4:
            return base[:channel_count]
        padded = np.zeros((channel_count, bev_height, bev_width), dtype=np.float32)
        padded[:4] = base
        return padded

    x_idx = np.floor((x - x_min) / max(x_max - x_min, 1e-6) * bev_width).astype(np.int64)
    y_idx = np.floor((y - y_min) / max(y_max - y_min, 1e-6) * bev_height).astype(np.int64)
    x_idx = np.clip(x_idx, 0, bev_width - 1)
    y_idx = np.clip(y_idx, 0, bev_height - 1)
    flat_idx = y_idx * bev_width + x_idx
    cell_count = bev_height * bev_width

    counts = np.bincount(flat_idx, minlength=cell_count).astype(np.float32)
    occupancy = (counts > 0).astype(np.float32)
    density = np.clip(np.log1p(counts) / np.log(32.0), 0.0, 1.0)

    max_height = np.full((cell_count,), z_min, dtype=np.float32)
    np.maximum.at(max_height, flat_idx, z.astype(np.float32))
    normalized_height = np.zeros_like(max_height)
    occupied = counts > 0
    normalized_height[occupied] = np.clip(
        (max_height[occupied] - z_min) / max(z_max - z_min, 1e-6),
        0.0,
        1.0,
    )

    intensity_sum = np.bincount(flat_idx, weights=intensities, minlength=cell_count).astype(np.float32)
    mean_intensity = np.zeros((cell_count,), dtype=np.float32)
    mean_intensity[occupied] = np.clip(intensity_sum[occupied] / counts[occupied], 0.0, 1.0)

    base[0] = occupancy.reshape(bev_height, bev_width)
    base[1] = density.reshape(bev_height, bev_width)
    base[2] = normalized_height.reshape(bev_height, bev_width)
    base[3] = mean_intensity.reshape(bev_height, bev_width)

    if channel_count <= 4:
        return base[:channel_count]

    padded = np.zeros((channel_count, bev_height, bev_width), dtype=np.float32)
    padded[:4] = base
    return padded


def build_sample_lidar_bev(sample: dict[str, Any], grid: GridConfig, channel_count: int, bin_columns: int) -> np.ndarray:
    lidar_entries = sample.get("lidars")
    if not isinstance(lidar_entries, dict) or not lidar_entries:
        raise ValueError("Expected non-empty sample['lidars'] dictionary")

    ego_points: list[np.ndarray] = []
    ego_intensities: list[np.ndarray] = []

    for lidar_name, lidar_info in lidar_entries.items():
        if not isinstance(lidar_info, dict) or "points" not in lidar_info:
            raise ValueError(f"Invalid lidar entry for {lidar_name!r}")

        points = _load_point_cloud(lidar_info["points"], bin_columns=bin_columns)
        xyz_ego = _points_to_ego(points, lidar_info)
        ego_points.append(xyz_ego.astype(np.float32))
        if points.shape[1] >= 4:
            ego_intensities.append(points[:, 3].astype(np.float32))
        else:
            ego_intensities.append(np.ones((points.shape[0],), dtype=np.float32))

    fused_points = np.concatenate(ego_points, axis=0) if ego_points else np.zeros((0, 3), dtype=np.float32)
    fused_intensities = (
        np.concatenate(ego_intensities, axis=0) if ego_intensities else np.zeros((0,), dtype=np.float32)
    )
    return build_lidar_bev(fused_points, grid=grid, channel_count=channel_count, intensities=fused_intensities)
