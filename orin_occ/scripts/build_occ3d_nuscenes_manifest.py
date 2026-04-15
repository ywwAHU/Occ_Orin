from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from orin_occ.config import load_config


def load_table(root: Path, version: str, name: str) -> list[dict[str, Any]]:
    table_path = root / version / f"{name}.json"
    with open(table_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def quat_to_rotmat(quat_wxyz: list[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat_wxyz]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def sensor_to_ego_matrix(translation: list[float], rotation: list[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = quat_to_rotmat(rotation)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
    return matrix


def ego_to_sensor_matrix(translation: list[float], rotation: list[float]) -> np.ndarray:
    sensor_to_ego = sensor_to_ego_matrix(translation, rotation)
    rotation_part = sensor_to_ego[:3, :3]
    translation_part = sensor_to_ego[:3, 3]

    inverse = np.eye(4, dtype=np.float32)
    inverse[:3, :3] = rotation_part.T
    inverse[:3, 3] = -(rotation_part.T @ translation_part)
    return inverse


def build_occ_map(occ3d_root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for labels_path in occ3d_root.rglob("labels.npz"):
        token = labels_path.parent.name
        mapping[token] = str(labels_path)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--nuscenes-root", required=True)
    parser.add_argument("--nuscenes-version", required=True, choices=["v1.0-mini", "v1.0-trainval"])
    parser.add_argument("--occ3d-root", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-missing-occupancy", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    nuscenes_root = Path(args.nuscenes_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_table = load_table(nuscenes_root, args.nuscenes_version, "sample")
    sample_data_table = load_table(nuscenes_root, args.nuscenes_version, "sample_data")
    calibrated_sensor_table = load_table(nuscenes_root, args.nuscenes_version, "calibrated_sensor")
    sensor_table = load_table(nuscenes_root, args.nuscenes_version, "sensor")

    sample_by_token = {item["token"]: item for item in sample_table}
    sample_data_by_token = {item["token"]: item for item in sample_data_table}
    calibrated_sensor_by_token = {item["token"]: item for item in calibrated_sensor_table}
    sensor_by_token = {item["token"]: item for item in sensor_table}

    sample_camera_map: dict[str, dict[str, str]] = {}
    for sample_data in sample_data_table:
        if not sample_data.get("is_key_frame", False):
            continue

        calibrated_sensor = calibrated_sensor_by_token[sample_data["calibrated_sensor_token"]]
        sensor = sensor_by_token[calibrated_sensor["sensor_token"]]
        channel = sensor["channel"]

        sample_token = sample_data["sample_token"]
        if sample_token not in sample_camera_map:
            sample_camera_map[sample_token] = {}
        sample_camera_map[sample_token][channel] = sample_data["token"]

    occ_paths: dict[str, str] = {}
    if args.occ3d_root:
        occ_paths = build_occ_map(Path(args.occ3d_root))

    written = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for sample_token, sample in sample_by_token.items():
            occupancy_path = occ_paths.get(sample_token, "")
            if not occupancy_path and not args.allow_missing_occupancy:
                continue

            cameras: dict[str, Any] = {}
            missing_camera = False
            sample_channels = sample_camera_map.get(sample_token, {})
            for camera_name in config.camera_names:
                sample_data_token = sample_channels.get(camera_name)
                if not sample_data_token:
                    missing_camera = True
                    break

                sample_data = sample_data_by_token[sample_data_token]
                calibrated_sensor = calibrated_sensor_by_token[sample_data["calibrated_sensor_token"]]

                image_path = str(nuscenes_root / sample_data["filename"])
                ego_to_camera = ego_to_sensor_matrix(
                    calibrated_sensor["translation"],
                    calibrated_sensor["rotation"],
                )
                cameras[camera_name] = {
                    "image": image_path,
                    "intrinsic": calibrated_sensor["camera_intrinsic"],
                    "ego_to_camera": ego_to_camera.tolist(),
                }

            if missing_camera:
                continue

            row = {
                "sample_id": sample_token,
                "cameras": cameras,
                "lidar_bev": "",
                "occupancy": occupancy_path,
            }
            handle.write(json.dumps(row) + "\n")
            written += 1

            if args.limit > 0 and written >= args.limit:
                break

    print(f"wrote {written} samples to {output_path}")


if __name__ == "__main__":
    main()
