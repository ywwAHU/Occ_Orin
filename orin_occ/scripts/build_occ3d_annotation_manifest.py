from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from orin_occ.config import load_config


def pose_to_ego_to_camera(extrinsic: dict[str, Any]) -> list[list[float]]:
    import numpy as np

    translation = extrinsic["translation"]
    rotation = extrinsic["rotation"]
    w, x, y, z = [float(v) for v in rotation]

    sensor_to_ego = np.eye(4, dtype=np.float32)
    sensor_to_ego[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    sensor_to_ego[:3, 3] = translation

    ego_to_sensor = np.eye(4, dtype=np.float32)
    rotation_part = sensor_to_ego[:3, :3]
    translation_part = sensor_to_ego[:3, 3]
    ego_to_sensor[:3, :3] = rotation_part.T
    ego_to_sensor[:3, 3] = -(rotation_part.T @ translation_part)
    return ego_to_sensor.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.annotations, "r", encoding="utf-8") as handle:
        annotations = json.load(handle)

    if args.split == "train":
        scene_names = set(annotations["train_split"])
    elif args.split == "val":
        scene_names = set(annotations["val_split"])
    else:
        scene_names = set(annotations["scene_infos"].keys())

    written = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for scene_name in scene_names:
            scene_infos = annotations["scene_infos"].get(scene_name, {})
            for sample_token, sample_info in scene_infos.items():
                camera_sensor = sample_info["camera_sensor"]
                cameras: dict[str, Any] = {}
                missing_camera = False

                for camera_name in config.camera_names:
                    camera_info = camera_sensor.get(camera_name)
                    if camera_info is None:
                        missing_camera = True
                        break

                    cameras[camera_name] = {
                        "image": str(dataset_root / "imgs" / camera_info["img_path"]),
                        "intrinsic": camera_info["intrinsics"],
                        "ego_to_camera": pose_to_ego_to_camera(camera_info["extrinsic"]),
                    }

                if missing_camera:
                    continue

                occupancy_path = str(dataset_root / sample_info["gt_path"])
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

            if args.limit > 0 and written >= args.limit:
                break

    print(f"wrote {written} samples to {output_path}")


if __name__ == "__main__":
    main()
