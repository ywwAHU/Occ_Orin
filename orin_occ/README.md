# Orin OCC Baseline

This project is a clean OCC baseline tailored for the current vehicle setup and converged to the V1 camera subset:

- 3 front cameras
- 4 surround cameras
- 3 rear low-light cameras present on the vehicle but excluded from the V1 OCC baseline
- 1 front Robosense-M1P
- 4 surround Robosense-E1R
- 1 Continental ARS548 mmWave radar
- 1 INS-5711 with RTK
- target deployment on Jetson AGX Orin

The design goal is not to reproduce a paper one-to-one. Instead, it provides a practical engineering baseline that is:

- camera-dominant
- geometry-aware
- lidar-extendable
- simple enough to train and export
- easier to adapt for TensorRT deployment later

## Design Summary

The model follows a deployment-friendly path:

1. Shared CNN encodes all camera images.
2. A geometry-aware projector samples image features onto a BEV grid using camera intrinsics and `ego_to_camera` extrinsics.
3. A lightweight BEV encoder refines the fused feature map.
4. An optional fused-lidar BEV branch injects pre-rasterized multi-lidar features.
5. A channel-to-height occupancy head predicts `num_classes x z_bins`.

This is intentionally closer to a FlashOCC-style engineering direction than to a sparse 3D convolution stack.

## Repository Layout

```text
orin_occ/
  configs/
  scripts/
  src/orin_occ/
```

## Expected Data Format

Training samples are described by a JSONL manifest. Each line is one sample:

```json
{
  "sample_id": "frame_000001",
  "cameras": {
    "front_left": {
      "image": "/path/to/front_left.jpg",
      "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "ego_to_camera": [[...4 values...], [...], [...], [...]]
    }
  },
  "lidar_bev": "/path/to/front_lidar_bev.npy",
  "occupancy": "/path/to/occupancy.npy"
}
```

Notes:

- The default vehicle baseline uses 7 cameras: `front_left`, `front_center`, `front_right`, `surround_left_front`, `surround_right_front`, `surround_left_rear`, `surround_right_rear`.
- The 3 rear low-light cameras are intentionally not part of the default V1 training and deployment path.
- `intrinsic` must match the resized training image resolution.
- `ego_to_camera` maps ego-frame homogeneous points to the camera frame.
- `lidar_bev` is optional. When used, it is expected to be a `float32` array with shape `[C, bev_h, bev_w]`.
- In the current vehicle setup, `lidar_bev` is intended to be a pre-fused BEV tensor built from the front `M1P` and the 4 surrounding `E1R` lidars.
- The current baseline does not ingest mmWave directly yet, but the repository layout keeps room for a later radar branch.
- The 4 surround cameras are expected to be undistorted or reprojected before training if their raw images come from 190-degree optics.
- `occupancy` is expected to be an `int64` array with shape `[z_bins, bev_h, bev_w]`.
- `occupancy` may also point to an `Occ3D` `labels.npz` file. In that case the loader will read:
  - `semantics`
  - `mask_camera`
  - `mask_lidar`

### Multi-Lidar Vehicle Bridge

For the current vehicle, the model still consumes a fused `lidar_bev` tensor at training time. To bridge the 5 raw lidars into that format, the repository now includes a preprocessing script.

Input source manifest example:

```json
{
  "sample_id": "frame_000001",
  "cameras": {
    "front_left": {
      "image": "/data/front_left.jpg",
      "intrinsic": [[...], [...], [...]],
      "ego_to_camera": [[...], [...], [...], [...]]
    }
  },
  "lidars": {
    "front_m1p": {
      "points": "/data/lidar/front_m1p.npy",
      "lidar_to_ego": [[...], [...], [...], [...]]
    },
    "left_e1r": {
      "points": "/data/lidar/left_e1r.npy",
      "lidar_to_ego": [[...], [...], [...], [...]]
    }
  },
  "occupancy": "/data/occ/frame_000001.npz"
}
```

Supported point formats:

- `.npy` with shape `[N, 3]`, `[N, 4]`, or `[N, >=5]`
- `.bin` flat `float32` buffers, together with `--bin-columns`

The script fuses all lidar points into ego frame and writes a `lidar_bev` tensor with 4 default channels:

- occupancy
- log-density
- normalized max height
- mean intensity

Example:

```bash
PYTHONPATH=src python scripts/build_vehicle_lidar_manifest.py \
  --config configs/baseline_orin_camera_lidar.json \
  --input-manifest /path/to/vehicle_raw_train.jsonl \
  --output-manifest work_dirs/manifests/vehicle_train_7cam.jsonl \
  --bev-dir work_dirs/lidar_bev/train
```

If your `.bin` point clouds contain 5 values per point, add:

```bash
  --bin-columns 5
```

For a vehicle-side collection plan and a raw manifest example, see:

- `../调研/OCC数据采集方案.md`
- `examples/vehicle_raw_manifest.example.jsonl`

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a smoke test:

```bash
PYTHONPATH=src python scripts/smoke_test.py --config configs/baseline_orin_camera_lidar.json
```

Train with synthetic data:

```bash
PYTHONPATH=src python scripts/train.py --config configs/baseline_orin_camera_lidar.json
```

Export ONNX:

```bash
PYTHONPATH=src python scripts/export_onnx.py --config configs/baseline_orin_camera_lidar.json
```

## Public Dataset Bootstrap

The project now supports preparing a manifest for `nuScenes + Occ3D`.

Recommended public bootstrap path:

1. Reuse the existing public `nuScenes` data on the server.
2. Download `Occ3D-nuScenes` labels separately.
3. Generate a JSONL manifest with the provided script.

Example:

```bash
PYTHONPATH=src python scripts/build_occ3d_nuscenes_manifest.py \
  --config configs/nuscenes_occ3d_public.json \
  --nuscenes-root /path/to/extracted/nuscenes \
  --nuscenes-version v1.0-trainval \
  --occ3d-root /path/to/Occpancy3D-nuScenes-V1.0/trainval/gts \
  --output work_dirs/manifests/nuscenes_occ3d_trainval.jsonl
```

For a quick pipeline check without labels, use `--allow-missing-occupancy`.

If you already downloaded the official `Occ3D-nuScenes-mini` package with:

- `annotations.json`
- `imgs.tar.gz`
- `gts.tar.gz`

you can skip the raw `nuScenes` table parsing and build the manifest directly from Occ3D annotations:

```bash
PYTHONPATH=src python scripts/build_occ3d_annotation_manifest.py \
  --config configs/occ3d_nuscenes_mini_public.json \
  --annotations /path/to/Occupancy3D-nuScenes-mini/annotations.json \
  --dataset-root /path/to/extracted/Occupancy3D-nuScenes-mini \
  --split train \
  --output work_dirs/manifests/occ3d_nuscenes_mini_train.jsonl
```

Expected extracted layout:

```text
Occupancy3D-nuScenes-mini/
  annotations.json
  imgs/
  gts/
```

### AutoDL Notes

On the current server we found:

- public `nuScenes` archives under `/autodl-pub/data/nuScenes`
- public `SemanticKITTI` under `/autodl-pub/data/SemanticKITTI`

Important note:

- `nuScenes` is provided as archives on the shared disk, not as an extracted working tree.
- `v1.0-mini` can be extracted to a local work directory for quick testing.
- full `v1.0-trainval` is much larger and should be extracted only after storage is planned.

Verified example on the current server:

```bash
mkdir -p /root/autodl-tmp/nuscenes_mini
tar -xzf /autodl-pub/data/nuScenes/Fulldatasetv1.0/Mini/v1.0-mini.tgz -C /root/autodl-tmp/nuscenes_mini
```

## Why This Project Fits The Current Vehicle

- It assumes 7 production cameras as the main sensing backbone: 3 front plus 4 surround.
- It intentionally leaves the 3 rear low-light cameras out of the first-stage baseline to keep the input domain cleaner and the Orin budget tighter.
- It now includes a direct preprocessing bridge from the vehicle's 5 raw lidars into the current fused `lidar_bev` input.
- It avoids sparse convolution dependencies.
- It keeps the occupancy head dense and export-friendly.

## Next Steps

This baseline is meant to be the starting point. The most important follow-up tasks are:

1. Replace synthetic training with the real manifest and calibration files.
2. Improve the projector using your exact camera models and image preprocessing, especially for the 190-degree surround cameras.
3. Add a front-ROI lidar refinement policy if the current rasterized lidar input is too coarse.
4. Benchmark and prune channels for Orin deployment.
5. Revisit the rear low-light cameras later only if night or reverse-scene evaluation shows a clear gap.
