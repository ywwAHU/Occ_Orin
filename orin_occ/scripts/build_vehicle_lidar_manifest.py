from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from orin_occ.config import load_config
from orin_occ.preprocess import build_sample_lidar_bev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--bev-dir", required=True)
    parser.add_argument("--bin-columns", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    input_manifest = Path(args.input_manifest)
    output_manifest = Path(args.output_manifest)
    bev_dir = Path(args.bev_dir)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    bev_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(input_manifest, "r", encoding="utf-8") as source, open(
        output_manifest, "w", encoding="utf-8"
    ) as sink:
        for line in source:
            line = line.strip()
            if not line:
                continue

            sample: dict[str, Any] = json.loads(line)
            sample_id = str(sample.get("sample_id", f"sample_{written:06d}"))
            lidar_bev = build_sample_lidar_bev(
                sample,
                grid=config.grid,
                channel_count=config.model.lidar_channels,
                bin_columns=args.bin_columns,
            )

            lidar_bev_path = bev_dir / f"{sample_id}.npy"
            np.save(lidar_bev_path, lidar_bev.astype(np.float32))

            row: dict[str, Any] = {
                "sample_id": sample_id,
                "cameras": sample["cameras"],
                "lidar_bev": str(lidar_bev_path),
                "occupancy": sample.get("occupancy", ""),
            }

            for key in ("timestamp", "metadata", "lidars"):
                if key in sample:
                    row[key] = sample[key]

            sink.write(json.dumps(row) + "\n")
            written += 1

            if args.limit > 0 and written >= args.limit:
                break

    print(f"wrote {written} samples to {output_manifest}")


if __name__ == "__main__":
    main()
