from __future__ import annotations

import argparse

from orin_occ.config import load_config
from orin_occ.export import export_onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = export_onnx(config, checkpoint=args.checkpoint)
    print(f"exported ONNX to {output_path}")


if __name__ == "__main__":
    main()
