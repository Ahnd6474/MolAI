"""Create a zero-copy geometric-noise view over molecular field shards."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=int, default=64)
    parser.add_argument("--sigma-min-ratio", type=float, default=0.01)
    parser.add_argument("--sigma-max-ratio", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float)
    parser.add_argument("--variants-per-field", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260930)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    source_manifest_path = source / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"missing source manifest: {source_manifest_path}")
    if args.levels < 2:
        raise ValueError("levels must be at least two")
    if not 0.0 < args.sigma_min_ratio < args.sigma_max_ratio:
        raise ValueError("sigma ratios must satisfy 0 < min < max")
    if args.variants_per_field < 1:
        raise ValueError("variants-per-field must be positive")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_records = sum(int(shard["records"]) for shard in source_manifest["shards"])
    noise_scale = args.noise_scale
    if noise_scale is None:
        statistics = source_manifest.get("statistics", {})
        elements = int(statistics.get("elements", 0))
        sum_of_squares = float(statistics.get("sum_of_squares", 0.0))
        if elements < 1 or sum_of_squares <= 0.0:
            raise ValueError(
                "source manifest has no usable statistics; pass --noise-scale"
            )
        noise_scale = math.sqrt(sum_of_squares / elements)
    if noise_scale <= 0.0:
        raise ValueError("noise-scale must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    relative_source = os.path.relpath(source, args.output.resolve())
    manifest = {
        "format": "molai-noise-view-v1",
        "source": relative_source,
        "source_records": source_records,
        "records": source_records * args.variants_per_field,
        "variants_per_field": args.variants_per_field,
        "seed": args.seed,
        "value_space": "raw",
        "scale_reference": {"type": "rms", "value": noise_scale},
        "noise_schedule": {
            "type": "geometric_ve",
            "levels": args.levels,
            "sigma_min": noise_scale * args.sigma_min_ratio,
            "sigma_max": noise_scale * args.sigma_max_ratio,
            "sigma_min_ratio": args.sigma_min_ratio,
            "sigma_max_ratio": args.sigma_max_ratio,
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"prepared {manifest['records']:,} zero-copy noise recipes over "
        f"{source_records:,} source fields"
    )


if __name__ == "__main__":
    main()
