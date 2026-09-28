"""Render a fast graph-derived pseudo-electron cloud."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from molai.fields import ElectronCloud2D, ElectronCloudConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", default="CC(=O)Oc1ccccc1C(=O)O")
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument("--output", type=Path, default=Path("artifacts/electron_cloud_aspirin.png"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ElectronCloudConfig(resolution=args.resolution)
    result = ElectronCloud2D(config, args.device).render(args.smiles)
    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    artist = axis.imshow(
        result.field[0].detach().cpu(),
        origin="lower",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    axis.set_title(f"Graph-derived electron cloud\n{result.canonical_smiles}")
    axis.set_axis_off()
    figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
