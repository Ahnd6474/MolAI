"""Solve and visualize one molecular 2D pseudo-DFT field."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from molai.dft import DFT2DConfig, OrbitalFreeDFT2D, canonical_nuclei, collate_nuclei


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", default="CC(=O)Oc1ccccc1C(=O)O")
    parser.add_argument("--config", type=Path, default=Path("configs/dft.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/dft_preview.png"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--softsign-scale", type=float, default=32.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configuration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    values = configuration["dft"]
    if args.resolution is not None:
        values["resolution"] = args.resolution
    if args.steps is not None:
        values["steps"] = args.steps
    config = DFT2DConfig(**values)
    layout = configuration["layout"]
    molecule = canonical_nuclei(
        args.smiles,
        extent=config.extent,
        margin=layout["margin"],
        target_bond_length=layout["target_bond_length"],
        collapse_hydrogens=layout["collapse_hydrogens"],
    )
    result = OrbitalFreeDFT2D(config, args.device).solve(collate_nuclei([molecule], args.device))

    signed_density = result.signed_density[0, 0]
    softsign_field = signed_density / (args.softsign_scale + signed_density.abs())
    panels = (
        (result.core_density[0, 0], "Effective cores", "magma"),
        (result.electron_density[0, 0], "Valence density", "viridis"),
        (signed_density, "Signed density", "coolwarm"),
        (result.field[0, 0], "tanh(0.035 Q)", "coolwarm"),
        (softsign_field, f"softsign(Q/{args.softsign_scale:g})", "coolwarm"),
    )
    figure, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
    for axis, (image, title, color_map) in zip(axes, panels, strict=True):
        artist = axis.imshow(image.detach().cpu(), origin="lower", cmap=color_map)
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        f"{molecule.canonical_smiles} | Q={float(result.integrated_charge[0]):.4f} | "
        f"iterations={result.iterations} converged={result.converged}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
