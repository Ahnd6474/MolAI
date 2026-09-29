"""Render the charge-conserving expected-valence-charge image sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from molai.fields import ExpectedCharge2D, ExpectedChargeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", default="CC(=O)Oc1ccccc1C(=O)O")
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/expected_charge_aspirin.png")
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--softsign-scale", type=float, default=32.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExpectedChargeConfig(
        resolution=args.resolution, softsign_scale=args.softsign_scale
    )
    result = ExpectedCharge2D(config, args.device).render(args.smiles)
    signed_limit = float(result.signed_charge[0].abs().quantile(0.995))
    potential_limit = float(result.electrostatic_potential[0].abs().quantile(0.995))
    panels = (
        (result.nuclear_density[0], "Valence nuclei", "magma", None),
        (result.bond_density[0], "Sigma + localized pi electrons", "viridis", None),
        (result.lone_pair_density[0], "Lone-pair / radical electrons", "viridis", None),
        (result.delocalized_density[0], "Aromatic delocalized electrons", "viridis", None),
        (result.electron_density[0], "Total expected electron density", "viridis", None),
        (result.signed_charge[0], "Raw Q = nuclei - electrons", "coolwarm", signed_limit),
        (result.field[0], "Bounded softsign(Q)", "coolwarm", 1.0),
        (
            result.electrostatic_potential[0],
            "Electrostatic potential G * Q",
            "coolwarm",
            potential_limit,
        ),
    )
    figure, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    extent = (-config.extent, config.extent, -config.extent, config.extent)
    for axis, (image, title, color_map, symmetric_limit) in zip(
        axes.flat, panels, strict=True
    ):
        options = {"interpolation": "bilinear"}
        if symmetric_limit is not None:
            options.update(vmin=-symmetric_limit, vmax=symmetric_limit)
        artist = axis.imshow(
            image.detach().cpu(), origin="lower", cmap=color_map, extent=extent, **options
        )
        axis.scatter(
            result.coordinates[:, 0].cpu(),
            result.coordinates[:, 1].cpu(),
            marker="x",
            s=13,
            linewidths=0.7,
            color="white" if color_map != "coolwarm" else "black",
        )
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        f"Charge-conserving expected valence field | {result.canonical_smiles}\n"
        f"integral(nuclei)={result.integrated_nuclear_charge:.4f}, "
        f"integral(electrons)={result.integrated_electron_count:.4f}, "
        f"integral(Q)={result.integrated_charge:.4f}, formal charge={result.formal_charge}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    report = {
        "smiles": result.canonical_smiles,
        "representation": "expected_valence_charge",
        "resolution": config.resolution,
        "formal_charge": result.formal_charge,
        "expected_electron_count": result.expected_electron_count,
        "integrated_nuclear_charge": result.integrated_nuclear_charge,
        "integrated_electron_count": result.integrated_electron_count,
        "integrated_charge": result.integrated_charge,
        "config": config.to_dict(),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output.resolve()}")
    print(f"Saved {report_path.resolve()}")


if __name__ == "__main__":
    main()
