"""Solve and visualize Kohn-Sham orbitals and bonding diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from molai.dft import KohnSham2D, KohnSham2DConfig, canonical_nuclei, collate_nuclei


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", default="CC(=O)Oc1ccccc1C(=O)O")
    parser.add_argument("--config", type=Path, default=Path("configs/kohn_sham.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/kohn_sham_preview.png"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--scf-iterations", type=int)
    parser.add_argument("--orbital-steps", type=int)
    parser.add_argument("--diagnostics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = yaml.safe_load(args.config.read_text(encoding="utf-8"))["kohn_sham"]
    if args.resolution is not None:
        values["resolution"] = args.resolution
    if args.scf_iterations is not None:
        values["scf_iterations"] = args.scf_iterations
    if args.orbital_steps is not None:
        values["orbital_steps"] = args.orbital_steps
    config = KohnSham2DConfig(**values)
    layout = yaml.safe_load(args.config.read_text(encoding="utf-8"))["layout"]
    molecule = canonical_nuclei(
        args.smiles,
        extent=config.extent,
        margin=layout["margin"],
        target_bond_length=layout["target_bond_length"],
        collapse_hydrogens=layout["collapse_hydrogens"],
    )
    result = KohnSham2D(config, args.device).solve(collate_nuclei([molecule], args.device))

    occupied = torch.nonzero(result.occupancies[0] > 0, as_tuple=False).flatten()
    homo_index = int(occupied[-1])
    homo = result.orbitals[0, homo_index]
    deformation = result.deformation_density[0, 0]
    deformation_limit = float(deformation.abs().quantile(0.995))
    localization = result.electron_localization[0, 0].clone()
    density = result.electron_density[0, 0]
    localization[density < density.max() * 1e-3] = 0.0

    if not args.diagnostics:
        figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
        artist = axis.imshow(
            result.field[0, 0].detach().cpu(),
            origin="lower",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        axis.set_title(
            f"Kohn-Sham signed density field\n{molecule.canonical_smiles}\n"
            f"SCF={result.iterations} converged={result.converged} "
            f"delta_n={result.final_density_change:.3g}"
        )
        axis.set_axis_off()
        figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=180)
        print(f"Saved {args.output.resolve()}")
        return

    panels = (
        (density, "Kohn-Sham density", "viridis", None),
        (deformation, "Deformation density", "coolwarm", deformation_limit),
        (localization, "Electron localization", "magma", None),
        (result.bond_order_density[0, 0], "Projected bond order", "magma", None),
        (homo, f"HOMO orbital {homo_index}", "coolwarm", float(homo.abs().max())),
        (result.field[0, 0], "Fused scalar field", "inferno", None),
    )
    figure, axes = plt.subplots(1, 6, figsize=(24, 4), constrained_layout=True)
    for axis, (image, title, color_map, symmetric_limit) in zip(axes, panels, strict=True):
        options = {}
        if symmetric_limit is not None and color_map == "coolwarm":
            options = {"vmin": -symmetric_limit, "vmax": symmetric_limit}
        artist = axis.imshow(image.detach().cpu(), origin="lower", cmap=color_map, **options)
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        f"{molecule.canonical_smiles} | Q={float(result.integrated_charge[0]):.4f} | "
        f"SCF={result.iterations} converged={result.converged} "
        f"delta_n={result.final_density_change:.3g}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
