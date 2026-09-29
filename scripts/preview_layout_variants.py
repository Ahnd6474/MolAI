"""Generate layout and hydrogen-influence comparison drafts for selected molecules."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from molai.fields import (
    CompiledExpectedChargeMolecule,
    ExpectedCharge2D,
    ExpectedChargeConfig,
)


@dataclass(frozen=True, slots=True)
class MoleculeDraft:
    name: str
    smiles: str


@dataclass(frozen=True, slots=True)
class LayoutVariant:
    slug: str
    label: str
    hydrogen_influence: float
    optimize_layout: bool


MOLECULES = (
    MoleculeDraft("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    MoleculeDraft(
        "atp-4",
        "C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H](O3)"
        "COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])O)O)N",
    ),
)

VARIANTS = (
    LayoutVariant("original", "Original", 1.0, False),
    LayoutVariant("layout-only", "Layout optimization only", 1.0, True),
    LayoutVariant("h25-only", "H influence 25% only", 0.25, False),
    LayoutVariant("layout-h25", "Layout optimization + H 25%", 0.25, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/layout_variant_drafts")
    )
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def render_variant(
    molecule: MoleculeDraft,
    variant: LayoutVariant,
    resolution: int,
    device: str,
) -> tuple[torch.Tensor, CompiledExpectedChargeMolecule, float]:
    config = ExpectedChargeConfig(
        resolution=resolution,
        hydrogen_influence=variant.hydrogen_influence,
        optimize_layout=variant.optimize_layout,
        layout_candidates=4,
        layout_samples_per_candidate=12,
    )
    renderer = ExpectedCharge2D(config, device)
    compiled = renderer.compile_molecule(molecule.smiles)
    with torch.inference_mode():
        result = renderer.render_training_compiled_batch(
            [compiled], "electrostatic_potential"
        )
    return (
        result.channel[0, 0].detach().float().cpu(),
        compiled,
        float(result.integrated_charge[0]),
    )


def draw_panel(
    axis: plt.Axes,
    image: torch.Tensor,
    compiled: CompiledExpectedChargeMolecule,
    title: str,
    charge: float,
    limit: float,
) -> None:
    axis.imshow(
        image,
        origin="lower",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        extent=(-1.0, 1.0, -1.0, 1.0),
        interpolation="bilinear",
    )
    axis.set_title(
        f"{title}\nQ={charge:.4f} · crossings={compiled.bond_crossings} "
        f"· score={compiled.layout_score:.3f}",
        fontsize=9,
    )
    axis.set_axis_off()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "representation": "electrostatic_potential_linear_g_times_q",
        "resolution": args.resolution,
        "layout_candidates": 4,
        "explicit_overlay": False,
        "molecules": [],
    }

    for molecule in MOLECULES:
        rendered = [
            render_variant(molecule, variant, args.resolution, args.device)
            for variant in VARIANTS
        ]
        limit = max(
            float(torch.cat([image.abs().flatten() for image, _, _ in rendered]).quantile(0.995)),
            1e-8,
        )
        figure, axes = plt.subplots(1, len(VARIANTS), figsize=(20, 5.2))
        figure.subplots_adjust(left=0.015, right=0.995, bottom=0.025, top=0.82, wspace=0.05)
        reports: list[dict[str, object]] = []
        for axis, variant, (image, compiled, charge) in zip(
            axes, VARIANTS, rendered, strict=True
        ):
            draw_panel(axis, image, compiled, variant.label, charge, limit)
            individual_figure, individual_axis = plt.subplots(
                figsize=(5.2, 5.2), constrained_layout=True
            )
            draw_panel(
                individual_axis, image, compiled, variant.label, charge, limit
            )
            individual_path = args.output_dir / f"{molecule.name}-{variant.slug}.png"
            individual_figure.savefig(individual_path, dpi=170)
            plt.close(individual_figure)
            reports.append(
                {
                    "variant": variant.slug,
                    "hydrogen_influence": variant.hydrogen_influence,
                    "optimize_layout": variant.optimize_layout,
                    "integrated_charge": charge,
                    "layout_score": compiled.layout_score,
                    "bond_crossings": compiled.bond_crossings,
                    "image": str(individual_path.resolve()),
                }
            )
        figure.suptitle(
            f"{molecule.name} · Electrostatic potential G * Q · common color scale ±{limit:.3g}",
            y=0.97,
        )
        comparison_path = args.output_dir / f"{molecule.name}-comparison.png"
        figure.savefig(comparison_path, dpi=170)
        plt.close(figure)
        manifest["molecules"].append(
            {
                "name": molecule.name,
                "smiles": molecule.smiles,
                "comparison": str(comparison_path.resolve()),
                "variants": reports,
            }
        )
        print(f"Saved {comparison_path.resolve()}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
