"""Generate 2x4 expected-charge diagnostic images for sample molecules.

Examples
--------
Generate the built-in sample set::

    python scripts/generate_expected_charge_samples.py --device cpu

Generate selected molecules::

    python scripts/generate_expected_charge_samples.py \
        --sample water=O \
        --sample acetate="CC(=O)[O-]"
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from molai.fields import ExpectedCharge2D, ExpectedChargeConfig, ExpectedChargeResult


@dataclass(frozen=True, slots=True)
class MoleculeSample:
    name: str
    smiles: str


DEFAULT_SAMPLES = (
    MoleculeSample("water", "O"),
    MoleculeSample("ammonia", "N"),
    MoleculeSample("carbon_dioxide", "O=C=O"),
    MoleculeSample("benzene", "c1ccccc1"),
    MoleculeSample("acetate", "CC(=O)[O-]"),
    MoleculeSample("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
)


def parse_sample(value: str) -> MoleculeSample:
    """Parse one ``name=SMILES`` command-line value."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("sample must have the form name=SMILES")
    name, smiles = value.split("=", 1)
    name = name.strip()
    smiles = smiles.strip()
    if not name or not smiles:
        raise argparse.ArgumentTypeError("sample name and SMILES must both be non-empty")
    return MoleculeSample(name, smiles)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        action="append",
        type=parse_sample,
        help="Molecule in name=SMILES form; repeat for multiple molecules",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/expected_charge_samples"),
    )
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument("--softsign-scale", type=float, default=32.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def safe_filename(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()).strip("._")
    return normalized or "molecule"


def image_panels(
    result: ExpectedChargeResult,
) -> tuple[tuple[torch.Tensor, str, str, float | None], ...]:
    signed_limit = max(float(result.signed_charge[0].abs().quantile(0.995)), 1e-8)
    potential_limit = max(
        float(result.electrostatic_potential[0].abs().quantile(0.995)), 1e-8
    )
    return (
        (result.nuclear_density[0], "Valence nuclei", "magma", None),
        (result.bond_density[0], "Sigma + localized pi electrons", "viridis", None),
        (result.lone_pair_density[0], "Lone-pair / radical electrons", "viridis", None),
        (
            result.delocalized_density[0],
            "Aromatic delocalized electrons",
            "viridis",
            None,
        ),
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


def render_figure(
    sample: MoleculeSample,
    result: ExpectedChargeResult,
    config: ExpectedChargeConfig,
) -> plt.Figure:
    """Create the standard 2x4 diagnostic figure for one molecule."""
    figure, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    extent = (-config.extent, config.extent, -config.extent, config.extent)
    for axis, (image, title, color_map, symmetric_limit) in zip(
        axes.flat, image_panels(result), strict=True
    ):
        image_options: dict[str, float | str] = {"interpolation": "bilinear"}
        if symmetric_limit is not None:
            image_options.update(vmin=-symmetric_limit, vmax=symmetric_limit)
        elif float(image.max()) <= 1e-12:
            image_options.update(vmin=0.0, vmax=1.0)
        artist = axis.imshow(
            image.detach().cpu(),
            origin="lower",
            cmap=color_map,
            extent=extent,
            **image_options,
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
        f"{sample.name} | {result.canonical_smiles}\n"
        f"integral(nuclei)={result.integrated_nuclear_charge:.4f}, "
        f"integral(electrons)={result.integrated_electron_count:.4f}, "
        f"integral(Q)={result.integrated_charge:.4f}, "
        f"formal charge={result.formal_charge}"
    )
    return figure


def sample_report(
    sample: MoleculeSample,
    result: ExpectedChargeResult,
    output: Path,
) -> dict[str, object]:
    return {
        "name": sample.name,
        "input_smiles": sample.smiles,
        "canonical_smiles": result.canonical_smiles,
        "image": str(output.resolve()),
        "formal_charge": result.formal_charge,
        "expected_electron_count": result.expected_electron_count,
        "integrated_nuclear_charge": result.integrated_nuclear_charge,
        "integrated_electron_count": result.integrated_electron_count,
        "integrated_charge": result.integrated_charge,
        "absolute_charge_error": abs(result.integrated_charge - result.formal_charge),
    }


def main() -> None:
    args = parse_args()
    samples = tuple(args.sample) if args.sample else DEFAULT_SAMPLES
    config = ExpectedChargeConfig(
        resolution=args.resolution,
        softsign_scale=args.softsign_scale,
    )
    renderer = ExpectedCharge2D(config, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, object]] = []
    for sample in samples:
        result = renderer.render(sample.smiles)
        output = args.output_dir / f"{safe_filename(sample.name)}.png"
        figure = render_figure(sample, result, config)
        figure.savefig(output, dpi=args.dpi)
        plt.close(figure)
        reports.append(sample_report(sample, result, output))
        print(
            f"Saved {output.resolve()} | Q={result.integrated_charge:.6g} "
            f"expected={result.formal_charge:+d}"
        )

    manifest = {
        "representation": "expected_valence_charge",
        "config": config.to_dict(),
        "samples": reports,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
