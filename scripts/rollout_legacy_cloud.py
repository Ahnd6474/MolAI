"""Render the committed previous-generation Cloud on a portable sample bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor


REPOSITORY = Path(__file__).resolve().parents[1]
LEGACY_SOURCE = REPOSITORY / "artifacts" / "legacy_epoch23_code" / "src"
sys.path.insert(0, str(LEGACY_SOURCE))

from molai.data.fields import collate_spectrum_field_batch  # noqa: E402
from molai.models.cloud import MolecularFieldCloud  # noqa: E402
from molai.models.condition import SpectrumConditionEncoder  # noqa: E402


LEVELS = (64, 56, 48, 40, 32, 28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=5)
    parser.add_argument("--seed", type=int, default=314159)
    return parser.parse_args()


def _correlation(first: Tensor, second: Tensor) -> float:
    first = first.float().flatten() - first.float().mean()
    second = second.float().flatten() - second.float().mean()
    denominator = first.square().sum().sqrt() * second.square().sum().sqrt()
    return float((first * second).sum() / denominator.clamp_min(1e-12))


def _normalize(image: Tensor) -> Tensor:
    image = image.detach().float().cpu().squeeze()
    scale = torch.quantile(image.abs().flatten(), 0.995).clamp_min(1e-6)
    return (image / scale).clamp(-1.0, 1.0)


def _build_models(config: dict, item: dict, checkpoint: dict) -> tuple[torch.nn.Module, torch.nn.Module]:
    model = config["model"]
    spectrum = config["spectrum_encoder"]
    condition_dim = int(model["condition_dim"])
    cloud = MolecularFieldCloud(
        condition_dim=condition_dim,
        field_channels=int(model["field_channels"]),
        dim=int(model["fullres_dim"]),
        heads=int(model["heads"]),
        condition_cross_depth=int(model["condition_cross_depth"]),
        noise_cross_depth=int(model["noise_cross_depth"]),
        noise_token_count=int(model["noise_token_count"]),
        noise_token_dim=int(model["noise_token_dim"]),
        noise_temperature=float(model["noise_temperature"]),
        noise_gate_init=float(model["noise_gate_init"]),
        refine_depth=int(model["refine_depth"]),
        window_size=int(model["window_size"]),
        ffn_ratio=float(model["ffn_ratio"]),
        cvt_kernel_sizes=model["cvt_kernel_sizes"],
        cvt_output_sizes=model["cvt_output_sizes"],
        condition_gate_init=float(model["condition_gate_init"]),
        max_resolution=int(model["max_resolution"]),
        noise_energy_min=float(model["noise_energy_min"]),
        noise_energy_init=float(model["noise_energy_init"]),
        noise_amplitude_max=float(model["noise_amplitude_max"]),
        zero_mean_output=bool(model.get("zero_mean_output", False)),
        gradient_checkpointing=False,
    )
    encoder = SpectrumConditionEncoder(
        metadata_dim=int(item["metadata"].shape[-1]),
        dim=condition_dim,
        heads=int(spectrum["heads"]),
        peak_conv_stages=int(spectrum["peak_conv_stages"]),
        peak_conv_kernel_size=int(spectrum["peak_conv_kernel_size"]),
        peak_tau_min=float(spectrum["peak_tau_min"]),
        peak_tau_max=float(spectrum["peak_tau_max"]),
        peak_cutoff_multiplier=float(spectrum["peak_cutoff_multiplier"]),
        spectrum_layers=int(spectrum["spectrum_layers"]),
        ffn_ratio=float(spectrum["ffn_ratio"]),
        dropout=float(spectrum["dropout"]),
        peak_position_dim=int(spectrum["peak_position_dim"]),
        mz_bin_width=float(spectrum["mz_bin_width"]),
        mz_upper_bound=float(spectrum["mz_upper_bound"]),
        peak_chunk_size=int(config["training"]["peak_chunk_size"]),
        peak_chunk_batch=int(spectrum["peak_chunk_batch"]),
    )
    cloud.load_state_dict(checkpoint["cloud"])
    encoder.load_state_dict(checkpoint["condition_encoder"])
    return cloud, encoder


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    bundle = torch.load(args.sample_bundle, map_location="cpu", weights_only=False)
    item = bundle["item"]
    config = checkpoint["config"]
    cloud, encoder = _build_models(config, item, checkpoint)
    cloud = cloud.to(device).eval()
    encoder = encoder.to(device).eval()

    batch = collate_spectrum_field_batch(
        [item], peak_chunk_size=int(config["training"]["peak_chunk_size"])
    )
    arguments = [
        batch[name].to(device)
        for name in (
            "peak_chunks",
            "peak_mask",
            "chunk_to_spectrum",
            "spectrum_to_molecule",
            "metadata",
            "precursor_mz",
        )
    ]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        condition = encoder.forward_ragged(*arguments, 1)

    clean = batch["field"].to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_noise = torch.randn(clean.shape, generator=generator, device=device)
    initial_noise -= initial_noise.mean(dim=(-2, -1), keepdim=True)
    noise_scale = float(config["cloud_matching"]["noise_schedule"]["noise_scale"])
    initial = noise_scale * initial_noise

    path_frames: list[list[Tensor]] = []
    for path in range(args.paths):
        current = initial.clone()
        frames = [current.cpu()]
        path_generator = torch.Generator(device=device).manual_seed(args.seed + path)
        for _level in LEVELS[:-1]:
            noise = torch.randn(
                1,
                1,
                cloud.noise_token_count,
                cloud.noise_token_dim,
                generator=path_generator,
                device=device,
            )
            if path == 0:
                noise.zero_()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted, _, _ = cloud(
                    current, condition, samples=1, noise=noise
                )
            current = predicted[:, 0]
            frames.append(current.float().cpu())
        path_frames.append(frames)

    clean_cpu = clean[0].cpu()
    metrics = []
    for path, frames in enumerate(path_frames):
        final = frames[-1][0]
        metrics.append(
            {
                "path": "zero" if path == 0 else f"random_{path}",
                "correlation": _correlation(final, clean_cpu),
                "mse": float((final - clean_cpu).square().mean()),
            }
        )

    columns = len(LEVELS) + 1
    figure, axes = plt.subplots(
        args.paths, columns, figsize=(1.45 * columns, 1.55 * args.paths), squeeze=False
    )
    for path, frames in enumerate(path_frames):
        for column, frame in enumerate(frames):
            axes[path, column].imshow(
                _normalize(frame[0]), cmap="coolwarm", vmin=-1, vmax=1
            )
            axes[path, column].axis("off")
            if path == 0:
                axes[path, column].set_title(f"L{LEVELS[column]}", fontsize=8)
        axes[path, -1].imshow(
            _normalize(clean_cpu), cmap="coolwarm", vmin=-1, vmax=1
        )
        axes[path, -1].axis("off")
        if path == 0:
            axes[path, -1].set_title("GT", fontsize=8)
        label = "Zero" if path == 0 else f"Random {path}"
        axes[path, 0].set_ylabel(f"{label}\nr={metrics[path]['correlation']:.3f}", fontsize=8)
    figure.suptitle(
        f"Previous generation Epoch {int(checkpoint['epoch']) + 1} | "
        f"{item['smiles']} | seed {args.seed}",
        fontsize=11,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    args.output.with_suffix(".json").write_text(
        json.dumps({"levels": LEVELS, "metrics": metrics}, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
