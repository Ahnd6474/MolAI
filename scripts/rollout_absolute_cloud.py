"""Render repeated image-space rollouts from an absolute Cloud checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from train_absolute_cloud import (
    RaggedSpectrumTokenRunner,
    _build_cloud,
    _build_condition_encoder,
    _condition_tokens,
)
from train_cloud import _make_schedule, _transition_jumps

from molai.data import (
    ShardShuffleSampler,
    SpectrumFieldDataset,
    collate_spectrum_field_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path)
    source.add_argument("--sample-bundle", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=5)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.paths < 2:
        raise ValueError("paths must include one zero and at least one random path")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if args.sample_bundle is not None:
        bundle = torch.load(args.sample_bundle, map_location="cpu", weights_only=False)
        item = bundle["item"]
        sample_index = int(bundle["sample_index"])
        dataset = SimpleNamespace(manifest=bundle["manifest"])
    else:
        dataset = SpectrumFieldDataset(args.data)
        sampler = ShardShuffleSampler(
            dataset,
            split="validation",
            validation_fraction=float(config["training"]["validation_fraction"]),
            split_seed=int(dataset.manifest.get("seed", 0)) + 17,
            shuffle=False,
        )
        sample_index = next(iter(sampler))
        item = dataset[sample_index]
    batch = collate_spectrum_field_batch(
        [item],
        peak_chunk_size=int(config["training"].get("peak_chunk_size", 256)),
    )

    cloud = _build_cloud(config, device)
    condition_encoder = _build_condition_encoder(dataset, config, device)
    cloud.load_state_dict(checkpoint["cloud"])
    condition_encoder.load_state_dict(checkpoint["condition_encoder"])
    cloud.eval()
    condition_encoder.eval()
    condition_runner = RaggedSpectrumTokenRunner(condition_encoder)

    clean = batch["field"].to(device)
    schedule = _make_schedule(dataset.manifest, config, device)
    maximum_level = len(schedule.alpha_bar) - 1
    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_noise = torch.randn(
        clean.shape,
        generator=generator,
        device=device,
        dtype=clean.dtype,
    )
    initial_noise = initial_noise - initial_noise.mean(dim=(-2, -1), keepdim=True)
    initial = schedule.noise_scale * initial_noise

    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        condition, condition_mask = _condition_tokens(condition_runner, batch, device)
    # Run one trajectory at a time. This keeps the 32x32 CvT rollout usable on
    # smaller local GPUs without changing any model computation within a path.
    path_frames: list[list[Tensor]] = []
    levels: list[int] | None = None
    for path in range(args.paths):
        current = initial.clone()
        current_frames = [current.detach().cpu()]
        current_levels = [maximum_level]
        level = maximum_level
        path_generator = torch.Generator(device=device).manual_seed(args.seed + path)
        while level > 0:
            level_tensor = torch.full((1,), level, device=device, dtype=torch.long)
            noise = torch.randn(
                1,
                1,
                cloud.noise_token_count,
                cloud.noise_token_dim,
                generator=path_generator,
                device=device,
                dtype=current.dtype,
            )
            if path == 0:
                noise.zero_()
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = cloud(
                    current,
                    condition,
                    level_tensor,
                    samples=1,
                    noise=noise,
                    condition_mask=condition_mask,
                )
            current = output.fields[:, 0]
            jump = int(_transition_jumps(level_tensor, config).item())
            level = max(0, level - jump)
            current_levels.append(level)
            current_frames.append(current.detach().cpu())
        if levels is None:
            levels = current_levels
        elif current_levels != levels:
            raise RuntimeError("rollout paths produced different level schedules")
        path_frames.append(current_frames)
    assert levels is not None
    frames = [
        torch.cat([path_frames[path][step] for path in range(args.paths)], dim=0)
        for step in range(len(levels))
    ]

    clean_cpu = clean[0].cpu()
    metrics = []
    for path in range(args.paths):
        final = frames[-1][path]
        metrics.append(
            {
                "path": "zero" if path == 0 else f"random_{path}",
                "correlation": _correlation(final, clean_cpu),
                "mse": float((final.float() - clean_cpu.float()).square().mean()),
            }
        )

    columns = len(frames) + 1
    figure, axes = plt.subplots(
        args.paths,
        columns,
        figsize=(1.45 * columns, 1.55 * args.paths),
        squeeze=False,
    )
    for path in range(args.paths):
        for column, frame in enumerate(frames):
            axes[path, column].imshow(
                _normalize(frame[path]), cmap="coolwarm", vmin=-1, vmax=1
            )
            axes[path, column].axis("off")
            if path == 0:
                axes[path, column].set_title(f"L{levels[column]}", fontsize=8)
        axes[path, -1].imshow(
            _normalize(clean_cpu), cmap="coolwarm", vmin=-1, vmax=1
        )
        axes[path, -1].axis("off")
        if path == 0:
            axes[path, -1].set_title("GT", fontsize=8)
        label = "Zero" if path == 0 else f"Random {path}"
        axes[path, 0].set_ylabel(
            f"{label}\nr={metrics[path]['correlation']:.3f}",
            fontsize=8,
        )
    figure.suptitle(
        f"Epoch {int(checkpoint['epoch']) + 1} repeated rollout | "
        f"{item['smiles']} | seed {args.seed}",
        fontsize=11,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "sample_index": sample_index,
                "smiles": str(item["smiles"]),
                "levels": levels,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
