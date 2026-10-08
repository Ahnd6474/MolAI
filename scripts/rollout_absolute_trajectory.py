"""Render persistent hidden-space trajectories from an absolute rollout checkpoint."""

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

from molai.data import collate_spectrum_field_batch
from molai.models.cloud import AbsoluteTrajectoryRolloutCloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--paths", type=int, default=4)
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


def _level_path(maximum: int, steps: int, config: dict, device: torch.device) -> list[int]:
    levels = [maximum]
    current = maximum
    for _ in range(steps):
        level_tensor = torch.tensor([current], device=device, dtype=torch.long)
        jump = int(_transition_jumps(level_tensor, config).item())
        current = max(0, current - jump)
        levels.append(current)
    return levels


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.paths < 2:
        raise ValueError("steps must be positive and paths must be at least two")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    bundle = torch.load(args.sample_bundle, map_location="cpu", weights_only=False)
    item = bundle["item"]
    sample_index = int(bundle["sample_index"])
    dataset = SimpleNamespace(manifest=bundle["manifest"])
    config = checkpoint["config"]
    batch = collate_spectrum_field_batch(
        [item], peak_chunk_size=int(config["training"].get("peak_chunk_size", 256))
    )

    cloud = _build_cloud(config, device)
    condition_encoder = _build_condition_encoder(dataset, config, device)
    cloud.load_state_dict(checkpoint["cloud"])
    condition_encoder.load_state_dict(checkpoint["condition_encoder"])
    rollout = AbsoluteTrajectoryRolloutCloud(
        cloud,
        intermediate_refine_depth=int(
            config["absolute_hidden_rollout"].get(
                "intermediate_refine_depth", len(cloud.refine_blocks)
            )
        ),
        use_level_conditioning=False,
    ).to(device)
    rollout.eval()
    condition_encoder.eval()
    condition_runner = RaggedSpectrumTokenRunner(condition_encoder)

    clean = batch["field"].to(device)
    schedule = _make_schedule(dataset.manifest, config, device)
    maximum_level = len(schedule.alpha_bar) - 1
    levels = _level_path(maximum_level, args.steps, config, device)
    input_levels = torch.tensor(levels[:-1], device=device, dtype=torch.long)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial = torch.randn(
        clean.shape, generator=generator, device=device, dtype=clean.dtype
    )
    initial = initial - initial.mean(dim=(-2, -1), keepdim=True)
    initial = schedule.noise_scale * initial
    noise = torch.randn(
        1,
        args.steps,
        args.paths,
        cloud.noise_token_count,
        cloud.noise_token_dim,
        generator=generator,
        device=device,
        dtype=clean.dtype,
    )

    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        condition, condition_mask = _condition_tokens(condition_runner, batch, device)
        output = rollout(
            initial,
            condition,
            input_levels,
            samples=args.paths,
            noise=noise,
            condition_mask=condition_mask,
        )

    fields = output.fields[0].detach().float().cpu()
    initial_cpu = initial[0].detach().float().cpu()
    clean_cpu = clean[0].detach().float().cpu()
    means = fields.mean(dim=1)
    rows = [torch.cat([initial_cpu[None], means], dim=0)]
    rows.extend(
        torch.cat([initial_cpu[None], fields[:, path]], dim=0)
        for path in range(args.paths)
    )

    metrics: list[dict[str, float | int | str]] = []
    for row_index, row in enumerate(rows):
        path_name = "ensemble_mean" if row_index == 0 else f"random_{row_index}"
        for step, frame in enumerate(row):
            metrics.append(
                {
                    "path": path_name,
                    "step": step,
                    "level": levels[step],
                    "correlation": _correlation(frame, clean_cpu),
                    "mse": float((frame - clean_cpu).square().mean()),
                    "rms": float(frame.square().mean().sqrt()),
                }
            )

    columns = args.steps + 2
    figure, axes = plt.subplots(
        len(rows),
        columns,
        figsize=(1.35 * columns, 1.55 * len(rows)),
        squeeze=False,
    )
    for row_index, row in enumerate(rows):
        for column, frame in enumerate(row):
            axes[row_index, column].imshow(
                _normalize(frame), cmap="coolwarm", vmin=-1, vmax=1
            )
            axes[row_index, column].axis("off")
            if row_index == 0:
                axes[row_index, column].set_title(
                    f"S{column}\nL{levels[column]}", fontsize=7
                )
        axes[row_index, -1].imshow(
            _normalize(clean_cpu), cmap="coolwarm", vmin=-1, vmax=1
        )
        axes[row_index, -1].axis("off")
        final_metric = metrics[row_index * len(row) + len(row) - 1]
        label = "Mean" if row_index == 0 else f"Random {row_index}"
        axes[row_index, 0].set_ylabel(
            f"{label}\nr={final_metric['correlation']:.3f}", fontsize=8
        )
        if row_index == 0:
            axes[row_index, -1].set_title("GT", fontsize=8)
    figure.suptitle(
        f"Epoch {int(checkpoint['epoch']) + 1} persistent hidden rollout | "
        f"{item['smiles']} | seed {args.seed}",
        fontsize=11,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=130, bbox_inches="tight")
    plt.close(figure)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "sample_index": sample_index,
                "smiles": str(item["smiles"]),
                "seed": args.seed,
                "levels": levels,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    final_metrics = [entry for entry in metrics if entry["step"] == args.steps]
    print(json.dumps(final_metrics), flush=True)


if __name__ == "__main__":
    main()
