"""Compare unchanged eager/compiled two-step rollout forward + backward.

Synthetic MS tokens isolate rollout cost. This is not an end-to-end dataloader
throughput estimate. No optimizer update or checkpoint mutation is performed.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from train_absolute_cloud import _build_cloud
from train_absolute_hidden_rollout import _losses

from molai.models.cloud import AbsoluteTrajectoryRolloutCloud
from molai.models.losses import FullBandEnergyDistance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-loss", action="store_true")
    parser.add_argument("--mode", default="default")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(1729)
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    cloud = _build_cloud(config, device)
    cloud.gradient_checkpointing = False
    cloud.load_state_dict(checkpoint["cloud"])
    rollout = AbsoluteTrajectoryRolloutCloud(
        cloud, intermediate_refine_depth=6, use_level_conditioning=False
    ).to(device).train()
    batch = args.batch_size
    initial = torch.randn(batch, 1, 128, 128, device=device)
    condition = torch.randn(batch, args.tokens, cloud.dim, device=device, requires_grad=True)
    mask = torch.ones(batch, args.tokens, dtype=torch.bool, device=device)
    levels = torch.full((batch, 2), 32, device=device, dtype=torch.long)
    noise = torch.randn(batch, 2, 4, cloud.noise_token_count, cloud.noise_token_dim, device=device)
    target_clouds = torch.randn(batch, 2, 4, 1, 128, 128, device=device)
    target_means = target_clouds.mean(dim=2)
    target_variances = torch.ones(batch, 2, device=device)
    objective = FullBandEnergyDistance(levels=3, include_target_constant=False)

    def step(runner: nn.Module, loss_runner=_losses) -> tuple[torch.Tensor, torch.Tensor]:
        rollout.zero_grad(set_to_none=True)
        condition.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = runner(
                initial, condition, levels, samples=4, noise=noise,
                condition_mask=mask, compute_hidden_consistency=True,
                hidden_consistency_samples=1,
            )
            loss = loss_runner(
                output.fields, output.hidden_consistency_mse,
                target_means, target_variances, target_clouds, initial,
                objective, 0.0, 0.1, 3.5927782573874403,
            )["total"]
        loss.backward()
        return loss.detach(), output.fields.detach()

    reference = None
    variants = ["eager"]
    if args.compile:
        variants.append("compiled")
    if args.compile_loss:
        variants.append("compiled_loss")
    for label in variants:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        runner = rollout if label == "eager" else torch.compile(
            rollout, fullgraph=True, dynamic=True, mode=args.mode
        )
        loss_runner = (
            torch.compile(_losses, fullgraph=True, dynamic=False)
            if label == "compiled_loss" else _losses
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss, fields = step(runner, loss_runner)
        torch.cuda.synchronize()
        first_seconds = time.perf_counter() - start
        fields = fields.cpu().float()
        gradient = condition.grad.detach().cpu().float()
        if reference is None:
            reference = (fields, gradient)
        field_error = float((fields - reference[0]).abs().max())
        field_relative_error = float(
            (fields - reference[0]).norm() / reference[0].norm().clamp_min(1e-12)
        )
        grad_relative_error = float(
            (gradient - reference[1]).norm() / reference[1].norm().clamp_min(1e-12)
        )
        for _ in range(args.warmup):
            step(runner, loss_runner)
        torch.cuda.synchronize()
        timings = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            step(runner, loss_runner)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
        seconds = statistics.median(timings)
        print(json.dumps({
            "variant": label, "batch": batch, "ms_tokens": args.tokens,
            "first_seconds": first_seconds, "median_seconds": seconds,
            "molecules_per_second_per_gpu": batch / seconds,
            "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
            "loss": float(loss), "max_field_difference": field_error,
            "field_relative_rms_error": field_relative_error,
            "condition_gradient_relative_error": grad_relative_error,
        }), flush=True)


if __name__ == "__main__":
    main()
