"""Train persistent random paths through a short hidden rollout.

Every decoded step receives an empirical U-statistic target and an absolute
ensemble-mean loss.  The model carries four independent random trajectories
without using level embeddings.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from train_absolute_cloud import (
    RaggedSpectrumTokenRunner,
    _build_cloud,
    _build_condition_encoder,
    _condition_tokens,
    _distributed_mean,
    _lr_factor,
    _postpool_peak_token_counts,
    _strip,
)
from train_cloud import _make_schedule, _transition_jumps

from molai.data import (
    ShardShuffleSampler,
    SpectrumFieldDataset,
    collate_spectrum_field_batch,
)
from molai.models.bridge import CosineVPSchedule
from molai.models.cloud import (
    AbsoluteMolecularFieldCloud,
    AbsoluteTrajectoryRolloutCloud,
)
from molai.models.condition import SpectrumTokenEncoder
from molai.models.losses import FullBandEnergyDistance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument(
        "--gradient-checkpointing",
        choices=("on", "off"),
        help="override absolute_model.gradient_checkpointing for throughput tests",
    )
    parser.add_argument(
        "--intermediate-refine-depth",
        type=int,
        help="override the number of CvT refine blocks in every rollout step",
    )
    parser.add_argument(
        "--compile-rollout",
        choices=("on", "off"),
        help="compile the complete training rollout before DDP wrapping",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        help="torch.compile mode for the complete rollout graph",
    )
    return parser.parse_args()


def _noise_inputs(
    cloud: AbsoluteMolecularFieldCloud,
    batch: int,
    steps: int,
    samples: int,
    reference: Tensor,
) -> Tensor:
    return torch.randn(
        batch,
        steps,
        samples,
        cloud.noise_token_count,
        cloud.noise_token_dim,
        device=reference.device,
        dtype=reference.dtype,
    )


def _level_path(start_levels: Tensor, steps: int, config: dict) -> tuple[Tensor, Tensor]:
    current = start_levels
    inputs = []
    targets = []
    for _ in range(steps):
        inputs.append(current)
        current = (current - _transition_jumps(current, config)).clamp_min(0)
        targets.append(current)
    return torch.stack(inputs, dim=1), torch.stack(targets, dim=1)


def _trajectory_targets(
    schedule: CosineVPSchedule,
    clean: Tensor,
    start_levels: Tensor,
    target_levels: Tensor,
    samples: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    initial = schedule.sample_training_batch(
        clean,
        samples=1,
        current_levels=start_levels,
        answer_jump=0,
    ).current
    means = []
    variances = []
    clouds = []
    for step in range(target_levels.shape[1]):
        step_levels = target_levels[:, step]
        mean, variance = schedule.target_mean_and_variance(
            clean, initial, start_levels, step_levels
        )
        means.append(mean)
        variances.append(variance)
        clouds.append(
            schedule.sample_target_cloud(
                clean, initial, start_levels, step_levels, samples
            )
        )
    return (
        initial,
        torch.stack(means, dim=1),
        torch.stack(variances, dim=1),
        torch.stack(clouds, dim=1),
    )


def _losses(
    output_fields: Tensor,
    hidden_consistency_mse: Tensor,
    target_means: Tensor,
    target_variances: Tensor,
    target_clouds: Tensor,
    initial: Tensor,
    loss_function: FullBandEnergyDistance,
    mean_weight: float,
    hidden_consistency_weight: float,
    noise_scale: float,
) -> dict[str, Tensor]:
    if hidden_consistency_mse.shape != output_fields.shape[:3]:
        raise ValueError("hidden consistency MSE must have shape [B,T,M]")
    step_totals = []
    step_u_statistics = []
    step_mean_mses = []
    step_normalized_mean_mses = []
    step_hidden_consistency_mses = []
    for step in range(output_fields.shape[1]):
        predicted = output_fields[:, step]
        target_mean = target_means[:, step]
        u_statistic = loss_function(
            predicted, target_clouds[:, step], initial
        )
        predicted_mean = predicted.float().mean(dim=1)
        squared_error = (predicted_mean - target_mean.float()).square().mean(
            dim=(1, 2, 3)
        )
        target_energy = target_mean.float().square().mean(dim=(1, 2, 3))
        target_energy = target_energy + noise_scale**2 * target_variances[:, step].float()
        normalized_mean_mse = (
            squared_error / target_energy.clamp_min(1e-4)
        ).mean()
        mean_mse = squared_error.mean()
        hidden_mse = hidden_consistency_mse[:, step].mean()
        step_u_statistics.append(u_statistic)
        step_mean_mses.append(mean_mse)
        step_normalized_mean_mses.append(normalized_mean_mse)
        step_hidden_consistency_mses.append(hidden_mse)
        step_totals.append(
            u_statistic
            + mean_weight * normalized_mean_mse
            + hidden_consistency_weight * hidden_mse
        )

    return {
        "total": torch.stack(step_totals).mean(),
        "u_statistic": torch.stack(step_u_statistics).mean(),
        "ensemble_mean_mse": torch.stack(step_mean_mses).mean(),
        "normalized_mean_mse": torch.stack(step_normalized_mean_mses).mean(),
        "hidden_consistency_mse": torch.stack(
            step_hidden_consistency_mses
        ).mean(),
        "diversity": output_fields.float().var(dim=2, unbiased=False).mean(),
        "step_total": torch.stack(step_totals),
        "step_u_statistic": torch.stack(step_u_statistics),
        "step_mean_mse": torch.stack(step_mean_mses),
        "step_hidden_consistency_mse": torch.stack(
            step_hidden_consistency_mses
        ),
    }


def _update_latest(path: Path, latest: Path) -> None:
    temporary = latest.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(path, temporary)
    except OSError:
        shutil.copyfile(path, temporary)
    os.replace(temporary, latest)


def _add_tensorboard_layout(writer: SummaryWriter, validation_steps: int) -> None:
    step_tags = lambda metric: [
        f"validation/step_{step}_{metric}"
        for step in range(1, validation_steps + 1)
    ]
    writer.add_custom_scalars(
        {
            "01 Loss": {
                "Batch objective": [
                    "Multiline",
                    [
                        "train/batch_total",
                        "train/batch_u_statistic",
                        "train/batch_normalized_mean_mse",
                        "train/batch_hidden_consistency_mse",
                    ],
                ],
                "Ensemble mean error": [
                    "Multiline",
                    [
                        "train/batch_ensemble_mean_mse",
                        "train/batch_normalized_mean_mse",
                    ],
                ],
                "Epoch train vs validation": [
                    "Multiline",
                    ["train/epoch_total", "validation/total"],
                ],
            },
            "02 Distribution": {
                "Diversity": [
                    "Multiline",
                    ["train/diversity", "validation/diversity"],
                ],
                "Field RMS": [
                    "Multiline",
                    ["model/output_rms", "validation/output_rms", "validation/clean_rms"],
                ],
                "Validation objective": [
                    "Multiline",
                    [
                        "validation/u_statistic",
                        "validation/normalized_mean_mse",
                        "validation/hidden_consistency_mse",
                    ],
                ],
            },
            "03 Hidden rollout validation": {
                "Total by step": ["Multiline", step_tags("total")],
                "U-stat by step": ["Multiline", step_tags("u_statistic")],
                "Mean MSE by step": ["Multiline", step_tags("mean_mse")],
                "Hidden consistency by step": [
                    "Multiline",
                    step_tags("hidden_consistency_mse"),
                ],
            },
            "04 Model dynamics": {
                "Residual dynamics": [
                    "Multiline",
                    ["model/gate_mean", "model/update_rms", "model/output_rms"],
                ],
            },
            "05 Optimization": {
                "Gradient norm": ["Multiline", ["train/grad_norm"]],
                "Learning rate": ["Multiline", ["train/learning_rate"]],
            },
        }
    )


def _save(
    output: Path,
    epoch: int,
    step: int,
    cloud: AbsoluteMolecularFieldCloud,
    condition_encoder: SpectrumTokenEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict,
    source_checkpoint: Path,
) -> Path:
    path = output / f"epoch-{epoch + 1:04d}.pt"
    temporary = output / f".{path.name}.tmp"
    torch.save(
        {
            "cloud": cloud.state_dict(),
            "condition_encoder": condition_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": step,
            "config": config,
            "source_checkpoint": str(source_checkpoint),
            "training_kind": "absolute-four-path-two-step-rollout-full-gradient-hidden-consistency",
        },
        temporary,
    )
    os.replace(temporary, path)
    _update_latest(path, output / "latest.pt")
    return path


@torch.no_grad()
def _validate(
    rollout: AbsoluteTrajectoryRolloutCloud,
    condition_runner: nn.Module,
    loader: DataLoader,
    schedule: CosineVPSchedule,
    steps: int,
    samples: int,
    loss_function: FullBandEnergyDistance,
    mean_weight: float,
    hidden_consistency_weight: float,
    config: dict,
    device: torch.device,
    seed: int,
    max_batches: int | None,
) -> tuple[dict[str, float], Tensor | None]:
    rollout.eval()
    condition_runner.eval()
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    totals = {
        "total": 0.0,
        "u_statistic": 0.0,
        "ensemble_mean_mse": 0.0,
        "normalized_mean_mse": 0.0,
        "hidden_consistency_mse": 0.0,
        "diversity": 0.0,
        "output_rms": 0.0,
        "clean_rms": 0.0,
    }
    step_totals = torch.zeros(steps, device=device, dtype=torch.float64)
    step_u_statistics = torch.zeros_like(step_totals)
    step_mean_mses = torch.zeros_like(step_totals)
    step_hidden_consistency_mses = torch.zeros_like(step_totals)
    count = 0
    preview = None
    try:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            clean = batch["field"].to(device, non_blocking=True)
            maximum = len(schedule.alpha_bar) - 1
            start_levels = torch.randint(
                1, maximum + 1, (len(clean),), device=device
            )
            input_levels, target_levels = _level_path(start_levels, steps, config)
            initial, target_means, target_variances, target_clouds = _trajectory_targets(
                schedule, clean, start_levels, target_levels, samples
            )
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                condition, condition_mask = _condition_tokens(
                    condition_runner, batch, device
                )
                noise = _noise_inputs(
                    rollout.cloud, len(clean), steps, samples, clean
                )
                output = rollout(
                    initial,
                    condition,
                    input_levels,
                    samples=samples,
                    noise=noise,
                    condition_mask=condition_mask,
                    compute_hidden_consistency=True,
                )
                if output.hidden_consistency_mse is None:
                    raise RuntimeError("hidden consistency MSE was not computed")
                losses = _losses(
                    output.fields,
                    output.hidden_consistency_mse,
                    target_means,
                    target_variances,
                    target_clouds,
                    initial,
                    loss_function,
                    mean_weight,
                    hidden_consistency_weight,
                    float(schedule.noise_scale),
                )
            size = len(clean)
            values = {
                "total": losses["total"],
                "u_statistic": losses["u_statistic"],
                "ensemble_mean_mse": losses["ensemble_mean_mse"],
                "normalized_mean_mse": losses["normalized_mean_mse"],
                "hidden_consistency_mse": losses["hidden_consistency_mse"],
                "diversity": losses["diversity"],
                "output_rms": output.fields.float().square().mean().sqrt(),
                "clean_rms": clean.float().square().mean().sqrt(),
            }
            for name, value in values.items():
                totals[name] += float(value) * size
            step_totals += losses["step_total"].double() * size
            step_u_statistics += losses["step_u_statistic"].double() * size
            step_mean_mses += losses["step_mean_mse"].double() * size
            step_hidden_consistency_mses += (
                losses["step_hidden_consistency_mse"].double() * size
            )
            count += size
            if preview is None:
                predicted_means = output.fields.float().mean(dim=2)
                preview = _strip(
                    clean[:4],
                    initial[:4],
                    *[predicted_means[:4, step] for step in range(steps)],
                    output.fields[:4, -1, 0],
                )
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
    metrics = {
        name: _distributed_mean(value, count, device) for name, value in totals.items()
    }
    for step in range(steps):
        metrics[f"step_{step + 1}_total"] = _distributed_mean(
            float(step_totals[step]), count, device
        )
        metrics[f"step_{step + 1}_u_statistic"] = _distributed_mean(
            float(step_u_statistics[step]), count, device
        )
        metrics[f"step_{step + 1}_mean_mse"] = _distributed_mean(
            float(step_mean_mses[step]), count, device
        )
        metrics[f"step_{step + 1}_hidden_consistency_mse"] = _distributed_mean(
            float(step_hidden_consistency_mses[step]), count, device
        )
    return metrics, preview


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", device_id=device)
    else:
        device = torch.device(args.device)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    options = config["absolute_hidden_rollout"]
    checkpointing = bool(options.get("gradient_checkpointing", True))
    if args.gradient_checkpointing is not None:
        checkpointing = args.gradient_checkpointing == "on"
    config["absolute_model"]["gradient_checkpointing"] = checkpointing
    batch_size = args.batch_size or int(options["batch_size_per_gpu"])
    validation_batch_size = int(options["validation_batch_size_per_gpu"])
    accumulation = args.accumulation_steps or int(options["gradient_accumulation_steps"])
    epochs = args.epochs or int(options["epochs"])
    steps = int(options["steps"])
    validation_steps = int(options.get("validation_steps", steps))
    samples = int(options["random_samples"])
    mean_weight = float(options["ensemble_mean_mse_weight"])
    hidden_consistency_weight = float(options["hidden_consistency_mse_weight"])
    if steps != 2:
        raise ValueError("this experiment requires exactly two training rollout steps")
    if validation_steps < steps:
        raise ValueError("validation_steps must be at least the training step count")
    if samples < 2:
        raise ValueError("U-statistic requires at least two final random samples")

    dataset = SpectrumFieldDataset(args.data)
    seed = int(dataset.manifest.get("seed", 0))
    torch.manual_seed(seed + rank)
    postpool_counts = _postpool_peak_token_counts(
        dataset,
        int(config["spectrum_encoder"]["peak_conv_stages"]),
        device,
        distributed,
        rank,
    )
    maximum_tokens = int(options["max_postpool_peak_tokens"])
    eligible_mask = postpool_counts <= maximum_tokens
    excluded_count = int((~eligible_mask).sum())

    source = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cloud = _build_cloud(config, device)
    condition_encoder = _build_condition_encoder(dataset, config, device)
    cloud.load_state_dict(source["cloud"])
    condition_encoder.load_state_dict(source["condition_encoder"])
    intermediate_refine_depth = (
        args.intermediate_refine_depth
        if args.intermediate_refine_depth is not None
        else int(options.get("intermediate_refine_depth", len(cloud.refine_blocks)))
    )
    rollout = AbsoluteTrajectoryRolloutCloud(
        cloud,
        intermediate_refine_depth=intermediate_refine_depth,
        use_level_conditioning=False,
    ).to(device)
    compile_rollout = bool(options.get("compile_rollout", False))
    if args.compile_rollout is not None:
        compile_rollout = args.compile_rollout == "on"
    compile_mode = args.compile_mode or str(options.get("compile_mode", "default"))
    rollout_execution: nn.Module = rollout
    if compile_rollout:
        if distributed:
            # The whole rollout is intentionally one graph.  DDPOptimizer's
            # bucket-based graph partitioning currently fails on the dynamic
            # MS-token dimension before Inductor code generation.
            torch._dynamo.config.optimize_ddp = False
        rollout_execution = torch.compile(
            rollout,
            fullgraph=True,
            dynamic=True,
            mode=compile_mode,
        )
    condition_module = RaggedSpectrumTokenRunner(condition_encoder)
    parameters = [*rollout.parameters(), *condition_encoder.parameters()]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(options["learning_rate"]),
        weight_decay=float(options["weight_decay"]),
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )

    peak_chunk_size = int(config["training"].get("peak_chunk_size", 256))
    collate = partial(collate_spectrum_field_batch, peak_chunk_size=peak_chunk_size)
    sampler_options = {
        "seed": seed,
        "rank": rank,
        "replicas": world_size,
        "batch_size": batch_size,
        "validation_fraction": float(config["training"]["validation_fraction"]),
        "split_seed": seed + 17,
        "eligible_mask": eligible_mask,
    }
    train_sampler = ShardShuffleSampler(
        dataset, split="train", shuffle=True, **sampler_options
    )
    validation_options = dict(sampler_options)
    validation_options["batch_size"] = validation_batch_size
    validation_sampler = ShardShuffleSampler(
        dataset, split="validation", shuffle=False, **validation_options
    )
    loader_options = {
        "collate_fn": collate,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        persistent_workers=args.workers > 0,
        **loader_options,
    )
    validation_loader = DataLoader(
        dataset,
        sampler=validation_sampler,
        batch_size=validation_batch_size,
        persistent_workers=False,
        **loader_options,
    )
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = max(1, epochs * updates_per_epoch)
    minimum_ratio = float(options["min_learning_rate"]) / float(
        options["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_factor(
            step, int(options["warmup_steps"]), total_updates, minimum_ratio
        ),
    )
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        resumed = torch.load(args.resume, map_location=device, weights_only=False)
        cloud.load_state_dict(resumed["cloud"])
        condition_encoder.load_state_dict(resumed["condition_encoder"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        start_epoch = int(resumed["epoch"]) + 1
        global_step = int(resumed["global_step"])

    schedule = _make_schedule(dataset.manifest, config, device)
    if not isinstance(schedule, CosineVPSchedule):
        raise TypeError("absolute hidden rollout requires a cosine VP schedule")
    loss_function = FullBandEnergyDistance(
        levels=int(config["cloud_matching"]["full_band_levels"]),
        include_target_constant=False,
    )
    rollout_runner: nn.Module = rollout_execution
    condition_runner: nn.Module = condition_module
    if distributed:
        rollout_runner = DistributedDataParallel(
            rollout_execution, device_ids=[local_rank], static_graph=True
        )
        condition_runner = DistributedDataParallel(
            condition_module, device_ids=[local_rank]
        )

    writer = None
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(args.output / "tensorboard")
        _add_tensorboard_layout(writer, validation_steps)
        writer.add_text("run/config", f"```yaml\n{yaml.safe_dump(config, sort_keys=False)}```")
        (args.output / "architecture.json").write_text(
            json.dumps(
                {
                    "source_checkpoint": str(args.checkpoint),
                    "source_epoch": int(source["epoch"]) + 1,
                    "rollout_steps": steps,
                    "validation_rollout_steps": validation_steps,
                    "persistent_random_paths": samples,
                    "ensemble_mean_mse_weight": mean_weight,
                    "hidden_consistency_mse_weight": hidden_consistency_weight,
                    "intermediate_refine_depth": intermediate_refine_depth,
                    "use_level_conditioning": False,
                    "compile_rollout": compile_rollout,
                    "compile_mode": compile_mode,
                    "cloud_parameters": sum(value.numel() for value in cloud.parameters()),
                    "condition_parameters": sum(
                        value.numel() for value in condition_encoder.parameters()
                    ),
                    "effective_global_batch": batch_size * world_size * accumulation,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"train={train_sampler.selected_count:,} "
            f"validation={validation_sampler.selected_count:,} "
            f"batch/GPU={batch_size} accumulation={accumulation} "
            f"effective_batch={batch_size * world_size * accumulation} "
            f"steps={steps} validation_steps={validation_steps} "
            f"persistent_paths={samples} "
            f"excluded_long_ms={excluded_count:,}",
            flush=True,
        )

    stop = False
    log_every = 1 if args.max_steps is not None else int(config["training"].get("log_every", 10))
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        rollout_execution.train()
        condition_module.train()
        totals = {
            "total": 0.0,
            "u_statistic": 0.0,
            "ensemble_mean_mse": 0.0,
            "normalized_mean_mse": 0.0,
            "hidden_consistency_mse": 0.0,
        }
        count = 0
        for micro_step, batch in enumerate(train_loader):
            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break
            started = time.perf_counter()
            clean = batch["field"].to(device, non_blocking=True)
            maximum = len(schedule.alpha_bar) - 1
            start_levels = torch.randint(
                1, maximum + 1, (len(clean),), device=device
            )
            input_levels, target_levels = _level_path(start_levels, steps, config)
            initial, target_means, target_variances, target_clouds = _trajectory_targets(
                schedule, clean, start_levels, target_levels, samples
            )
            noise = _noise_inputs(cloud, len(clean), steps, samples, clean)
            should_step = (
                (micro_step + 1) % accumulation == 0
                or micro_step + 1 == len(train_loader)
            )
            with ExitStack() as synchronization:
                if distributed and not should_step:
                    synchronization.enter_context(rollout_runner.no_sync())  # type: ignore[attr-defined]
                    synchronization.enter_context(condition_runner.no_sync())  # type: ignore[attr-defined]
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    condition, condition_mask = _condition_tokens(
                        condition_runner, batch, device
                    )
                    output = rollout_runner(
                        initial,
                        condition,
                        input_levels,
                        samples=samples,
                        noise=noise,
                        condition_mask=condition_mask,
                        compute_hidden_consistency=True,
                    )
                    if output.hidden_consistency_mse is None:
                        raise RuntimeError("hidden consistency MSE was not computed")
                    losses = _losses(
                        output.fields,
                        output.hidden_consistency_mse,
                        target_means,
                        target_variances,
                        target_clouds,
                        initial,
                        loss_function,
                        mean_weight,
                        hidden_consistency_weight,
                        float(schedule.noise_scale),
                    )
                (losses["total"] / accumulation).backward()
            size = len(clean)
            for name in totals:
                totals[name] += float(losses[name].detach()) * size
            count += size
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0 and global_step % log_every == 0:
                    assert writer is not None
                    seconds = time.perf_counter() - started
                    writer.add_scalar(
                        "train/batch_total", float(losses["total"].detach()), global_step
                    )
                    writer.add_scalar(
                        "train/batch_u_statistic",
                        float(losses["u_statistic"].detach()),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/batch_ensemble_mean_mse",
                        float(losses["ensemble_mean_mse"].detach()),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/batch_normalized_mean_mse",
                        float(losses["normalized_mean_mse"].detach()),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/batch_hidden_consistency_mse",
                        float(losses["hidden_consistency_mse"].detach()),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/diversity", float(losses["diversity"].detach()), global_step
                    )
                    writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                    writer.add_scalar(
                        "train/learning_rate", scheduler.get_last_lr()[0], global_step
                    )
                    writer.add_scalar(
                        "model/gate_mean",
                        float(output.gate_means.detach().mean()),
                        global_step,
                    )
                    writer.add_scalar(
                        "model/update_rms",
                        float(output.update_rms.detach().mean()),
                        global_step,
                    )
                    writer.add_scalar(
                        "model/output_rms",
                        float(output.fields.detach().float().square().mean().sqrt()),
                        global_step,
                    )
                    memory = (
                        torch.cuda.max_memory_allocated(device) / 2**30
                        if device.type == "cuda"
                        else 0.0
                    )
                    print(
                        f"epoch={epoch + 1}/{epochs} step={global_step} "
                        f"loss={float(losses['total'].detach()):.5f} "
                        f"ustat={float(losses['u_statistic'].detach()):.5f} "
                        f"mean_mse={float(losses['ensemble_mean_mse'].detach()):.5f} "
                        f"norm_mean={float(losses['normalized_mean_mse'].detach()):.5f} "
                        f"hidden_mse={float(losses['hidden_consistency_mse'].detach()):.5f} "
                        f"diversity={float(losses['diversity'].detach()):.5f} "
                        f"grad={float(grad_norm):.3f} "
                        f"micro_speed={size * world_size / seconds:.1f}mol/s "
                        f"peak_mem={memory:.2f}GiB",
                        flush=True,
                    )

        train_means = {
            name: _distributed_mean(value, count, device)
            for name, value in totals.items()
        }
        validation, preview = _validate(
            rollout,
            condition_module,
            validation_loader,
            schedule,
            validation_steps,
            samples,
            loss_function,
            mean_weight,
            hidden_consistency_weight,
            config,
            device,
            seed + 100_003 + rank,
            args.max_validation_batches,
        )
        if rank == 0:
            assert writer is not None
            for name, value in train_means.items():
                writer.add_scalar(f"train/epoch_{name}", value, epoch + 1)
            for name, value in validation.items():
                writer.add_scalar(f"validation/{name}", value, epoch + 1)
            if preview is not None:
                writer.add_images(
                    "samples/clean_input_step_means_final_random",
                    preview,
                    epoch + 1,
                    dataformats="NCHW",
                )
            checkpoint_path = _save(
                args.output,
                epoch,
                global_step,
                cloud,
                condition_encoder,
                optimizer,
                scheduler,
                config,
                args.checkpoint,
            )
            writer.flush()
            print(
                f"epoch={epoch + 1} train={train_means['total']:.5f} "
                f"validation={validation['total']:.5f} saved={checkpoint_path}",
                flush=True,
            )
        if distributed:
            dist.barrier()
        if stop:
            break

    if writer is not None:
        writer.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
