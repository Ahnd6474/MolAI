"""Fine-tune the absolute Cloud as an eight-step hidden residual rollout.

The input field and MS spectra are encoded once.  Seven shared steps follow a
single stochastic hidden path.  The final shared step branches into one
zero-token output and four random outputs; only those final decoded fields
receive the U-statistic and clean zero-token losses.
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
from torch.nn import functional as F
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
from train_cloud import _make_schedule

from molai.data import (
    ShardShuffleSampler,
    SpectrumFieldDataset,
    collate_spectrum_field_batch,
)
from molai.models.bridge import CosineVPSchedule
from molai.models.cloud import AbsoluteHiddenRolloutCloud, AbsoluteMolecularFieldCloud
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
        help="override the number of CvT refine blocks in the first seven steps",
    )
    parser.add_argument(
        "--compile-rollout",
        choices=("on", "off"),
        help="compile the complete eight-step rollout before DDP wrapping",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        help="torch.compile mode for the complete rollout graph",
    )
    return parser.parse_args()


def _initial_noise(clean: Tensor, noise_scale: float) -> Tensor:
    noise = torch.randn_like(clean)
    noise = noise - noise.float().mean(dim=(-2, -1), keepdim=True).to(noise.dtype)
    return noise_scale * noise


def _noise_inputs(
    cloud: AbsoluteMolecularFieldCloud,
    batch: int,
    steps: int,
    final_samples: int,
    reference: Tensor,
) -> tuple[Tensor, Tensor]:
    intermediate = torch.randn(
        batch,
        steps - 1,
        cloud.noise_token_count,
        cloud.noise_token_dim,
        device=reference.device,
        dtype=reference.dtype,
    )
    final = torch.randn(
        batch,
        final_samples,
        cloud.noise_token_count,
        cloud.noise_token_dim,
        device=reference.device,
        dtype=reference.dtype,
    )
    final[:, 0].zero_()
    return intermediate, final


def _losses(
    output_fields: Tensor,
    clean: Tensor,
    initial: Tensor,
    loss_function: FullBandEnergyDistance,
    zero_weight: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    zero = output_fields[:, 0]
    random = output_fields[:, 1:]
    target = clean[:, None].expand_as(random)
    u_statistic = loss_function(random, target, initial)
    zero_mse = F.mse_loss(zero.float(), clean.float())
    loss = u_statistic + zero_weight * zero_mse
    diversity = random.float().var(dim=1, unbiased=False).mean()
    return loss, u_statistic, zero_mse, diversity


def _update_latest(path: Path, latest: Path) -> None:
    temporary = latest.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(path, temporary)
    except OSError:
        shutil.copyfile(path, temporary)
    os.replace(temporary, latest)


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
            "training_kind": "absolute-hidden-final-branch-rollout",
        },
        temporary,
    )
    os.replace(temporary, path)
    _update_latest(path, output / "latest.pt")
    return path


@torch.no_grad()
def _validate(
    rollout: AbsoluteHiddenRolloutCloud,
    condition_runner: nn.Module,
    loader: DataLoader,
    levels: Tensor,
    noise_scale: float,
    final_samples: int,
    loss_function: FullBandEnergyDistance,
    zero_weight: float,
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
        "zero_token_mse": 0.0,
        "final_diversity": 0.0,
        "output_rms": 0.0,
        "clean_rms": 0.0,
    }
    count = 0
    preview = None
    try:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            clean = batch["field"].to(device, non_blocking=True)
            initial = _initial_noise(clean, noise_scale)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                condition, condition_mask = _condition_tokens(
                    condition_runner, batch, device
                )
                intermediate_noise, final_noise = _noise_inputs(
                    rollout.cloud, len(clean), len(levels), final_samples, clean
                )
                output = rollout(
                    initial,
                    condition,
                    levels,
                    final_samples=final_samples,
                    intermediate_noise=intermediate_noise,
                    final_noise=final_noise,
                    condition_mask=condition_mask,
                )
                loss, u_statistic, zero_mse, diversity = _losses(
                    output.fields, clean, initial, loss_function, zero_weight
                )
            size = len(clean)
            values = {
                "total": loss,
                "u_statistic": u_statistic,
                "zero_token_mse": zero_mse,
                "final_diversity": diversity,
                "output_rms": output.fields.float().square().mean().sqrt(),
                "clean_rms": clean.float().square().mean().sqrt(),
            }
            for name, value in values.items():
                totals[name] += float(value) * size
            count += size
            if preview is None:
                preview = _strip(
                    clean[:4], initial[:4], output.fields[:4, 0], output.fields[:4, 1]
                )
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
    metrics = {
        name: _distributed_mean(value, count, device) for name, value in totals.items()
    }
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
    levels = torch.tensor([int(value) for value in options["levels"]], device=device)
    random_samples = int(options["final_random_samples"])
    final_samples = random_samples + 1
    zero_weight = float(options["zero_token_mse_weight"])
    if len(levels) != 8:
        raise ValueError("absolute hidden rollout requires exactly eight levels")
    if random_samples < 2:
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
    rollout = AbsoluteHiddenRolloutCloud(
        cloud, intermediate_refine_depth=intermediate_refine_depth
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
        writer.add_text("run/config", f"```yaml\n{yaml.safe_dump(config, sort_keys=False)}```")
        (args.output / "architecture.json").write_text(
            json.dumps(
                {
                    "source_checkpoint": str(args.checkpoint),
                    "source_epoch": int(source["epoch"]) + 1,
                    "rollout_levels": levels.tolist(),
                    "intermediate_samples": 1,
                    "final_zero_samples": 1,
                    "final_random_samples": random_samples,
                    "intermediate_refine_depth": intermediate_refine_depth,
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
            f"levels={levels.tolist()} final_samples=1+{random_samples} "
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
        totals = {"total": 0.0, "u_statistic": 0.0, "zero_token_mse": 0.0}
        count = 0
        for micro_step, batch in enumerate(train_loader):
            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break
            started = time.perf_counter()
            clean = batch["field"].to(device, non_blocking=True)
            initial = _initial_noise(clean, float(schedule.noise_scale))
            intermediate_noise, final_noise = _noise_inputs(
                cloud, len(clean), len(levels), final_samples, clean
            )
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
                        levels,
                        final_samples=final_samples,
                        intermediate_noise=intermediate_noise,
                        final_noise=final_noise,
                        condition_mask=condition_mask,
                    )
                    loss, u_statistic, zero_mse, diversity = _losses(
                        output.fields, clean, initial, loss_function, zero_weight
                    )
                (loss / accumulation).backward()
            size = len(clean)
            for name, value in (
                ("total", loss),
                ("u_statistic", u_statistic),
                ("zero_token_mse", zero_mse),
            ):
                totals[name] += float(value.detach()) * size
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
                    writer.add_scalar("train/batch_total", float(loss.detach()), global_step)
                    writer.add_scalar(
                        "train/batch_u_statistic", float(u_statistic.detach()), global_step
                    )
                    writer.add_scalar(
                        "train/batch_zero_token_mse", float(zero_mse.detach()), global_step
                    )
                    writer.add_scalar(
                        "train/final_diversity", float(diversity.detach()), global_step
                    )
                    writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                    writer.add_scalar(
                        "train/learning_rate", scheduler.get_last_lr()[0], global_step
                    )
                    writer.add_scalar(
                        "model/intermediate_gate_mean",
                        float(output.intermediate_gate_means.detach().mean()),
                        global_step,
                    )
                    writer.add_scalar(
                        "model/intermediate_update_rms",
                        float(output.intermediate_update_rms.detach().mean()),
                        global_step,
                    )
                    writer.add_scalar(
                        "model/final_gate_mean",
                        float(output.final_gate_means.detach().mean()),
                        global_step,
                    )
                    writer.add_scalar(
                        "model/final_update_rms",
                        float(output.final_update_rms.detach().mean()),
                        global_step,
                    )
                    writer.add_scalar(
                        "model/final_output_rms",
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
                        f"loss={float(loss.detach()):.5f} "
                        f"ustat={float(u_statistic.detach()):.5f} "
                        f"zero_mse={float(zero_mse.detach()):.5f} "
                        f"diversity={float(diversity.detach()):.5f} "
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
            levels,
            float(schedule.noise_scale),
            final_samples,
            loss_function,
            zero_weight,
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
                    "samples/clean_noise_zero_random",
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
