"""Train the MS-token-conditioned absolute Cloud from scratch."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
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
from train_cloud import _make_schedule, _sample_transition

from molai.data import ShardShuffleSampler, SpectrumFieldDataset, collate_spectrum_field_batch
from molai.models.bridge import CosineVPSchedule
from molai.models.cloud import AbsoluteMolecularFieldCloud
from molai.models.condition import SpectrumTokenEncoder
from molai.models.losses import FullBandEnergyDistance


class RaggedSpectrumTokenRunner(nn.Module):
    def __init__(self, encoder: SpectrumTokenEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        peak_chunks: Tensor,
        peak_mask: Tensor,
        chunk_to_spectrum: Tensor,
        spectrum_to_molecule: Tensor,
        metadata: Tensor,
        precursor_mz: Tensor,
        molecule_count: int,
    ) -> tuple[Tensor, Tensor]:
        return self.encoder.forward_ragged_tokens(
            peak_chunks,
            peak_mask,
            chunk_to_spectrum,
            spectrum_to_molecule,
            metadata,
            precursor_mz,
            molecule_count,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
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
    return parser.parse_args()


def _condition_tokens(
    runner: nn.Module, batch: dict[str, Tensor | list[str]], device: torch.device
) -> tuple[Tensor, Tensor]:
    arguments = [
        batch[name].to(device, non_blocking=True)
        for name in (
            "peak_chunks",
            "peak_mask",
            "chunk_to_spectrum",
            "spectrum_to_molecule",
            "metadata",
            "precursor_mz",
        )
    ]
    return runner(*arguments, int(batch["field"].shape[0]))


def _build_condition_encoder(
    dataset: SpectrumFieldDataset, config: dict, device: torch.device
) -> SpectrumTokenEncoder:
    model = config["absolute_model"]
    options = config["spectrum_encoder"]
    return SpectrumTokenEncoder(
        metadata_dim=int(dataset.manifest["metadata_dim"]),
        dim=int(model["condition_dim"]),
        heads=int(options["heads"]),
        peak_conv_stages=int(options["peak_conv_stages"]),
        peak_conv_kernel_size=int(options["peak_conv_kernel_size"]),
        peak_tau_min=float(options["peak_tau_min"]),
        peak_tau_max=float(options["peak_tau_max"]),
        peak_cutoff_multiplier=float(options["peak_cutoff_multiplier"]),
        spectrum_layers=int(options["spectrum_layers"]),
        ffn_ratio=float(options["ffn_ratio"]),
        dropout=float(options["dropout"]),
        peak_position_dim=int(options["peak_position_dim"]),
        mz_bin_width=float(options["mz_bin_width"]),
        mz_upper_bound=float(options["mz_upper_bound"]),
        peak_chunk_size=int(config["training"].get("peak_chunk_size", 256)),
        peak_chunk_batch=int(options.get("peak_chunk_batch", 256)),
    ).to(device)


def _build_cloud(config: dict, device: torch.device) -> AbsoluteMolecularFieldCloud:
    model = config["absolute_model"]
    return AbsoluteMolecularFieldCloud(
        field_channels=int(model["field_channels"]),
        condition_dim=int(model["condition_dim"]),
        dim=int(model["image_dim"]),
        heads=int(model["heads"]),
        condition_cross_depth=int(model["condition_cross_depth"]),
        condition_gate_init=float(model["condition_gate_init"]),
        noise_token_count=int(model["noise_token_count"]),
        noise_token_dim=int(model["noise_token_dim"]),
        noise_temperature=float(model["noise_temperature"]),
        noise_gate_init=float(model["noise_gate_init"]),
        refine_depth=int(model["refine_depth"]),
        ffn_ratio=float(model["ffn_ratio"]),
        cvt_kernel_sizes=[int(value) for value in model["cvt_kernel_sizes"]],
        cvt_grid_sizes=[int(value) for value in model["cvt_grid_sizes"]],
        max_level=int(model["max_level"]),
        anchor_gate_init=float(model["anchor_gate_init"]),
        decoder_dim=int(model["decoder_dim"]),
        noise_energy_min=float(model["noise_energy_min"]),
        noise_energy_init=float(model["noise_energy_init"]),
        noise_amplitude_max=float(model["noise_amplitude_max"]),
        zero_mean_output=bool(model["zero_mean_output"]),
        gradient_checkpointing=bool(model["gradient_checkpointing"]),
    ).to(device)


def _postpool_peak_token_counts(
    dataset: SpectrumFieldDataset,
    pooling_stages: int,
    device: torch.device,
    distributed: bool,
    rank: int,
) -> Tensor:
    """Count retained peak tokens per molecule without loading peak values."""

    counts = torch.empty(len(dataset), dtype=torch.long)
    if rank == 0:
        reduction = 2**pooling_stages
        start = 0
        for shard in dataset.shards:
            payload = torch.load(
                dataset.root / str(shard["file"]),
                map_location="cpu",
                weights_only=False,
            )
            peak_offsets = payload["peak_offsets"].long()
            peak_counts = peak_offsets[1:] - peak_offsets[:-1]
            spectrum_tokens = (peak_counts + reduction - 1) // reduction
            prefix = torch.cat(
                (torch.zeros(1, dtype=torch.long), spectrum_tokens.cumsum(0))
            )
            spectrum_offsets = payload["spectrum_offsets"].long()
            shard_counts = (
                prefix[spectrum_offsets[1:]] - prefix[spectrum_offsets[:-1]]
            )
            counts[start : start + len(shard_counts)] = shard_counts
            start += len(shard_counts)
    if distributed:
        transferred = counts.to(device)
        dist.broadcast(transferred, src=0)
        counts = transferred.cpu()
    return counts


def _strip(*images: Tensor) -> Tensor:
    panels = [image[:, :1].detach().float().cpu() for image in images]
    output = []
    for row in zip(*panels, strict=True):
        scale = torch.quantile(torch.cat([value.flatten() for value in row]).abs(), 0.995)
        scale = scale.clamp_min(1e-6)
        normalized = [((value / scale).clamp(-1, 1) + 1) * 0.5 for value in row]
        separator = torch.ones(1, row[0].shape[-2], 3)
        pieces = []
        for index, value in enumerate(normalized):
            if index:
                pieces.append(separator)
            pieces.append(value)
        output.append(torch.cat(pieces, dim=-1))
    return torch.stack(output)


def _distributed_mean(total: float, count: int, device: torch.device) -> float:
    values = torch.tensor([total, count], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(values)
    return float(values[0] / values[1].clamp_min(1))


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
        },
        temporary,
    )
    os.replace(temporary, path)
    _update_latest(path, output / "latest.pt")
    return path


def _lr_factor(step: int, warmup: int, total: int, minimum_ratio: float) -> float:
    if step < warmup:
        return max(1, step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def _validate(
    cloud: AbsoluteMolecularFieldCloud,
    condition_runner: nn.Module,
    loader: DataLoader,
    schedule: CosineVPSchedule,
    loss_function: FullBandEnergyDistance,
    config: dict,
    device: torch.device,
    samples: int,
    zero_weight: float,
    max_batches: int | None,
) -> tuple[dict[str, float], Tensor | None]:
    cloud.eval()
    condition_runner.eval()
    totals = {
        "total": 0.0,
        "u_statistic": 0.0,
        "zero_token_mse": 0.0,
        "predicted_variance": 0.0,
        "target_variance": 0.0,
    }
    count = 0
    preview = None
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        clean = batch["field"].to(device, non_blocking=True)
        transition = _sample_transition(schedule, clean, samples, config)
        noise = torch.randn(
            clean.shape[0],
            samples + 1,
            cloud.noise_token_count,
            cloud.noise_token_dim,
            device=device,
            dtype=clean.dtype,
        )
        noise[:, 0].zero_()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            condition, condition_mask = _condition_tokens(
                condition_runner, batch, device
            )
            output = cloud(
                transition.current,
                condition,
                transition.current_levels,
                samples=samples + 1,
                noise=noise,
                condition_mask=condition_mask,
            )
            zero, random = output.fields[:, 0], output.fields[:, 1:]
            u_statistic = loss_function(random, transition.target_cloud, transition.current)
            target_mean = schedule.target_mean_and_variance(
                clean,
                transition.current,
                transition.current_levels,
                transition.answer_levels,
            )[0]
            zero_loss = F.mse_loss(
                (zero - transition.current).float(),
                (target_mean - transition.current).float(),
            )
            loss = u_statistic + zero_weight * zero_loss
            predicted_variance = random.float().var(dim=1, unbiased=False).mean()
            target_variance = transition.target_cloud.float().var(
                dim=1, unbiased=False
            ).mean()
        size = clean.shape[0]
        for name, value in (
            ("total", loss),
            ("u_statistic", u_statistic),
            ("zero_token_mse", zero_loss),
            ("predicted_variance", predicted_variance),
            ("target_variance", target_variance),
        ):
            totals[name] += float(value) * size
        count += size
        if preview is None:
            preview = _strip(
                clean[:4], transition.current[:4], zero[:4], random[:4, 0]
            )
    metrics = {
        name: _distributed_mean(value, count, device) for name, value in totals.items()
    }
    metrics["diversity_ratio"] = metrics["predicted_variance"] / max(
        metrics["target_variance"], 1e-12
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
    options = config["absolute_training"]
    batch_size = args.batch_size or int(options["batch_size_per_gpu"])
    validation_batch_size = int(options["validation_batch_size_per_gpu"])
    accumulation = args.accumulation_steps or int(options["gradient_accumulation_steps"])
    epochs = args.epochs or int(options["epochs"])
    samples = int(options["samples"])
    zero_weight = float(options["zero_token_mse_weight"])

    dataset = SpectrumFieldDataset(args.data)
    seed = int(dataset.manifest.get("seed", 0))
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
    torch.manual_seed(seed + rank)
    cloud = _build_cloud(config, device)
    condition_encoder = _build_condition_encoder(dataset, config, device)
    condition_module = RaggedSpectrumTokenRunner(condition_encoder)
    parameters = [*cloud.parameters(), *condition_encoder.parameters()]
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
    train_sampler = ShardShuffleSampler(dataset, split="train", shuffle=True, **sampler_options)
    validation_sampler_options = dict(sampler_options)
    validation_sampler_options["batch_size"] = validation_batch_size
    validation_sampler = ShardShuffleSampler(
        dataset,
        split="validation",
        shuffle=False,
        **validation_sampler_options,
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
    minimum_ratio = float(options["min_learning_rate"]) / float(options["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_factor(step, int(options["warmup_steps"]), total_updates, minimum_ratio),
    )
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        cloud.load_state_dict(checkpoint["cloud"])
        condition_encoder.load_state_dict(checkpoint["condition_encoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])

    schedule = _make_schedule(dataset.manifest, config, device)
    if not isinstance(schedule, CosineVPSchedule):
        raise TypeError("absolute Cloud training requires the cosine VP schedule")
    loss_function = FullBandEnergyDistance(
        levels=int(config["cloud_matching"]["full_band_levels"]),
        include_target_constant=False,
    )
    cloud_runner: nn.Module = cloud
    condition_runner: nn.Module = condition_module
    if distributed:
        cloud_runner = DistributedDataParallel(cloud, device_ids=[local_rank])
        condition_runner = DistributedDataParallel(condition_module, device_ids=[local_rank])

    writer = None
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(args.output / "tensorboard")
        writer.add_text("run/config", f"```yaml\n{yaml.safe_dump(config, sort_keys=False)}```")
        (args.output / "architecture.json").write_text(
            json.dumps(
                {
                    "cloud_parameters": sum(value.numel() for value in cloud.parameters()),
                    "condition_parameters": sum(value.numel() for value in condition_encoder.parameters()),
                    "effective_global_batch": batch_size * world_size * accumulation,
                    "random_samples": samples,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"train={train_sampler.selected_count:,} validation={validation_sampler.selected_count:,} "
            f"microbatch/GPU={batch_size} accumulation={accumulation} "
            f"validation_batch/GPU={validation_batch_size} "
            f"effective_batch={batch_size * world_size * accumulation} samples={samples} "
            f"excluded_long_ms={excluded_count:,} "
            f"max_postpool_tokens={maximum_tokens:,}",
            flush=True,
        )

    stop = False
    log_every = (
        1 if args.max_steps is not None else int(config["training"].get("log_every", 10))
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        cloud.train()
        condition_module.train()
        totals = {"total": 0.0, "u_statistic": 0.0, "zero_token_mse": 0.0}
        count = 0
        for micro_step, batch in enumerate(train_loader):
            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break
            started = time.perf_counter()
            clean = batch["field"].to(device, non_blocking=True)
            transition = _sample_transition(schedule, clean, samples, config)
            noise = torch.randn(
                clean.shape[0], samples + 1, cloud.noise_token_count, cloud.noise_token_dim,
                device=device, dtype=clean.dtype,
            )
            noise[:, 0].zero_()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                condition, condition_mask = _condition_tokens(
                    condition_runner, batch, device
                )
                output = cloud_runner(
                    transition.current,
                    condition,
                    transition.current_levels,
                    samples=samples + 1,
                    noise=noise,
                    condition_mask=condition_mask,
                )
                zero, random = output.fields[:, 0], output.fields[:, 1:]
                u_statistic = loss_function(random, transition.target_cloud, transition.current)
                target_mean = schedule.target_mean_and_variance(
                    clean, transition.current, transition.current_levels, transition.answer_levels
                )[0]
                zero_loss = F.mse_loss(
                    (zero - transition.current).float(),
                    (target_mean - transition.current).float(),
                )
                loss = u_statistic + zero_weight * zero_loss
                predicted_variance = random.float().var(dim=1, unbiased=False).mean()
                target_variance = transition.target_cloud.float().var(
                    dim=1, unbiased=False
                ).mean()
            (loss / accumulation).backward()
            size = clean.shape[0]
            for name, value in (
                ("total", loss), ("u_statistic", u_statistic),
                ("zero_token_mse", zero_loss),
            ):
                totals[name] += float(value.detach()) * size
            count += size
            should_step = (micro_step + 1) % accumulation == 0 or micro_step + 1 == len(train_loader)
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0 and global_step % log_every == 0:
                    assert writer is not None
                    seconds = time.perf_counter() - started
                    diversity_ratio = predicted_variance.detach() / (
                        target_variance.detach().clamp_min(1e-12)
                    )
                    for name, value in (
                        ("total", loss), ("u_statistic", u_statistic),
                        ("zero_token_mse", zero_loss),
                    ):
                        writer.add_scalar(f"train/batch_{name}", float(value.detach()), global_step)
                    writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                    writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)
                    writer.add_scalar("model/anchor_gate", float(output.gate_means.detach().mean()), global_step)
                    writer.add_scalar("model/update_rms", float(output.update_rms.detach().mean()), global_step)
                    random_gate_bias = torch.sigmoid(
                        cloud.noise_attention.gate_projection.bias.detach()
                    ).mean()
                    writer.add_scalar(
                        "model/random_glu_gate_bias",
                        float(random_gate_bias),
                        global_step,
                    )
                    writer.add_scalar(
                        "diversity/batch_ratio",
                        float(diversity_ratio),
                        global_step,
                    )
                    token_counts = condition_mask.sum(dim=1)
                    writer.add_scalar(
                        "model/ms_tokens_mean",
                        float(token_counts.float().mean()),
                        global_step,
                    )
                    writer.add_scalar(
                        "model/ms_tokens_max",
                        float(token_counts.max()),
                        global_step,
                    )
                    memory = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
                    print(
                        f"epoch={epoch + 1}/{epochs} step={global_step} "
                        f"loss={float(loss.detach()):.5f} "
                        f"ustat={float(u_statistic.detach()):.5f} "
                        f"zero_mse={float(zero_loss.detach()):.5f} "
                        f"diversity={float(diversity_ratio):.3f} "
                        f"grad={float(grad_norm):.3f} ms_tokens="
                        f"{float(token_counts.float().mean()):.0f}/"
                        f"{int(token_counts.max())}mean/max "
                        f"micro_speed={size * world_size / seconds:.1f}mol/s "
                        f"peak_mem={memory:.2f}GiB",
                        flush=True,
                    )

        train_means = {name: _distributed_mean(value, count, device) for name, value in totals.items()}
        validation, preview = _validate(
            cloud, condition_module, validation_loader, schedule, loss_function, config,
            device, samples, zero_weight, args.max_validation_batches,
        )
        if rank == 0:
            assert writer is not None
            for name, value in train_means.items():
                writer.add_scalar(f"train/epoch_{name}", value, epoch + 1)
            for name, value in validation.items():
                writer.add_scalar(f"validation/{name}", value, epoch + 1)
            if preview is not None:
                writer.add_images(
                    "samples/clean_input_zero_random",
                    preview,
                    epoch + 1,
                    dataformats="NCHW",
                )
            checkpoint_path = _save(
                args.output, epoch, global_step, cloud, condition_encoder, optimizer, scheduler, config
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
