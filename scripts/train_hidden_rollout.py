"""Jointly train the encoder-anchor decoder and one-step hidden transition.

The pretrained Cloud and condition encoder are frozen.  One stochastic teacher
sample is produced by the original image-space model, while the trainable path
learns both ``D(E(x)) = x`` and ``D(anchor + GLU(cross)) = teacher(x)``.
"""

from __future__ import annotations

import argparse
import json
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
from train_cloud import (
    RaggedSpectrumConditionRunner,
    _condition_from_batch,
    _make_schedule,
    _sample_transition,
)

from molai.data import ShardShuffleSampler, SpectrumFieldDataset, collate_spectrum_field_batch
from molai.models.cloud import HiddenRolloutCloud, MolecularFieldCloud
from molai.models.condition import SpectrumConditionEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, help="stop early for a smoke test")
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def _build_cloud(config: dict, device: torch.device) -> MolecularFieldCloud:
    model = config["model"]
    return MolecularFieldCloud(
        condition_dim=int(model["condition_dim"]),
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
        cvt_kernel_sizes=[int(value) for value in model["cvt_kernel_sizes"]],
        cvt_grid_sizes=[int(value) for value in model["cvt_grid_sizes"]],
        condition_gate_init=float(model["condition_gate_init"]),
        max_resolution=int(model["max_resolution"]),
        noise_energy_min=float(model["noise_energy_min"]),
        noise_energy_init=float(model["noise_energy_init"]),
        noise_amplitude_max=float(model["noise_amplitude_max"]),
        zero_mean_output=bool(model.get("zero_mean_output", False)),
        gradient_checkpointing=False,
    ).to(device)


def _build_condition_encoder(
    dataset: SpectrumFieldDataset, config: dict, device: torch.device
) -> SpectrumConditionEncoder:
    options = config["spectrum_encoder"]
    peak_chunk_size = int(config["training"].get("peak_chunk_size", 256))
    return SpectrumConditionEncoder(
        metadata_dim=int(dataset.manifest["metadata_dim"]),
        dim=int(config["model"]["condition_dim"]),
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
        peak_chunk_size=peak_chunk_size,
        peak_chunk_batch=int(options.get("peak_chunk_batch", 256)),
    ).to(device)


def _zero_mean_like_model(field: Tensor, cloud: MolecularFieldCloud) -> Tensor:
    if not cloud.zero_mean_output:
        return field
    return field - field.float().mean(dim=(-2, -1), keepdim=True).to(field.dtype)


def _image_strip(*images: Tensor) -> Tensor:
    panels = [image[:, :1].detach().float().cpu() for image in images]
    strips = []
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
        strips.append(torch.cat(pieces, dim=-1))
    return torch.stack(strips)


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


def _save_checkpoint(
    output: Path,
    epoch: int,
    global_step: int,
    model: HiddenRolloutCloud,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict,
    source_checkpoint: Path,
) -> Path:
    path = output / f"epoch-{epoch + 1:04d}.pt"
    temporary = output / f".{path.name}.tmp"
    torch.save(
        {
            "state_update": model.state_update.state_dict(),
            "absolute_decoder": model.output_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
            "source_checkpoint": str(source_checkpoint),
        },
        temporary,
    )
    os.replace(temporary, path)
    _update_latest(path, output / "latest.pt")
    return path


@torch.no_grad()
def _validate(
    model: HiddenRolloutCloud,
    cloud: MolecularFieldCloud,
    condition_runner: nn.Module,
    loader: DataLoader,
    schedule: object,
    config: dict,
    device: torch.device,
    reconstruction_weight: float,
    transition_weight: float,
    seed: int,
    max_batches: int | None = None,
) -> tuple[dict[str, float], Tensor | None]:
    model.eval()
    condition_runner.eval()
    generator_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    totals = {"total": 0.0, "reconstruction": 0.0, "transition": 0.0}
    count = 0
    preview = None
    try:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            clean = batch["field"].to(device, non_blocking=True)
            transition = _sample_transition(schedule, clean, 1, config)
            current = transition.current
            condition = _condition_from_batch(condition_runner, batch, device)
            noise = torch.randn(
                current.shape[0],
                1,
                cloud.noise_token_count,
                cloud.noise_token_dim,
                device=device,
                dtype=current.dtype,
            )
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                teacher, _, _ = cloud(current, condition, samples=1, noise=noise)
                output = model(
                    current,
                    condition,
                    transition.current_levels[:, None],
                    samples=1,
                    noise=noise[:, :, None],
                )
                reconstruction_target = _zero_mean_like_model(current, cloud)
                reconstruction = F.mse_loss(
                    output.anchor_reconstruction.float(), reconstruction_target.float()
                )
                transition_loss = F.mse_loss(
                    output.fields[:, 0, 0].float(), teacher[:, 0].float()
                )
                loss = (
                    reconstruction_weight * reconstruction
                    + transition_weight * transition_loss
                )
            size = clean.shape[0]
            totals["total"] += float(loss) * size
            totals["reconstruction"] += float(reconstruction) * size
            totals["transition"] += float(transition_loss) * size
            count += size
            if preview is None:
                preview = _image_strip(
                    current[:4],
                    output.anchor_reconstruction[:4],
                    teacher[:4, 0],
                    output.fields[:4, 0, 0],
                )
    finally:
        torch.random.set_rng_state(generator_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
    return (
        {name: _distributed_mean(value, count, device) for name, value in totals.items()},
        preview,
    )


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
    hidden_config = config["hidden_rollout"]
    batch_size = args.batch_size or int(hidden_config["batch_size_per_gpu"])
    epochs = args.epochs or int(hidden_config["epochs"])
    reconstruction_weight = float(hidden_config["reconstruction_weight"])
    transition_weight = float(hidden_config["transition_weight"])
    if int(hidden_config.get("samples", 1)) != 1:
        raise ValueError("joint hidden-rollout training requires exactly one sample")

    dataset = SpectrumFieldDataset(args.data)
    seed = int(dataset.manifest.get("seed", 0))
    torch.manual_seed(seed + rank)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cloud = _build_cloud(config, device)
    condition_encoder = _build_condition_encoder(dataset, config, device)
    cloud.load_state_dict(checkpoint["cloud"])
    condition_encoder.load_state_dict(checkpoint["condition_encoder"])
    cloud.requires_grad_(False).eval()
    condition_encoder.requires_grad_(False).eval()
    condition_module = RaggedSpectrumConditionRunner(condition_encoder)

    model = HiddenRolloutCloud(
        cloud,
        max_level=int(hidden_config["max_level"]),
        gate_init=float(hidden_config["gate_init"]),
        decoder_dim=int(hidden_config["decoder_dim"]),
        kv_grid_sizes=[int(value) for value in hidden_config["kv_grid_sizes"]],
    ).to(device)
    trainable = [*model.state_update.parameters(), *model.output_head.parameters()]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(hidden_config["learning_rate"]),
        weight_decay=1e-4,
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
    }
    train_sampler = ShardShuffleSampler(
        dataset, split="train", shuffle=True, **sampler_options
    )
    validation_sampler = ShardShuffleSampler(
        dataset, split="validation", shuffle=False, **sampler_options
    )
    loader_options = {
        "batch_size": batch_size,
        "collate_fn": collate,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        dataset,
        sampler=train_sampler,
        persistent_workers=args.workers > 0,
        **loader_options,
    )
    validation_loader = DataLoader(
        dataset, sampler=validation_sampler, persistent_workers=False, **loader_options
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs * len(train_loader)),
        eta_min=float(hidden_config["min_learning_rate"]),
    )
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        resumed = torch.load(args.resume, map_location=device, weights_only=False)
        model.state_update.load_state_dict(resumed["state_update"])
        model.output_head.load_state_dict(resumed["absolute_decoder"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        start_epoch = int(resumed["epoch"]) + 1
        global_step = int(resumed["global_step"])

    schedule = _make_schedule(dataset.manifest, config, device)
    runner: nn.Module = model
    if distributed:
        runner = DistributedDataParallel(model, device_ids=[local_rank], static_graph=True)

    writer = None
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(args.output / "tensorboard")
        writer.add_text("run/config", f"```yaml\n{yaml.safe_dump(config, sort_keys=False)}```")
        (args.output / "source.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
                    "samples": 1,
                    "trainable_parameters": sum(value.numel() for value in trainable),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"joint training: train={train_sampler.selected_count:,} "
            f"validation={validation_sampler.selected_count:,} batch/GPU={batch_size} "
            f"samples=1 steps/epoch={len(train_loader):,}",
            flush=True,
        )

    stop = False
    log_every = int(config["training"].get("log_every", 10))
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        cloud.eval()
        condition_module.eval()
        totals = {"total": 0.0, "reconstruction": 0.0, "transition": 0.0}
        count = 0
        for batch in train_loader:
            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break
            started = time.perf_counter()
            clean = batch["field"].to(device, non_blocking=True)
            transition = _sample_transition(schedule, clean, 1, config)
            current = transition.current
            with torch.no_grad():
                condition = _condition_from_batch(condition_module, batch, device)
                noise = torch.randn(
                    current.shape[0],
                    1,
                    cloud.noise_token_count,
                    cloud.noise_token_dim,
                    device=device,
                    dtype=current.dtype,
                )
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    teacher, _, _ = cloud(current, condition, samples=1, noise=noise)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = runner(
                    current,
                    condition,
                    transition.current_levels[:, None],
                    samples=1,
                    noise=noise[:, :, None],
                )
                reconstruction_target = _zero_mean_like_model(current, cloud)
                reconstruction = F.mse_loss(
                    output.anchor_reconstruction.float(), reconstruction_target.float()
                )
                transition_loss = F.mse_loss(
                    output.fields[:, 0, 0].float(), teacher[:, 0].float()
                )
                loss = (
                    reconstruction_weight * reconstruction
                    + transition_weight * transition_loss
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            size = clean.shape[0]
            totals["total"] += float(loss.detach()) * size
            totals["reconstruction"] += float(reconstruction.detach()) * size
            totals["transition"] += float(transition_loss.detach()) * size
            count += size
            if rank == 0 and global_step % log_every == 0:
                assert writer is not None
                seconds = time.perf_counter() - started
                writer.add_scalar("train/batch_total", float(loss.detach()), global_step)
                writer.add_scalar(
                    "train/batch_reconstruction", float(reconstruction.detach()), global_step
                )
                writer.add_scalar(
                    "train/batch_transition", float(transition_loss.detach()), global_step
                )
                writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)
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
                memory = (
                    torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else 0.0
                )
                print(
                    f"epoch={epoch + 1}/{epochs} step={global_step} "
                    f"loss={float(loss.detach()):.6f} "
                    f"recon={float(reconstruction.detach()):.6f} "
                    f"transition={float(transition_loss.detach()):.6f} "
                    f"grad={float(grad_norm):.4f} "
                    f"speed={size * world_size / seconds:.1f}mol/s "
                    f"peak_mem={memory:.2f}GiB",
                    flush=True,
                )

        train_means = {
            name: _distributed_mean(value, count, device) for name, value in totals.items()
        }
        validation, preview = _validate(
            model,
            cloud,
            condition_module,
            validation_loader,
            schedule,
            config,
            device,
            reconstruction_weight,
            transition_weight,
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
                    "samples/input_reconstruction_teacher_student",
                    preview,
                    epoch + 1,
                    dataformats="NCHW",
                )
            checkpoint_path = _save_checkpoint(
                args.output,
                epoch,
                global_step,
                model,
                optimizer,
                scheduler,
                config,
                args.checkpoint,
            )
            writer.flush()
            print(
                f"epoch={epoch + 1} train={train_means['total']:.6f} "
                f"validation={validation['total']:.6f} saved={checkpoint_path}",
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
