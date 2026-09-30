"""Train the image-only molecular Cloud Matching model."""

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
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from molai.data import (
    ShardShuffleSampler,
    SpectrumFieldDataset,
    collate_field_batch,
    collate_spectrum_field_batch,
    open_field_dataset,
)
from molai.models.bridge import GeometricVESchedule, VPSchedule
from molai.models.cloud import MolecularFieldCloud
from molai.models.condition import SmilesConditionEncoder, SpectrumConditionEncoder
from molai.models.losses import FullBandEnergyDistance
from molai.models.smiles import SmilesTokenizer


class RaggedSpectrumConditionRunner(nn.Module):
    """Expose the ragged spectrum path through ``forward`` for DDP."""

    def __init__(self, encoder: SpectrumConditionEncoder) -> None:
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
    ) -> Tensor:
        return self.encoder.forward_ragged(
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
    parser.add_argument("--output", type=Path, default=Path("outputs/cloud_pretrain"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--condition-dropout", type=float)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="compile the field generator (enabled by default on CUDA)",
    )
    parser.add_argument("--max-steps", type=int, help="stop early for smoke tests")
    return parser.parse_args()


def _training_value(args: argparse.Namespace, config: dict, name: str) -> object:
    argument = getattr(args, name)
    if argument is not None:
        return argument
    return config["training"][name]


def _make_schedule(
    manifest: dict[str, object], device: torch.device
) -> VPSchedule | GeometricVESchedule:
    noise_config = manifest.get("noise_schedule")
    if isinstance(noise_config, dict):
        return GeometricVESchedule(
            levels=int(noise_config["levels"]),
            sigma_min=float(noise_config["sigma_min"]),
            sigma_max=float(noise_config["sigma_max"]),
            device=device,
        )
    return VPSchedule(device=device)


def _sample_transition(
    schedule: VPSchedule | GeometricVESchedule,
    clean: Tensor,
    samples: int,
    config: dict,
    current_levels: Tensor | None = None,
):
    options = {
        "answer_jump": int(config["cloud_matching"].get("answer_jump", 8)),
        "clean_answer_probability": float(
            config["cloud_matching"].get("clean_answer_probability", 0.25)
        ),
    }
    if isinstance(schedule, GeometricVESchedule):
        return schedule.sample_training_batch(
            clean,
            samples,
            current_levels=current_levels,
            **options,
        )
    return schedule.sample_training_batch(clean, samples, **options)


def _condition_from_batch(
    runner: nn.Module,
    batch: dict[str, Tensor | list[str]],
    device: torch.device,
) -> Tensor:
    if "peak_chunks" not in batch:
        return runner(batch["token_ids"].to(device, non_blocking=True))
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


def _distributed_mean(total: float, count: int, device: torch.device) -> float:
    statistics = torch.tensor([total, count], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(statistics)
    return float(statistics[0] / statistics[1].clamp_min(1.0))


def _triptych(clean: Tensor, current: Tensor, predicted: Tensor) -> Tensor:
    """Create comparable actual/input/output panels using one scale per sample."""

    clean = clean[:, :1].float().cpu()
    current = current[:, :1].float().cpu()
    predicted = predicted[:, :1].float().cpu()
    triptychs = []
    for actual, noisy, output in zip(clean, current, predicted, strict=True):
        values = torch.cat((actual.flatten(), noisy.flatten(), output.flatten())).abs()
        scale = torch.quantile(values, 0.995).clamp_min(1e-6)
        panels = [((image / scale).clamp(-1.0, 1.0) + 1.0) * 0.5 for image in (actual, noisy, output)]
        separator = torch.ones(1, actual.shape[-2], 3)
        triptychs.append(
            torch.cat((panels[0], separator, panels[1], separator, panels[2]), dim=-1)
        )
    return torch.stack(triptychs)


def _update_latest(checkpoint_path: Path, latest_path: Path) -> None:
    temporary = latest_path.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(checkpoint_path, temporary)
    except OSError:
        shutil.copyfile(checkpoint_path, temporary)
    os.replace(temporary, latest_path)


def _save_checkpoint(
    output: Path,
    epoch: int,
    global_step: int,
    cloud: MolecularFieldCloud,
    condition_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict,
    vocabulary: list[str] | None,
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
            "global_step": global_step,
            "config": config,
            "vocabulary": vocabulary,
        },
        temporary,
    )
    os.replace(temporary, path)
    _update_latest(path, output / "latest.pt")
    return path


@torch.no_grad()
def _validate(
    cloud_runner: nn.Module,
    condition_runner: nn.Module,
    loader: DataLoader,
    schedule: VPSchedule | GeometricVESchedule,
    loss_function: FullBandEnergyDistance,
    samples: int,
    config: dict,
    device: torch.device,
    validation_seed: int,
    writer: SummaryWriter | None,
    epoch: int,
) -> float:
    cloud_runner.eval()
    condition_runner.eval()
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    torch.manual_seed(validation_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(validation_seed)
    total = 0.0
    count = 0
    images: Tensor | None = None
    try:
        for batch in loader:
            clean = batch["field"].to(device, non_blocking=True)
            levels = batch.get("noise_level")
            current_levels = levels.to(device, non_blocking=True) if isinstance(levels, Tensor) else None
            transition = _sample_transition(
                schedule, clean, samples, config, current_levels=current_levels
            )
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                condition = _condition_from_batch(condition_runner, batch, device)
                predicted, _, _ = cloud_runner(transition.current, condition, samples=samples)
                loss = loss_function(predicted, transition.target_cloud, transition.current)
            total += float(loss) * clean.shape[0]
            count += clean.shape[0]
            if images is None:
                images = _triptych(clean[:4], transition.current[:4], predicted[:4, 0])
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
    mean = _distributed_mean(total, count, device)
    if writer is not None and images is not None:
        writer.add_images(
            "samples/actual_input_output",
            images,
            epoch + 1,
            dataformats="NCHW",
        )
    return mean


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed Cloud training requires CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=device)
    else:
        device = torch.device(args.device)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    epochs = int(_training_value(args, config, "epochs"))
    learning_rate = float(_training_value(args, config, "learning_rate"))
    min_learning_rate = float(_training_value(args, config, "min_learning_rate"))
    validation_fraction = float(_training_value(args, config, "validation_fraction"))
    condition_dropout = float(_training_value(args, config, "condition_dropout"))

    dataset = open_field_dataset(args.data)
    training_seed = args.seed if args.seed is not None else int(dataset.manifest.get("seed", 0))
    torch.manual_seed(training_seed + rank)
    expected_channels = int(config["model"]["field_channels"])
    actual_channels = len(dataset.manifest.get("field_channels", ["signed_charge"]))
    if actual_channels != expected_channels:
        raise ValueError(
            f"dataset has {actual_channels} field channels but model expects "
            f"{expected_channels}; select the matching model config"
        )

    batch_size = args.batch_size or int(config["cloud_matching"].get("batch_size_per_gpu", 1))
    peak_chunk_size = int(training.get("peak_chunk_size", 256))
    vocabulary: list[str] | None = None
    tokenizer: SmilesTokenizer | None = None
    if isinstance(dataset, SpectrumFieldDataset):
        collate = partial(collate_spectrum_field_batch, peak_chunk_size=peak_chunk_size)
    else:
        if rank == 0:
            vocabulary = SmilesTokenizer.from_smiles(dataset.iter_smiles()).id_to_token
        if distributed:
            payload: list[object] = [vocabulary]
            dist.broadcast_object_list(payload, src=0)
            vocabulary = payload[0]  # type: ignore[assignment]
        if vocabulary is None:
            raise RuntimeError("tokenizer vocabulary was not initialized")
        tokenizer = SmilesTokenizer(vocabulary)
        collate = partial(collate_field_batch, tokenizer=tokenizer)

    sampler_options = {
        "seed": training_seed,
        "rank": rank,
        "replicas": world_size,
        "batch_size": batch_size,
        "validation_fraction": validation_fraction,
        "split_seed": training_seed + 17,
    }
    train_sampler = ShardShuffleSampler(dataset, split="train", shuffle=True, **sampler_options)
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
        dataset,
        sampler=validation_sampler,
        persistent_workers=False,
        **loader_options,
    )

    model_config = config["model"]
    condition_dim = int(model_config["condition_dim"])
    cloud = MolecularFieldCloud(
        condition_dim=condition_dim,
        field_channels=int(model_config["field_channels"]),
        dim=int(model_config["fullres_dim"]),
        heads=int(model_config["heads"]),
        condition_cross_depth=int(model_config["condition_cross_depth"]),
        noise_cross_depth=int(model_config["noise_cross_depth"]),
        noise_token_count=int(model_config["noise_token_count"]),
        refine_depth=int(model_config["refine_depth"]),
        window_size=int(model_config["window_size"]),
        ffn_ratio=float(model_config["ffn_ratio"]),
        max_resolution=int(model_config["max_resolution"]),
        max_residual=float(model_config["max_residual"]),
        noise_energy_min=float(model_config["noise_energy_min"]),
        noise_energy_init=float(model_config["noise_energy_init"]),
        noise_amplitude_max=float(model_config["noise_amplitude_max"]),
        gradient_checkpointing=bool(model_config["gradient_checkpointing"]),
    ).to(device)
    if isinstance(dataset, SpectrumFieldDataset):
        spectrum_config = config["spectrum_encoder"]
        condition_encoder: nn.Module = SpectrumConditionEncoder(
            metadata_dim=int(dataset.manifest["metadata_dim"]),
            dim=condition_dim,
            fourier_bands=int(spectrum_config["fourier_bands"]),
            heads=int(spectrum_config["heads"]),
            peak_layers=int(spectrum_config["peak_layers"]),
            spectrum_layers=int(spectrum_config["spectrum_layers"]),
            relative_bands=int(spectrum_config["relative_bands"]),
            relative_mass_max=float(spectrum_config["relative_mass_max"]),
            ffn_ratio=float(spectrum_config["ffn_ratio"]),
            dropout=float(spectrum_config["dropout"]),
            max_mz=float(spectrum_config["max_mz"]),
            max_peaks=peak_chunk_size,
            peak_chunk_batch=int(spectrum_config.get("peak_chunk_batch", 256)),
        ).to(device)
        condition_module: nn.Module = RaggedSpectrumConditionRunner(condition_encoder)
    else:
        if tokenizer is None:
            raise RuntimeError("SMILES tokenizer is unavailable")
        condition_encoder = SmilesConditionEncoder(
            len(tokenizer), tokenizer.pad_id, dim=condition_dim
        ).to(device)
        condition_module = condition_encoder

    parameters = [*cloud.parameters(), *condition_encoder.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs * len(train_loader)),
        eta_min=min_learning_rate,
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

    schedule = _make_schedule(dataset.manifest, device)
    cloud_samples = int(config["cloud_matching"]["samples"])
    cloud_loss = FullBandEnergyDistance(
        levels=int(config["cloud_matching"]["full_band_levels"]),
        include_target_constant=False,
    )
    compile_enabled = args.compile if args.compile is not None else device.type == "cuda"
    compiled_cloud = torch.compile(cloud, mode="reduce-overhead") if compile_enabled else cloud
    cloud_runner: nn.Module = compiled_cloud
    condition_runner: nn.Module = condition_module
    if distributed:
        cloud_runner = DistributedDataParallel(
            compiled_cloud,
            device_ids=[local_rank],
            static_graph=True,
        )
        condition_runner = DistributedDataParallel(
            condition_module,
            device_ids=[local_rank],
            static_graph=True,
        )

    writer: SummaryWriter | None = None
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        if vocabulary is not None:
            (args.output / "vocabulary.json").write_text(
                json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        writer = SummaryWriter(args.output / "tensorboard")
        writer.add_text("run/config", f"```yaml\n{yaml.safe_dump(config, sort_keys=False)}```")
        print(
            f"train={train_sampler.selected_count:,} validation="
            f"{validation_sampler.selected_count:,} batch/GPU={batch_size} "
            f"steps/epoch={len(train_loader):,} epochs={epochs}",
            flush=True,
        )
    if distributed:
        dist.barrier()

    log_every = int(training.get("log_every", 1))
    stop = False
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        cloud_runner.train()
        condition_runner.train()
        train_total = 0.0
        train_count = 0
        batch_end = time.perf_counter()
        for batch in train_loader:
            batch_start = batch_end
            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break
            clean = batch["field"].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            levels = batch.get("noise_level")
            current_levels = levels.to(device, non_blocking=True) if isinstance(levels, Tensor) else None
            transition = _sample_transition(
                schedule, clean, cloud_samples, config, current_levels=current_levels
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                condition = _condition_from_batch(condition_runner, batch, device)
                keep = torch.rand(condition.shape[0], 1, device=device) >= condition_dropout
                condition = condition * keep
                predicted_fields, _, molecular_embeddings = cloud_runner(
                    transition.current,
                    condition,
                    samples=cloud_samples,
                )
                distribution_loss = cloud_loss(
                    predicted_fields, transition.target_cloud, transition.current
                )
                loss = distribution_loss + molecular_embeddings.mean() * 0.0
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            loss_value = float(loss.detach())
            batch_seconds = time.perf_counter() - batch_start
            batch_end = time.perf_counter()
            train_total += loss_value * clean.shape[0]
            train_count += clean.shape[0]
            if rank == 0 and global_step % log_every == 0:
                assert writer is not None
                writer.add_scalar("batch/loss", loss_value, global_step)
                writer.add_scalar("batch/grad_norm", float(grad_norm), global_step)
                writer.add_scalar(
                    "batch/learning_rate", optimizer.param_groups[0]["lr"], global_step
                )
                writer.add_scalar("batch/seconds", batch_seconds, global_step)
                writer.add_scalar(
                    "batch/molecules_per_second",
                    clean.shape[0] * world_size / batch_seconds,
                    global_step,
                )
                peak_memory = None
                peak_reserved = None
                if device.type == "cuda":
                    peak_memory = torch.cuda.max_memory_allocated(device) / 2**30
                    peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30
                    writer.add_scalar("batch/gpu_peak_memory_gib", peak_memory, global_step)
                    writer.add_scalar(
                        "batch/gpu_peak_reserved_gib", peak_reserved, global_step
                    )
                if "spectrum_to_molecule" in batch:
                    writer.add_scalar(
                        "batch/spectra",
                        int(batch["spectrum_to_molecule"].numel()),
                        global_step,
                    )
                    writer.add_scalar(
                        "batch/peaks",
                        int(batch["peak_mask"].sum()),
                        global_step,
                    )
                    writer.add_scalar(
                        "batch/peak_chunks",
                        int(batch["peak_chunks"].shape[0]),
                        global_step,
                    )
                print(
                    f"epoch={epoch + 1}/{epochs} step={global_step} "
                    f"loss={loss_value:.5f} grad={float(grad_norm):.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.7f}"
                    + (
                        f" speed={clean.shape[0] * world_size / batch_seconds:.1f}mol/s"
                        f" peak_mem={peak_memory:.2f}/{peak_reserved:.2f}GiB"
                        if peak_memory is not None and peak_reserved is not None
                        else ""
                    ),
                    flush=True,
                )

        train_mean = _distributed_mean(train_total, train_count, device)
        validation_mean = _validate(
            cloud_runner,
            condition_runner,
            validation_loader,
            schedule,
            cloud_loss,
            cloud_samples,
            config,
            device,
            training_seed + 100_003 + rank,
            writer,
            epoch,
        )
        if rank == 0:
            assert writer is not None
            writer.add_scalar("epoch/train_loss_mean", train_mean, epoch + 1)
            writer.add_scalar("epoch/validation_loss", validation_mean, epoch + 1)
            checkpoint_path = _save_checkpoint(
                args.output,
                epoch,
                global_step,
                cloud,
                condition_encoder,
                optimizer,
                scheduler,
                config,
                vocabulary,
            )
            writer.flush()
            print(
                f"epoch={epoch + 1} train_mean={train_mean:.5f} "
                f"validation={validation_mean:.5f} saved={checkpoint_path}",
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
