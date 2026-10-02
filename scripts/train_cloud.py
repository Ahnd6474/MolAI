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
from torch.nn import functional as F
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
from molai.models.bridge import BridgeBatch, CosineVPSchedule, GeometricVESchedule, VPSchedule
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
    manifest: dict[str, object], config: dict, device: torch.device
) -> VPSchedule | GeometricVESchedule:
    training_schedule = config["cloud_matching"].get("noise_schedule")
    if isinstance(training_schedule, dict):
        schedule_type = str(training_schedule.get("type", "")).lower()
        if schedule_type == "cosine_vp":
            return CosineVPSchedule(
                levels=int(training_schedule.get("levels", 64)),
                offset=float(training_schedule.get("cosine_offset", 0.008)),
                noise_scale=float(training_schedule.get("noise_scale", 1.0)),
                zero_mean_noise=bool(training_schedule.get("zero_mean_noise", False)),
                device=device,
            )
        if schedule_type != "geometric_ve":
            raise ValueError(f"unsupported cloud noise schedule: {schedule_type}")
        return GeometricVESchedule(
            levels=int(training_schedule["levels"]),
            sigma_min=float(training_schedule["sigma_min"]),
            sigma_max=float(training_schedule["sigma_max"]),
            device=device,
        )
    noise_config = manifest.get("noise_schedule")
    if isinstance(noise_config, dict):
        return GeometricVESchedule(
            levels=int(noise_config["levels"]),
            sigma_min=float(noise_config["sigma_min"]),
            sigma_max=float(noise_config["sigma_max"]),
            device=device,
        )
    return VPSchedule(device=device)


def _transition_jumps(current_levels: Tensor, config: dict) -> Tensor:
    configured = config["cloud_matching"].get("transition_jumps")
    if not configured:
        return torch.full_like(
            current_levels, int(config["cloud_matching"].get("answer_jump", 8))
        )
    jumps = torch.zeros_like(current_levels)
    previous_maximum = 0
    for band in configured:
        maximum = int(band["max_level"])
        jump = int(band["jump"])
        if maximum <= previous_maximum or jump < 1:
            raise ValueError("transition jump bands must have increasing maxima and positive jumps")
        selected = (current_levels > previous_maximum) & (current_levels <= maximum)
        jumps = torch.where(selected, jump, jumps)
        previous_maximum = maximum
    if bool((jumps[current_levels > 0] == 0).any()):
        raise ValueError("transition jump bands do not cover every positive noise level")
    return jumps


def _sample_cosine_levels(batch: int, schedule: CosineVPSchedule, config: dict) -> Tensor:
    options = config["cloud_matching"]
    maximum = len(schedule.alpha_bar) - 1
    sampling = str(options.get("level_sampling", "stratified"))
    device = schedule.alpha_bar.device
    if sampling == "uniform":
        return torch.randint(0, maximum + 1, (batch,), device=device)
    if sampling != "stratified":
        raise ValueError("level_sampling must be uniform or stratified")
    low_maximum = int(options.get("low_noise_max_level", min(16, maximum)))
    low_probability = float(options.get("low_noise_probability", 0.25))
    fixed_probability = float(options.get("clean_fixed_probability", 0.0))
    if not 1 <= low_maximum < maximum:
        raise ValueError("low_noise_max_level must lie below the terminal level")
    if not 0.0 <= low_probability <= 1.0 or not 0.0 <= fixed_probability <= 1.0:
        raise ValueError("noise-level sampling probabilities must lie in [0, 1]")
    low = torch.rand(batch, device=device) < low_probability
    low_levels = torch.randint(1, low_maximum + 1, (batch,), device=device)
    high_levels = torch.randint(low_maximum + 1, maximum + 1, (batch,), device=device)
    levels = torch.where(low, low_levels, high_levels)
    fixed = torch.rand(batch, device=device) < fixed_probability
    return torch.where(fixed, torch.zeros_like(levels), levels)


def _sample_transition(
    schedule: VPSchedule | GeometricVESchedule,
    clean: Tensor,
    samples: int,
    config: dict,
    current_levels: Tensor | None = None,
):
    if isinstance(schedule, GeometricVESchedule):
        return schedule.sample_training_batch(
            clean,
            samples,
            current_levels=current_levels,
            answer_jump=int(config["cloud_matching"].get("answer_jump", 8)),
            clean_answer_probability=0.0,
        )
    if isinstance(schedule, CosineVPSchedule):
        if current_levels is None:
            current_levels = _sample_cosine_levels(clean.shape[0], schedule, config)
        jumps = _transition_jumps(current_levels, config)
        return schedule.sample_training_batch(
            clean,
            samples,
            current_levels=current_levels,
            answer_jump=jumps,
            clean_answer_probability=0.0,
        )
    return schedule.sample_training_batch(
        clean,
        samples,
        current_levels=current_levels,
        answer_jump=int(config["cloud_matching"].get("answer_jump", 8)),
        clean_answer_probability=0.0,
    )


def _continue_cosine_transition(
    schedule: CosineVPSchedule,
    clean: Tensor,
    current: Tensor,
    current_levels: Tensor,
    samples: int,
    config: dict,
) -> BridgeBatch:
    jumps = _transition_jumps(current_levels, config)
    answer_levels = (current_levels - jumps).clamp_min(0)
    target_cloud = schedule.sample_target_cloud(
        clean,
        current,
        current_levels,
        answer_levels,
        samples,
    )
    return BridgeBatch(current, target_cloud, current_levels, answer_levels)


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


def _comparison_strip(
    clean: Tensor,
    current: Tensor,
    zero_output: Tensor,
    random_output: Tensor,
) -> Tensor:
    """Create actual/input/zero-token/random-output panels on a shared scale."""

    clean = clean[:, :1].float().cpu()
    current = current[:, :1].float().cpu()
    zero_output = zero_output[:, :1].float().cpu()
    random_output = random_output[:, :1].float().cpu()
    strips = []
    for actual, noisy, deterministic, stochastic in zip(
        clean, current, zero_output, random_output, strict=True
    ):
        values = torch.cat(
            (
                actual.flatten(),
                noisy.flatten(),
                deterministic.flatten(),
                stochastic.flatten(),
            )
        ).abs()
        scale = torch.quantile(values, 0.995).clamp_min(1e-6)
        panels = [
            ((image / scale).clamp(-1.0, 1.0) + 1.0) * 0.5
            for image in (actual, noisy, deterministic, stochastic)
        ]
        separator = torch.ones(1, actual.shape[-2], 3)
        pieces = []
        for index, panel in enumerate(panels):
            if index:
                pieces.append(separator)
            pieces.append(panel)
        strips.append(torch.cat(pieces, dim=-1))
    return torch.stack(strips)


def _predict_zero_and_random(
    cloud_runner: nn.Module,
    current: Tensor,
    condition: Tensor,
    random_samples: int,
    noise_token_count: int,
    noise_token_dim: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run one deterministic zero-token sample plus an independent random cloud."""

    noise = torch.randn(
        current.shape[0],
        random_samples + 1,
        noise_token_count,
        noise_token_dim,
        device=current.device,
        dtype=current.dtype,
    )
    noise[:, 0].zero_()
    fields, energy, embeddings = cloud_runner(
        current,
        condition,
        samples=random_samples + 1,
        noise=noise,
    )
    return fields[:, 0], fields[:, 1:], energy, embeddings


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
    zero_token_clean_mse_weight: float,
    noise_token_count: int,
    noise_token_dim: int,
) -> dict[str, float]:
    cloud_runner.eval()
    condition_runner.eval()
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    torch.manual_seed(validation_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(validation_seed)
    total = 0.0
    u_statistic_total = 0.0
    zero_mse_total = 0.0
    predicted_variance_total = 0.0
    target_variance_total = 0.0
    count = 0
    images: Tensor | None = None
    preview_diversity: list[tuple[int, float, float]] = []
    try:
        for batch in loader:
            clean = batch["field"].to(device, non_blocking=True)
            levels = batch.get("noise_level")
            current_levels = (
                levels.to(device, non_blocking=True) if isinstance(levels, Tensor) else None
            )
            transition = _sample_transition(
                schedule, clean, samples, config, current_levels=current_levels
            )
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                condition = _condition_from_batch(condition_runner, batch, device)
                zero_output, predicted, _, _ = _predict_zero_and_random(
                    cloud_runner,
                    transition.current,
                    condition,
                    samples,
                    noise_token_count,
                    noise_token_dim,
                )
                u_statistic = loss_function(
                    predicted, transition.target_cloud, transition.current
                )
                zero_mse = F.mse_loss(zero_output.float(), clean.float())
                loss = u_statistic + zero_token_clean_mse_weight * zero_mse
            batch_count = clean.shape[0]
            total += float(loss) * batch_count
            u_statistic_total += float(u_statistic) * batch_count
            zero_mse_total += float(zero_mse) * batch_count
            predicted_variance_total += float(
                predicted.float().var(dim=1, unbiased=False).mean()
            ) * batch_count
            target_variance_total += float(
                transition.target_cloud.float().var(dim=1, unbiased=False).mean()
            ) * batch_count
            count += clean.shape[0]
            if images is None:
                preview_levels = config["cloud_matching"].get(
                    "validation_preview_levels", [8, 24, 48, 64]
                )
                preview_count = min(len(preview_levels), clean.shape[0])
                preview_level_tensor = torch.tensor(
                    preview_levels[:preview_count], device=device, dtype=torch.long
                )
                preview_transition = _sample_transition(
                    schedule,
                    clean[:preview_count],
                    samples,
                    config,
                    current_levels=preview_level_tensor,
                )
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    preview_zero, preview_predicted, _, _ = _predict_zero_and_random(
                        cloud_runner,
                        preview_transition.current,
                        condition[:preview_count],
                        samples,
                        noise_token_count,
                        noise_token_dim,
                    )
                images = _comparison_strip(
                    clean[:preview_count],
                    preview_transition.current,
                    preview_zero,
                    preview_predicted[:, 0],
                )
                for index, level in enumerate(preview_levels[:preview_count]):
                    predicted_variance = float(
                        preview_predicted[index].float().var(dim=0, unbiased=False).mean()
                    )
                    target_variance = float(
                        preview_transition.target_cloud[index]
                        .float()
                        .var(dim=0, unbiased=False)
                        .mean()
                    )
                    preview_diversity.append(
                        (int(level), predicted_variance, target_variance)
                    )
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
    metrics = {
        "total": _distributed_mean(total, count, device),
        "u_statistic": _distributed_mean(u_statistic_total, count, device),
        "zero_token_clean_mse": _distributed_mean(zero_mse_total, count, device),
        "predicted_variance": _distributed_mean(
            predicted_variance_total, count, device
        ),
        "target_variance": _distributed_mean(target_variance_total, count, device),
    }
    metrics["diversity_ratio"] = metrics["predicted_variance"] / max(
        metrics["target_variance"], 1e-12
    )
    if writer is not None and images is not None:
        writer.add_images(
            "samples/actual_input_zero_random",
            images,
            epoch + 1,
            dataformats="NCHW",
        )
        for level, predicted_variance, target_variance in preview_diversity:
            writer.add_scalar(
                f"diversity/ratio_L{level:02d}",
                predicted_variance / max(target_variance, 1e-12),
                epoch + 1,
            )
    return metrics


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
        noise_token_dim=int(model_config["noise_token_dim"]),
        noise_temperature=float(model_config["noise_temperature"]),
        noise_gate_init=float(model_config["noise_gate_init"]),
        refine_depth=int(model_config["refine_depth"]),
        window_size=int(model_config["window_size"]),
        ffn_ratio=float(model_config["ffn_ratio"]),
        cvt_kernel_sizes=[int(value) for value in model_config["cvt_kernel_sizes"]],
        cvt_output_sizes=[int(value) for value in model_config["cvt_output_sizes"]],
        condition_gate_init=float(model_config["condition_gate_init"]),
        max_resolution=int(model_config["max_resolution"]),
        noise_energy_min=float(model_config["noise_energy_min"]),
        noise_energy_init=float(model_config["noise_energy_init"]),
        noise_amplitude_max=float(model_config["noise_amplitude_max"]),
        zero_mean_output=bool(model_config.get("zero_mean_output", False)),
        gradient_checkpointing=bool(model_config["gradient_checkpointing"]),
    ).to(device)
    if isinstance(dataset, SpectrumFieldDataset):
        spectrum_config = config["spectrum_encoder"]
        condition_encoder: nn.Module = SpectrumConditionEncoder(
            metadata_dim=int(dataset.manifest["metadata_dim"]),
            dim=condition_dim,
            heads=int(spectrum_config["heads"]),
            peak_conv_stages=int(spectrum_config["peak_conv_stages"]),
            peak_conv_kernel_size=int(spectrum_config["peak_conv_kernel_size"]),
            peak_tau_min=float(spectrum_config["peak_tau_min"]),
            peak_tau_max=float(spectrum_config["peak_tau_max"]),
            peak_cutoff_multiplier=float(spectrum_config["peak_cutoff_multiplier"]),
            spectrum_layers=int(spectrum_config["spectrum_layers"]),
            ffn_ratio=float(spectrum_config["ffn_ratio"]),
            dropout=float(spectrum_config["dropout"]),
            peak_position_dim=int(spectrum_config["peak_position_dim"]),
            mz_bin_width=float(spectrum_config["mz_bin_width"]),
            mz_upper_bound=float(spectrum_config["mz_upper_bound"]),
            peak_chunk_size=peak_chunk_size,
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

    schedule = _make_schedule(dataset.manifest, config, device)
    cloud_samples = int(config["cloud_matching"]["samples"])
    zero_token_clean_mse_weight = float(
        config["cloud_matching"].get("zero_token_clean_mse_weight", 0.0)
    )
    if zero_token_clean_mse_weight < 0.0:
        raise ValueError("zero_token_clean_mse_weight must be non-negative")
    noise_token_count = int(model_config["noise_token_count"])
    noise_token_dim = int(model_config["noise_token_dim"])
    unroll_fraction = float(config["cloud_matching"].get("unroll_fraction", 0.0))
    if not 0.0 <= unroll_fraction <= 1.0:
        raise ValueError("unroll_fraction must lie in [0, 1]")
    if unroll_fraction > 0.0 and not isinstance(schedule, CosineVPSchedule):
        raise ValueError("two-step unrolling currently requires a cosine VP schedule")
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
        train_u_statistic_total = 0.0
        train_zero_mse_total = 0.0
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
            current_levels = (
                levels.to(device, non_blocking=True) if isinstance(levels, Tensor) else None
            )
            transition = _sample_transition(
                schedule, clean, cloud_samples, config, current_levels=current_levels
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                condition = _condition_from_batch(condition_runner, batch, device)
                keep = torch.rand(condition.shape[0], 1, device=device) >= condition_dropout
                condition = condition * keep
                zero_output, predicted_fields, _, molecular_embeddings = (
                    _predict_zero_and_random(
                        cloud_runner,
                        transition.current,
                        condition,
                        cloud_samples,
                        noise_token_count,
                        noise_token_dim,
                    )
                )
                distribution_loss = cloud_loss(
                    predicted_fields, transition.target_cloud, transition.current
                )
                zero_token_clean_mse = F.mse_loss(
                    zero_output.float(), clean.float()
                )
                primary_loss = (
                    distribution_loss
                    + zero_token_clean_mse_weight * zero_token_clean_mse
                )
                unroll_loss = distribution_loss.new_zeros(())
                unroll_count = min(
                    clean.shape[0], round(clean.shape[0] * unroll_fraction)
                )
                if unroll_fraction > 0.0 and unroll_count == 0:
                    unroll_count = 1
                if unroll_count:
                    assert isinstance(schedule, CosineVPSchedule)
                    selected = torch.randperm(clean.shape[0], device=device)[:unroll_count]
                    rollout_current = predicted_fields[selected, 0]
                    continued = _continue_cosine_transition(
                        schedule,
                        clean[selected],
                        rollout_current,
                        transition.answer_levels[selected],
                        cloud_samples,
                        config,
                    )
                    rollout_predicted, _, _ = cloud_runner(
                        rollout_current,
                        condition[selected],
                        samples=cloud_samples,
                    )
                    unroll_loss = cloud_loss(
                        rollout_predicted,
                        continued.target_cloud,
                        continued.current,
                    )
                effective_unroll_fraction = unroll_count / clean.shape[0]
                loss = (
                    primary_loss + effective_unroll_fraction * unroll_loss
                ) / (1.0 + effective_unroll_fraction)
                loss = loss + molecular_embeddings.mean() * 0.0
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            loss_value = float(loss.detach())
            batch_seconds = time.perf_counter() - batch_start
            batch_end = time.perf_counter()
            train_total += loss_value * clean.shape[0]
            train_u_statistic_total += float(distribution_loss.detach()) * clean.shape[0]
            train_zero_mse_total += float(zero_token_clean_mse.detach()) * clean.shape[0]
            train_count += clean.shape[0]
            if rank == 0 and global_step % log_every == 0:
                assert writer is not None
                writer.add_scalar("train/batch_total", loss_value, global_step)
                writer.add_scalar(
                    "train/batch_u_statistic",
                    float(distribution_loss.detach()),
                    global_step,
                )
                writer.add_scalar(
                    "train/batch_zero_token_clean_mse",
                    float(zero_token_clean_mse.detach()),
                    global_step,
                )
                writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                writer.add_scalar(
                    "train/learning_rate", optimizer.param_groups[0]["lr"], global_step
                )
                writer.add_scalar(
                    "performance/molecules_per_second",
                    clean.shape[0] * world_size / batch_seconds,
                    global_step,
                )
                predicted_variance = float(
                    predicted_fields.detach().float().var(dim=1, unbiased=False).mean()
                )
                target_variance = float(
                    transition.target_cloud.detach()
                    .float()
                    .var(dim=1, unbiased=False)
                    .mean()
                )
                diversity_ratio = predicted_variance / max(target_variance, 1e-12)
                writer.add_scalar(
                    "diversity/batch_ratio", diversity_ratio, global_step
                )
                gate = cloud.noise_attention.gate.detach()
                writer.add_scalar("model/random_attention_gate", float(gate), global_step)
                peak_memory = (
                    torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else None
                )
                print(
                    f"epoch={epoch + 1}/{epochs} step={global_step} "
                    f"loss={loss_value:.5f} ustat={float(distribution_loss.detach()):.5f} "
                    f"zero_mse={float(zero_token_clean_mse.detach()):.5f} "
                    f"diversity={diversity_ratio:.3f} "
                    f"grad={float(grad_norm):.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.7f}"
                    + (
                        f" speed={clean.shape[0] * world_size / batch_seconds:.1f}mol/s"
                        f" peak_mem={peak_memory:.2f}GiB"
                        if peak_memory is not None
                        else ""
                    ),
                    flush=True,
                )

        train_mean = _distributed_mean(train_total, train_count, device)
        train_u_statistic_mean = _distributed_mean(
            train_u_statistic_total, train_count, device
        )
        train_zero_mse_mean = _distributed_mean(
            train_zero_mse_total, train_count, device
        )
        validation = _validate(
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
            zero_token_clean_mse_weight,
            noise_token_count,
            noise_token_dim,
        )
        if rank == 0:
            assert writer is not None
            writer.add_scalar("train/epoch_total_mean", train_mean, epoch + 1)
            writer.add_scalar(
                "train/epoch_u_statistic_mean", train_u_statistic_mean, epoch + 1
            )
            writer.add_scalar(
                "train/epoch_zero_token_clean_mse_mean",
                train_zero_mse_mean,
                epoch + 1,
            )
            writer.add_scalar("validation/total", validation["total"], epoch + 1)
            writer.add_scalar(
                "validation/u_statistic", validation["u_statistic"], epoch + 1
            )
            writer.add_scalar(
                "validation/zero_token_clean_mse",
                validation["zero_token_clean_mse"],
                epoch + 1,
            )
            writer.add_scalar(
                "diversity/validation_ratio",
                validation["diversity_ratio"],
                epoch + 1,
            )
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
                f"validation={validation['total']:.5f} "
                f"diversity={validation['diversity_ratio']:.4f} "
                f"saved={checkpoint_path}",
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
