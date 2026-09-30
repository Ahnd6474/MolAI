"""Train the image-only molecular Cloud Matching model."""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from molai.data import FieldShardDataset, ShardShuffleSampler, collate_field_batch
from molai.models.bridge import VPSchedule
from molai.models.cloud import MolecularFieldCloud
from molai.models.condition import SmilesConditionEncoder
from molai.models.losses import FullBandEnergyDistance
from molai.models.smiles import SmilesTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/cloud_pretrain"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--condition-dropout", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="compile the field generator (enabled by default on CUDA)",
    )
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


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
    torch.manual_seed(args.seed + rank)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = FieldShardDataset(args.data)
    expected_channels = int(config["model"]["field_channels"])
    actual_channels = len(dataset.manifest.get("field_channels", ["signed_charge"]))
    if actual_channels != expected_channels:
        raise ValueError(
            f"dataset has {actual_channels} field channels but model expects "
            f"{expected_channels}; select the matching model config"
        )
    vocabulary: list[str] | None = None
    if rank == 0:
        vocabulary = SmilesTokenizer.from_smiles(dataset.iter_smiles()).id_to_token
    if distributed:
        payload: list[object] = [vocabulary]
        dist.broadcast_object_list(payload, src=0)
        vocabulary = payload[0]  # type: ignore[assignment]
    if vocabulary is None:
        raise RuntimeError("tokenizer vocabulary was not initialized")
    tokenizer = SmilesTokenizer(vocabulary)
    sampler = ShardShuffleSampler(
        dataset,
        seed=args.seed,
        rank=rank,
        replicas=world_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=partial(collate_field_batch, tokenizer=tokenizer),
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
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
    condition_encoder = SmilesConditionEncoder(
        len(tokenizer), tokenizer.pad_id, dim=condition_dim
    ).to(device)
    parameters = [*cloud.parameters(), *condition_encoder.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    schedule = VPSchedule(device=device)
    cloud_samples = int(config["cloud_matching"]["samples"])
    cloud_loss = FullBandEnergyDistance(
        levels=int(config["cloud_matching"]["full_band_levels"]),
        include_target_constant=False,
    )
    compile_enabled = args.compile if args.compile is not None else device.type == "cuda"
    compiled_cloud = torch.compile(cloud, mode="reduce-overhead") if compile_enabled else cloud
    cloud_runner: torch.nn.Module = compiled_cloud
    condition_runner: torch.nn.Module = condition_encoder
    if distributed:
        cloud_runner = DistributedDataParallel(
            compiled_cloud,
            device_ids=[local_rank],
            static_graph=True,
        )
        condition_runner = DistributedDataParallel(
            condition_encoder,
            device_ids=[local_rank],
            static_graph=True,
        )

    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "vocabulary.json").write_text(
            json.dumps(tokenizer.id_to_token, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if distributed:
        dist.barrier()
    global_step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            clean = batch["field"].to(device, non_blocking=True)
            token_ids = batch["token_ids"].to(device, non_blocking=True)
            transition = schedule.sample_training_batch(clean, cloud_samples)
            condition = condition_runner(token_ids)
            keep = torch.rand(condition.shape[0], 1, device=device) >= args.condition_dropout
            condition = condition * keep

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
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
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            global_step += 1
            if rank == 0:
                print(
                    f"epoch={epoch} step={global_step} loss={float(loss.detach()):.5f} "
                    f"cloud={float(distribution_loss.detach()):.5f}"
                )
            if args.max_steps is not None and global_step >= args.max_steps:
                break

        if rank == 0:
            checkpoint = {
                "cloud": cloud.state_dict(),
                "condition_encoder": condition_encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": config,
                "vocabulary": tokenizer.id_to_token,
            }
            torch.save(checkpoint, args.output / "latest.pt")
        if args.max_steps is not None and global_step >= args.max_steps:
            break
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
