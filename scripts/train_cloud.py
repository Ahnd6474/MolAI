"""Train the SMILES-conditioned molecular Cloud Matching model."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from molai.data import FieldShardDataset, collate_field_batch
from molai.models.bridge import VPSchedule
from molai.models.cloud import MolecularCloudModel
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
    parser.add_argument("--smiles-weight", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = FieldShardDataset(args.data)
    expected_channels = int(config["model"]["field_channels"])
    actual_channels = len(dataset.manifest.get("field_channels", ["signed_charge"]))
    if actual_channels != expected_channels:
        raise ValueError(
            f"dataset has {actual_channels} field channels but model expects "
            f"{expected_channels}; select the matching model config"
        )
    tokenizer = SmilesTokenizer.from_smiles(dataset.iter_smiles())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_field_batch, tokenizer=tokenizer),
    )

    model_config = config["model"]
    condition_dim = int(model_config["condition_dim"])
    cloud = MolecularCloudModel(
        smiles_vocab_size=len(tokenizer),
        smiles_pad_token_id=tokenizer.pad_id,
        condition_dim=condition_dim,
        field_channels=int(model_config["field_channels"]),
        dim=int(model_config["fullres_dim"]),
        heads=int(model_config["heads"]),
        condition_cross_depth=int(model_config["condition_cross_depth"]),
        noise_cross_depth=int(model_config["noise_cross_depth"]),
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
    cloud_loss = FullBandEnergyDistance(levels=int(config["cloud_matching"]["full_band_levels"]))

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "vocabulary.json").write_text(
        json.dumps(tokenizer.id_to_token, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    global_step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            clean = batch["field"].to(device)
            token_ids = batch["token_ids"].to(device)
            transition = schedule.sample_training_batch(clean, cloud_samples)
            condition = condition_encoder(token_ids)
            keep = torch.rand(condition.shape[0], 1, device=device) >= args.condition_dropout
            condition = condition * keep

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = cloud(
                    transition.current,
                    condition,
                    samples=cloud_samples,
                    smiles_input_ids=token_ids[:, :-1],
                )
                distribution_loss = cloud_loss(
                    output.fields, transition.target_cloud, transition.current
                )
                target = token_ids[:, None, 1:].expand(-1, cloud_samples, -1)
                smiles_loss = F.cross_entropy(
                    output.smiles_logits.flatten(0, 1).transpose(1, 2),
                    target.flatten(0, 1),
                    ignore_index=tokenizer.pad_id,
                )
                loss = distribution_loss + args.smiles_weight * smiles_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            global_step += 1
            print(
                f"epoch={epoch} step={global_step} loss={float(loss.detach()):.5f} "
                f"cloud={float(distribution_loss.detach()):.5f} "
                f"smiles={float(smiles_loss.detach()):.5f}"
            )
            if args.max_steps is not None and global_step >= args.max_steps:
                break

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


if __name__ == "__main__":
    main()
