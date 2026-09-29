"""Measure field collisions and train a reproducible image-to-SMILES probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from molai.dft.dataset import iter_structure_records
from molai.fields import ExpectedCharge2D, ExpectedChargeConfig
from molai.models import FieldToSmiles, SmilesTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--render-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-token-length", type=int, default=192)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("artifacts/electron_cloud_eval.json"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def canonicalize(value: str) -> str | None:
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def render_fields(
    input_path: Path,
    limit: int,
    resolution: int,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, list[str], int]:
    renderer = ExpectedCharge2D(ExpectedChargeConfig(resolution=resolution), device)
    fields: list[torch.Tensor] = []
    smiles: list[str] = []
    seen: set[str] = set()
    failures = 0
    records = list(islice(iter_structure_records(input_path), limit))
    progress = tqdm(total=len(records), desc="render fields")
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        try:
            result = renderer.render_batch([record.smiles for record in batch])
        except (RuntimeError, ValueError):
            for record in batch:
                try:
                    result = renderer.render_batch([record.smiles])
                except (RuntimeError, ValueError):
                    failures += 1
                    continue
                canonical = result.canonical_smiles[0]
                if canonical not in seen:
                    seen.add(canonical)
                    fields.append(result.field[0].detach().cpu().to(torch.float16))
                    smiles.append(canonical)
            progress.update(len(batch))
            continue
        batch_fields = result.field.detach().cpu().to(torch.float16)
        for index, canonical in enumerate(result.canonical_smiles):
            if canonical in seen:
                continue
            seen.add(canonical)
            fields.append(batch_fields[index])
            smiles.append(canonical)
        progress.update(len(batch))
    progress.close()
    if not fields:
        raise RuntimeError("no molecular fields could be rendered")
    return torch.stack(fields), smiles, failures


def collision_metrics(fields: torch.Tensor, smiles: list[str]) -> dict[str, object]:
    hashes: defaultdict[str, list[int]] = defaultdict(list)
    quantized = ((fields.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    for index, image in enumerate(quantized):
        digest = hashlib.sha256(image.numpy().tobytes()).hexdigest()
        hashes[digest].append(index)
    colliding = [indices for indices in hashes.values() if len(indices) > 1]
    exact_pairs = sum(len(indices) * (len(indices) - 1) // 2 for indices in colliding)

    pooled = functional.adaptive_avg_pool2d(fields.float(), (32, 32)).flatten(1).numpy()
    if len(pooled) < 2:
        nearest_rmse = np.asarray([], dtype=np.float32)
        nearest_indices = np.asarray([], dtype=np.int64)
    else:
        neighbors = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(pooled)
        distances, indices = neighbors.kneighbors(pooled)
        nearest_rmse = distances[:, 1] / math.sqrt(pooled.shape[1])
        nearest_indices = indices[:, 1]
    thresholds = (0.0025, 0.005, 0.010, 0.020)
    examples = []
    if len(nearest_rmse):
        for index in np.argsort(nearest_rmse)[:10]:
            other = int(nearest_indices[index])
            examples.append({
                "smiles_a": smiles[int(index)],
                "smiles_b": smiles[other],
                "rmse": float(nearest_rmse[index]),
            })
    return {
        "quantization_bits": 8,
        "exact_collision_groups": len(colliding),
        "exact_collision_pairs": exact_pairs,
        "exact_collision_rate": exact_pairs / max(len(smiles) * (len(smiles) - 1) / 2, 1),
        "nearest_neighbor_rmse_mean": (
            float(nearest_rmse.mean()) if len(nearest_rmse) else None
        ),
        "nearest_neighbor_rmse_median": (
            float(np.median(nearest_rmse)) if len(nearest_rmse) else None
        ),
        "near_collision_fraction": {
            str(threshold): float((nearest_rmse < threshold).mean()) if len(nearest_rmse) else 0.0
            for threshold in thresholds
        },
        "closest_distinct_examples": examples,
    }


def encoded_targets(
    tokenizer: SmilesTokenizer,
    smiles: list[str],
    max_length: int,
) -> tuple[torch.Tensor, list[int]]:
    accepted: list[list[int]] = []
    accepted_indices: list[int] = []
    for index, value in enumerate(smiles):
        encoded = tokenizer.encode(value)
        if len(encoded) <= max_length:
            accepted.append(encoded)
            accepted_indices.append(index)
    targets = torch.full(
        (len(accepted), max_length), tokenizer.pad_id, dtype=torch.long
    )
    for row, encoded in enumerate(accepted):
        targets[row, : len(encoded)] = torch.tensor(encoded)
    return targets, accepted_indices


def evaluate_recovery(
    model: FieldToSmiles,
    loader: DataLoader,
    tokenizer: SmilesTokenizer,
    target_smiles: list[str],
    device: torch.device,
    max_length: int,
) -> dict[str, float]:
    model.eval()
    predictions: list[str] = []
    correct_tokens = 0
    token_count = 0
    validation_loss = 0.0
    with torch.no_grad():
        for fields, targets in loader:
            targets = targets.to(device, non_blocking=True)
            encoded_fields = fields.to(device, non_blocking=True)
            logits = model(encoded_fields, targets[:, :-1])
            expected = targets[:, 1:]
            mask = expected.ne(tokenizer.pad_id)
            correct_tokens += int(logits.argmax(dim=-1).eq(expected).logical_and(mask).sum())
            token_count += int(mask.sum())
            validation_loss += float(functional.cross_entropy(
                logits.flatten(0, 1), expected.flatten(),
                ignore_index=tokenizer.pad_id, reduction="sum",
            ))
            generated = model.generate(
                encoded_fields,
                tokenizer.bos_id,
                tokenizer.eos_id,
                max_length - 1,
            )
            predictions.extend(tokenizer.decode(row.tolist()) for row in generated.cpu())
    valid = 0
    exact = 0
    connectivity = 0
    for prediction, target in zip(predictions, target_smiles, strict=True):
        predicted_canonical = canonicalize(prediction)
        if predicted_canonical is None:
            continue
        valid += 1
        exact += predicted_canonical == target
        predicted_molecule = Chem.MolFromSmiles(predicted_canonical)
        target_molecule = Chem.MolFromSmiles(target)
        if predicted_molecule is not None and target_molecule is not None:
            connectivity += (
                Chem.MolToInchiKey(predicted_molecule).split("-")[0]
                == Chem.MolToInchiKey(target_molecule).split("-")[0]
            )
    total = max(len(target_smiles), 1)
    return {
        "exact_canonical_smiles_rate": exact / total,
        "inchikey14_connectivity_rate": connectivity / total,
        "valid_smiles_rate": valid / total,
        "teacher_forced_token_accuracy": correct_tokens / max(token_count, 1),
        "validation_cross_entropy": validation_loss / max(token_count, 1),
    }


def train_probe(
    fields: torch.Tensor,
    smiles: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], FieldToSmiles, SmilesTokenizer]:
    original_count = len(smiles)
    tokenizer = SmilesTokenizer.from_smiles(smiles)
    targets, accepted_indices = encoded_targets(tokenizer, smiles, args.max_token_length)
    fields = fields[accepted_indices].float()
    smiles = [smiles[index] for index in accepted_indices]
    if len(smiles) < 8:
        raise RuntimeError("at least eight encodable structures are required")

    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(smiles), generator=generator)
    validation_size = max(1, round(len(smiles) * args.validation_fraction))
    validation_indices = permutation[:validation_size]
    train_indices = permutation[validation_size:]
    train_data = TensorDataset(fields[train_indices], targets[train_indices])
    validation_data = TensorDataset(fields[validation_indices], targets[validation_indices])
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda"
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=device.type == "cuda"
    )
    validation_smiles = [smiles[index] for index in validation_indices.tolist()]

    model = FieldToSmiles(len(tokenizer), pad_token_id=tokenizer.pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        model.train()
        loss_total = 0.0
        token_count = 0
        for batch_fields, batch_targets in tqdm(
            train_loader, desc=f"probe epoch {epoch + 1}/{args.epochs}", leave=False
        ):
            batch_fields = batch_fields.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(batch_fields, batch_targets[:, :-1])
                loss = criterion(logits.flatten(0, 1), batch_targets[:, 1:].flatten())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            count = int(batch_targets[:, 1:].ne(tokenizer.pad_id).sum())
            loss_total += float(loss.detach()) * count
            token_count += count
        epoch_metrics = evaluate_recovery(
            model, validation_loader, tokenizer, validation_smiles, device,
            args.max_token_length,
        )
        epoch_metrics["train_cross_entropy"] = loss_total / max(token_count, 1)
        history.append(epoch_metrics)
        print(json.dumps({"epoch": epoch + 1, **epoch_metrics}, ensure_ascii=False))
    return {
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "skipped_overlength": original_count - len(smiles),
        "vocabulary_size": len(tokenizer),
        "history": history,
        "final": history[-1],
    }, model, tokenizer


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    device = torch.device(args.device)

    fields, smiles, failures = render_fields(
        args.input, args.limit, args.resolution, device, args.render_batch_size
    )
    collisions = collision_metrics(fields, smiles)
    recovery, model, tokenizer = train_probe(fields, smiles, args, device)
    report = {
        "input": str(args.input.resolve()),
        "seed": args.seed,
        "device": str(device),
        "rendered_structures": len(smiles),
        "render_failures": failures,
        "resolution": args.resolution,
        "field_config": ExpectedChargeConfig(resolution=args.resolution).to_dict(),
        "representation": "expected_valence_charge",
        "collisions": collisions,
        "image_to_smiles": recovery,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    checkpoint = args.checkpoint or args.output.with_suffix(".pt")
    torch.save({
        "model": model.state_dict(),
        "vocabulary": tokenizer.id_to_token,
        "resolution": args.resolution,
        "field_config": report["field_config"],
    }, checkpoint)
    print(f"report: {args.output.resolve()}")
    print(f"checkpoint: {checkpoint.resolve()}")


if __name__ == "__main__":
    main()
