"""Align replacement MS tokens to a frozen relative one-step teacher."""

from __future__ import annotations

import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from train_absolute_cloud import RaggedSpectrumTokenRunner, _condition_tokens
from train_cloud import RaggedSpectrumConditionRunner, _condition_from_batch, _update_latest

from molai.models.relative_rollout import build_relative_models


def alignment_objective(student, teacher, tokens, token_mask, teacher_condition, initial, noise):
    """Same input and random tokens isolate changes in the conditioning path."""
    with torch.no_grad():
        expected = teacher(initial, teacher_condition, samples=noise.shape[1], noise=noise)[0]
    actual = student(initial, tokens, samples=noise.shape[1], noise=noise,
                     condition_mask=token_mask)[0]
    return F.mse_loss(actual.float(), expected.float()), actual, expected


def run_alignment(cloud, encoder, source, loader, val_loader, schedule, options,
                  output, device, rank, world, local, *, prepare_only=False, benchmark_batches=3):
    """Freeze all legacy image modules; train only new MS/cross, then unfreeze."""
    teacher, teacher_encoder = build_relative_models(source, encoder.metadata_embedding[0].in_features)
    teacher, teacher_encoder = teacher.to(device).eval(), teacher_encoder.to(device).eval()
    teacher.requires_grad_(False)
    teacher_encoder.requires_grad_(False)
    teacher_condition_runner = RaggedSpectrumConditionRunner(teacher_encoder)
    cloud.requires_grad_(False)
    cloud.condition_blocks.requires_grad_(True)
    encoder.requires_grad_(True)
    student_condition_module = RaggedSpectrumTokenRunner(encoder)
    student = DDP(cloud, device_ids=[local], static_graph=True) if world > 1 else cloud
    condition_runner = DDP(student_condition_module, device_ids=[local]) if world > 1 else student_condition_module
    parameters = [p for p in (*cloud.parameters(), *encoder.parameters()) if p.requires_grad]
    iterations = benchmark_batches if prepare_only else int(options["alignment_steps"])
    optimizer = None if prepare_only else torch.optim.AdamW(
        parameters, lr=float(options["alignment_learning_rate"]),
        weight_decay=float(options["weight_decay"]), fused=device.type == "cuda",
    )
    writer = SummaryWriter(output / "tensorboard") if rank == 0 and not prepare_only else None
    measurements = []
    student.train()
    condition_runner.train()
    iterator = iter(loader)
    for step in range(iterations):
        try:
            batch = next(iterator)
        except StopIteration:
            loader.sampler.set_epoch(step // max(1, len(loader)) + 1)
            iterator = iter(loader)
            batch = next(iterator)
        for parameter in parameters:
            parameter.grad = None
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        clean = batch["field"].to(device, non_blocking=True)
        levels = torch.randint(0, len(schedule.alpha_bar), (len(clean),), device=device)
        initial = schedule.sample_training_batch(clean, 1, current_levels=levels, answer_jump=0).current
        noise = torch.randn(len(clean), 5, cloud.noise_token_count, cloud.noise_token_dim, device=device)
        noise[:, 0].zero_()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            with torch.no_grad():
                teacher_condition = _condition_from_batch(teacher_condition_runner, batch, device)
            tokens, mask = _condition_tokens(condition_runner, batch, device)
            loss, actual, expected = alignment_objective(
                student, teacher, tokens, mask, teacher_condition, initial, noise,
            )
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(grad):
            raise RuntimeError("non-finite alignment loss/gradient")
        if any(p.grad is not None for name, p in cloud.named_parameters()
               if not name.startswith("condition_blocks.")):
            raise RuntimeError("frozen backbone received an alignment gradient")
        if optimizer is not None:
            progress = step / max(1, iterations - 1)
            rate = float(options["alignment_min_learning_rate"]) + (
                float(options["alignment_learning_rate"]) - float(options["alignment_min_learning_rate"])
            ) * (1 + math.cos(math.pi * progress)) / 2
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        measurement = {"step": step + 1, "mse": float(loss.detach()), "grad_norm": float(grad),
                       "seconds": time.perf_counter() - begin,
                       "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0,
                       "optimizer_updates": 0 if prepare_only else step + 1}
        measurements.append(measurement)
        if rank == 0 and (prepare_only or (step + 1) % int(options["log_every"]) == 0):
            print("alignment " + json.dumps(measurement), flush=True)
        if writer and (step + 1) % int(options["log_every"]) == 0:
            for key in ("mse", "grad_norm"):
                writer.add_scalar(f"alignment/{key}", measurement[key], step + 1)
            writer.add_scalar("alignment/learning_rate", optimizer.param_groups[0]["lr"], step + 1)
            writer.flush()
        del actual, expected, loss, tokens, teacher_condition
        if not prepare_only and ((step + 1) % 250 == 0 or step + 1 == iterations):
            # A held-out paired one-step check isolates conditioning transfer.
            student.eval()
            condition_runner.eval()
            total, count = 0.0, 0
            with torch.no_grad(), torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                torch.manual_seed(8100 + rank)
                for index, batch_val in enumerate(val_loader):
                    if index >= 4:
                        break
                    clean_val = batch_val["field"].to(device, non_blocking=True)
                    levels_val = torch.randint(0, len(schedule.alpha_bar), (len(clean_val),), device=device)
                    input_val = schedule.sample_training_batch(clean_val, 1, current_levels=levels_val, answer_jump=0).current
                    noise_val = torch.randn(len(clean_val), 5, cloud.noise_token_count, cloud.noise_token_dim, device=device)
                    noise_val[:, 0].zero_()
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        goal_val = _condition_from_batch(teacher_condition_runner, batch_val, device)
                        tokens_val, mask_val = _condition_tokens(condition_runner, batch_val, device)
                        val_loss, _, _ = alignment_objective(student, teacher, tokens_val, mask_val, goal_val, input_val, noise_val)
                    total += float(val_loss) * len(clean_val)
                    count += len(clean_val)
            statistics = torch.tensor([total, count], device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(statistics)
            val_mse = float(statistics[0] / statistics[1].clamp_min(1))
            if rank == 0:
                path = output / f"alignment-{step + 1:04d}.pt"
                temporary = path.with_suffix(".tmp")
                torch.save({"cloud": cloud.state_dict(), "condition_encoder": encoder.state_dict(),
                            "alignment_step": step + 1, "validation_mse": val_mse,
                            "optimizer": optimizer.state_dict(),
                            "source_config": source["config"], "options": options,
                            "training_kind": "relative-token-condition-alignment"}, temporary)
                os.replace(temporary, path)
                _update_latest(path, output / "alignment-latest.pt")
                writer.add_scalar("alignment/validation_mse", val_mse, step + 1)
                writer.flush()
                print(f"alignment validation step={step + 1} mse={val_mse:.6f} saved={path}", flush=True)
            student.train()
            condition_runner.train()
    if rank == 0:
        (output / "alignment_benchmark.json").write_text(json.dumps(measurements, indent=2), encoding="utf-8")
    if writer:
        writer.close()
    for parameter in (*cloud.parameters(), *encoder.parameters()):
        parameter.grad = None
    del optimizer, student, condition_runner, teacher, teacher_encoder, teacher_condition_runner
    cloud.requires_grad_(True)
    cloud.readout.requires_grad_(False)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if world > 1:
        dist.barrier()
