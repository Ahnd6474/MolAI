"""Paired forward/backward benchmarks; never update training weights."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from train_absolute_cloud import RaggedSpectrumTokenRunner, _condition_tokens
from train_cloud import _make_schedule
from train_relative_image_rollout import rollout_losses, trajectory_targets

from molai.data import SpectrumFieldDataset, collate_spectrum_field_batch
from molai.models.losses import FullBandEnergyDistance
from molai.models.relative_rollout import RelativeImageRolloutCloud, build_token_relative_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--data", type=Path, help="use real clean fields and the aligned MS encoder")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--checkpointing", choices=("all", "none"), default="all")
    parser.add_argument("--uncheckpointed-refine", type=int, default=0)
    parser.add_argument("--compile-blocks", action="store_true")
    parser.add_argument("--compile-pointwise", action="store_true")
    parser.add_argument("--compile-loss", action="store_true")
    parser.add_argument("--cache-ms-kv", action="store_true")
    parser.add_argument("--packed-ms-kv", action="store_true")
    parser.add_argument("--verify-eager", action="store_true")
    parser.add_argument("--preserve-bf16-rounding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(1729)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata_dim = source["condition_encoder"]["metadata_embedding.0.weight"].shape[1]
    cloud, encoder, _ = build_token_relative_models(source, metadata_dim,
        gradient_checkpointing=args.checkpointing == "all")
    config = source["config"]
    del source
    if args.warm_start:
        aligned = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        cloud.load_state_dict(aligned["cloud"], strict=True)
        if args.data:
            encoder.load_state_dict(aligned["condition_encoder"], strict=True)
        del aligned
    cloud = cloud.cuda().train()
    cloud.cache_condition_kv = args.cache_ms_kv
    cloud.packed_condition_kv = args.packed_ms_kv
    compile_options = {"emulate_precision_casts": args.preserve_bf16_rounding}

    def compile_call(function):
        return torch.compile(function, fullgraph=True, dynamic=True, options=compile_options)
    def enable_compilation():
        # Keep state_dict names and module objects unchanged. Shared block code
        # compiles once per shape rather than unrolling all four time steps.
        if args.compile_blocks:
            cloud.field_encoder.forward = compile_call(cloud.field_encoder.forward)
            for module in (*cloud.condition_blocks, cloud.noise_attention, *cloud.refine_blocks):
                module.forward = compile_call(module.forward)
            if args.cache_ms_kv:
                for module in cloud.condition_blocks:
                    module.forward_projected = compile_call(module.forward_projected)
                    module.forward_varlen = compile_call(module.forward_varlen)
        elif args.compile_pointwise:
            for block in (*cloud.condition_blocks, *cloud.refine_blocks):
                block.ff.forward = compile_call(block.ff.forward)
    rollout = RelativeImageRolloutCloud(cloud)
    batch = args.batch
    initial = torch.randn(batch, 1, 128, 128, device="cuda")
    clean = torch.randn_like(initial)
    condition = torch.randn(batch, args.tokens, 64, device="cuda", requires_grad=True)
    mask = torch.ones(batch, args.tokens, dtype=torch.bool, device="cuda")
    for row in range(batch):
        mask[row, max(1, args.tokens - row * args.tokens // (2 * batch)):] = False
    noise = torch.randn(batch, 4, 5, 64, 64, device="cuda")
    targets = torch.randn(batch, 4, 4, 1, 128, 128, device="cuda")
    if args.data:
        dataset = SpectrumFieldDataset(args.data)
        records = collate_spectrum_field_batch([dataset[i] for i in range(batch)], peak_chunk_size=256)
        encoder = encoder.cuda().eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            condition, mask = _condition_tokens(RaggedSpectrumTokenRunner(encoder), records, torch.device("cuda"))
        condition = condition.detach().requires_grad_()
        clean = records["field"].cuda()
        schedule = _make_schedule(dataset.manifest, config, torch.device("cuda"))
        levels = torch.randint(0, len(schedule.alpha_bar), (batch,), device="cuda")
        initial, targets, _ = trajectory_targets(schedule, clean, levels, 4, 4, config)
    del encoder
    distance = FullBandEnergyDistance(include_target_constant=False, unbiased=True)
    objective = rollout_losses

    def step():
        cloud.zero_grad(set_to_none=True)
        condition.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fields = rollout(initial, condition, condition_mask=mask, steps=4, noise=noise)
            loss = objective(fields, targets, initial, clean, distance, 0.05)["total"]
        loss.backward()
        return fields.detach(), loss.detach()

    reference = None
    if args.verify_eager:
        cloud.cache_condition_kv = False
        fields, loss = step()
        reference = (fields.float().cpu(), condition.grad.float().cpu(),
                     torch.cat([p.grad.float().flatten().cpu() for p in cloud.parameters() if p.grad is not None]))
        cloud.cache_condition_kv = args.cache_ms_kv
    enable_compilation()
    cloud.uncheckpointed_refine_blocks = args.uncheckpointed_refine
    if args.compile_loss:
        objective = compile_call(rollout_losses)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    fields, loss = step()
    torch.cuda.synchronize()
    first = time.perf_counter() - start
    reference_fields = fields.float().cpu()
    reference_grad = condition.grad.float().cpu()
    parity = {}
    if reference is not None:
        parameter_grad = torch.cat([p.grad.float().flatten().cpu() for p in cloud.parameters() if p.grad is not None])
        parity = {"eager_field_relative_rms": float((reference_fields - reference[0]).norm() / reference[0].norm().clamp_min(1e-12)),
                  "eager_step_relative_rms": [float((reference_fields[:, i] - reference[0][:, i]).norm() / reference[0][:, i].norm().clamp_min(1e-12)) for i in range(4)],
                  "eager_ms_gradient_relative_rms": float((reference_grad - reference[1]).norm() / reference[1].norm().clamp_min(1e-12)),
                  "eager_parameter_gradient_relative_rms": float((parameter_grad - reference[2]).norm() / reference[2].norm().clamp_min(1e-12))}
    for _ in range(args.warmup):
        step()
    timings = []
    for _ in range(args.iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fields, loss = step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    print(json.dumps({"batch": batch, "tokens": condition.shape[1], "real_data": bool(args.data), "checkpointing": args.checkpointing,
        "compile_blocks": args.compile_blocks, "compile_loss": args.compile_loss,
        "compile_pointwise": args.compile_pointwise,
        "uncheckpointed_refine_blocks": args.uncheckpointed_refine,
        "cache_ms_kv": args.cache_ms_kv,
        "packed_ms_kv": args.packed_ms_kv, **parity,
        "first_seconds": first, "median_seconds": statistics.median(timings),
        "molecules_per_second_one_gpu": batch / statistics.median(timings),
        "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "loss": float(loss), "zero_mean_error": float(fields.float().mean(dim=(-2, -1)).abs().max()),
        "deterministic_field_error": float((fields.cpu().float() - reference_fields).abs().max()),
        "deterministic_gradient_error": float((condition.grad.cpu().float() - reference_grad).abs().max())}), flush=True)
    if args.profile:
        gc.collect()
        torch.cuda.empty_cache()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profiler:
            step()
            torch.cuda.synchronize()
        print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=30), flush=True)


if __name__ == "__main__":
    main()
