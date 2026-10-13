"""Replace only the first L64->L56 output; leave subsequent inference unchanged."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diagnose_midstart_ms_ablation import measures
from PIL import Image, ImageDraw
from rollout_alternating_zero import tile
from train_absolute_cloud import RaggedSpectrumTokenRunner, _condition_tokens
from train_cloud import _make_schedule, _transition_jumps

from molai.data import collate_spectrum_field_batch
from molai.models.relative_rollout import restore_token_relative_models


@torch.no_grad()
def oracle_first_state(schedule, clean, initial, *, next_level, paths, generator):
    """Zero is the exact posterior mean; Random paths are independent VP samples."""
    start = torch.full((len(clean),), 64, device=clean.device, dtype=torch.long)
    after = torch.full_like(start, next_level)
    if float(schedule.alpha_bar[64]) != 0:
        raise ValueError("This experiment assumes exactly zero signal at L64")
    mean, variance = schedule.target_mean_and_variance(clean, initial, start, after)
    epsilon = torch.randn(len(clean), paths - 1, *clean.shape[1:], device=clean.device,
                          dtype=clean.dtype, generator=generator)
    if schedule.zero_mean_noise:
        epsilon -= epsilon.mean((-2, -1), keepdim=True)
    random = mean[:, None] + schedule.noise_scale * variance.sqrt()[:, None, None, None, None] * epsilon
    return torch.cat((mean[:, None], random), dim=1)


@torch.no_grad()
def paired_rollout(cloud, initial, condition, mask, tokens, first_replacement=None):
    """Persistent Zero+Random paths; token index and all later calls are identical."""
    if cloud.cache_condition_kv:
        raise ValueError("Diagnostic uses uncached conditioning to keep the interface explicit")
    batch, steps, paths = tokens.shape[:3]
    if first_replacement is not None and first_replacement.shape != (batch, paths, *initial.shape[1:]):
        raise ValueError("Replacement must have shape [B,paths,C,H,W]")
    tokens = tokens.clone()
    tokens[:, :, 0] = 0
    expanded_condition = condition.repeat_interleave(paths, 0)
    expanded_mask = mask.repeat_interleave(paths, 0)
    frames = []
    for step in range(steps):
        if step == 0:
            if first_replacement is None:
                current = cloud(initial, condition, samples=paths, noise=tokens[:, step],
                                condition_mask=mask)[0]
            else:
                current = first_replacement
        else:
            fields = cloud(current.flatten(0, 1), expanded_condition, samples=1,
                           noise=tokens[:, step].reshape(batch * paths, 1, *tokens.shape[-2:]),
                           condition_mask=expanded_mask)[0]
            current = fields[:, 0].reshape(batch, paths, *initial.shape[1:])
        if not torch.isfinite(current).all():
            raise RuntimeError(f"Non-finite output at step {step + 1}")
        frames.append(current.float())
    return torch.stack(frames, 1)


def render_comparison(stored, output, path):
    canvas = Image.new("RGB", (992, 32 + 128 * len(stored)), "white")
    draw = ImageDraw.Draw(canvas)
    titles = ["GT", "Model / s1", "Oracle / s1", "Model / s4", "Oracle / s4",
              "Model / s20", "Oracle / s20"]
    for col, title in enumerate(titles):
        draw.text((96 + col * 128, 9), title, fill="black")
    for row, (index, data) in enumerate(stored.items()):
        scale = float(data["clean"].abs().quantile(.995).clamp_min(1e-6))
        panels = [data["clean"]] + [data[mode][step - 1, path]
                   for step in (1, 4, 20) for mode in ("baseline", "oracle")]
        for col, field in enumerate(panels):
            canvas.paste(tile(field, scale), (96 + col * 128, 32 + row * 128))
        draw.text((3, 83 + row * 128), f"Mol {index}", fill="black")
    canvas.save(output / ("comparison-zero.png" if path == 0 else f"comparison-random{path}.png"))


def render_evolution(index, data, output):
    selected = (1, 2, 4, 8, 12, 16, 20)
    canvas = Image.new("RGB", (1136, 544), "white")
    draw = ImageDraw.Draw(canvas)
    for col, title in enumerate(["GT", *[f"Step {s}" for s in selected]]):
        draw.text((112 + col * 128, 9), title, fill="black")
    scale = float(data["clean"].abs().quantile(.995).clamp_min(1e-6))
    for row, (mode, path) in enumerate((("baseline", 0), ("oracle", 0),
                                       ("baseline", 1), ("oracle", 1))):
        panels = [data["clean"], *[data[mode][s - 1, path] for s in selected]]
        for col, field in enumerate(panels):
            canvas.paste(tile(field, scale), (112 + col * 128, 32 + row * 128))
        draw.text((3, 83 + row * 128), f"{mode}\n{'Zero' if path == 0 else 'Random 1'}", fill="black")
    canvas.save(output / f"evolution-{index}.png")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    bundle = torch.load(args.samples, map_location="cpu", weights_only=False)
    records = [r for r in bundle["samples"] if r["split"] == "validation"]
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cloud, encoder, audit = restore_token_relative_models(
        source, int(bundle["manifest"]["metadata_dim"]), gradient_checkpointing=False)
    config = source["config"]
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "epoch": source["epoch"] + 1,
        "global_step": source["global_step"], "restore_audit": audit,
        "device": torch.cuda.get_device_name(), "steps": 20, "random_paths": 4,
        "protocol": "Baseline: 20 model calls. Oracle: first output replaced, then 19 identical "
                    "model calls with the same per-step random tokens and MS. Zero replacement "
                    "is sqrt(alpha56)*GT; Random replacements independently sample the exact "
                    "L64->L56 posterior, equal to the L56 marginal. No later GT injection, "
                    "variance adjustment, clipping or extra projection. This is an oracle "
                    "diagnostic with target leakage, not unconditional generation quality.",
        "molecules": {},
    }
    del source
    cloud, encoder = cloud.to(device).eval(), encoder.to(device).eval()
    cloud.cache_condition_kv = False
    cloud.packed_condition_kv = False
    runner = RaggedSpectrumTokenRunner(encoder).eval()
    schedule = _make_schedule(bundle["manifest"], config, device)
    next_level = int((torch.tensor([64], device=device) - _transition_jumps(
        torch.tensor([64], device=device), config))[0])
    if next_level != 56:
        raise ValueError(f"Expected first transition to L56, got L{next_level}")
    levels, level = [], torch.tensor([64], device=device)
    for _ in range(20):
        level = (level - _transition_jumps(level, config)).clamp_min(0)
        levels.append(int(level[0]))
    report["nominal_target_levels"] = levels
    report["first_signal_amplitude"] = float(schedule.alpha_bar[56].sqrt())
    report["first_noise_scale"] = float(schedule.noise_scale * (1-schedule.alpha_bar[56]).sqrt())
    parameter_versions = [p._version for p in (*cloud.parameters(), *encoder.parameters())]
    started, stored = time.perf_counter(), {}
    for record in records:
        index = record["index"]
        batch = collate_spectrum_field_batch([record["item"]], peak_chunk_size=256)
        clean = batch["field"].to(device)
        generator = torch.Generator(device=device).manual_seed(923001 + index)
        epsilon = torch.randn(clean.shape, device=device, generator=generator)
        epsilon -= epsilon.mean((-2, -1), keepdim=True)
        initial = epsilon * schedule.noise_scale
        tokens = torch.randn(1, 20, 5, cloud.noise_token_count, cloud.noise_token_dim,
                             device=device, generator=generator)
        oracle_generator = torch.Generator(device=device).manual_seed(856001 + index)
        replacement = oracle_first_state(schedule, clean, initial, next_level=56, paths=5,
                                         generator=oracle_generator)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            condition, mask = _condition_tokens(runner, batch, device)
            baseline = paired_rollout(cloud, initial, condition, mask, tokens)
            oracle = paired_rollout(cloud, initial, condition, mask, tokens, replacement)
        torch.testing.assert_close(oracle[:, 0], replacement.float(), rtol=0, atol=0)
        stored[index] = {"clean": clean[0].cpu(), "initial": initial[0].cpu(),
                         "baseline": baseline[0].cpu(), "oracle": oracle[0].cpu()}
        report["molecules"][str(index)] = {
            "smiles": record["item"]["smiles"],
            **{mode: measures(stored[index][mode], stored[index]["clean"])
               for mode in ("baseline", "oracle")},
        }
        print(json.dumps({"molecule": index, "elapsed_seconds": time.perf_counter()-started,
                          **{mode: {key: report["molecules"][str(index)][mode][key][-1]
                                    for key in ("random_correlation", "zero_correlation", "random_mse")}
                             for mode in ("baseline", "oracle")}}), flush=True)
        render_evolution(index, stored[index], args.output)
        del baseline, oracle, condition, mask
    report["summary"] = {
        mode: {key: [sum(row[mode][key][step] for row in report["molecules"].values()) / len(records)
                     for step in range(20)]
               for key in next(iter(report["molecules"].values()))[mode]}
        for mode in ("baseline", "oracle")
    }
    assert parameter_versions == [p._version for p in (*cloud.parameters(), *encoder.parameters())]
    report.update(parameters_unchanged=True, elapsed_seconds=time.perf_counter()-started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), count=len(records))
    for path in range(5):
        render_comparison(stored, args.output, path)
    torch.save(stored, args.output / "frames.pt")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("First-transition diagnostic complete", flush=True)


if __name__ == "__main__":
    main()
