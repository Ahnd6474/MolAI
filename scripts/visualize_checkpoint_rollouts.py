"""Paired, unranked L64 rollout images before and after focused fine-tuning."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from rollout_alternating_zero import tile
from train_absolute_cloud import RaggedSpectrumTokenRunner, _condition_tokens
from train_cloud import _make_schedule

from molai.data import collate_spectrum_field_batch
from molai.models.relative_rollout import RelativeImageRolloutCloud, restore_token_relative_models


def draw_grid(rows, labels, titles, destination):
    side, left, top = 128, 100, 28
    canvas = Image.new("RGB", (left + side * len(titles), top + side * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for col, title in enumerate(titles):
        draw.text((left + side * col + 3, 7), title, fill="black")
    for index, (fields, scale) in enumerate(rows):
        draw.text((4, top + index * side + 43), labels[index], fill="black")
        for col, field in enumerate(fields):
            canvas.paste(tile(field, scale), (left + col * side, top + index * side))
    canvas.save(destination)


@torch.no_grad()
def infer(checkpoint, records, manifest, seed, steps):
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cloud, encoder, audit = restore_token_relative_models(
        source, int(manifest["metadata_dim"]), gradient_checkpointing=False)
    config = source["config"]
    info = {"checkpoint": str(checkpoint.resolve()), "epoch": source["epoch"] + 1,
            "step": source["global_step"], "restore": audit,
            "model_config": config["model"], "spectrum_config": config["spectrum_encoder"]}
    del source
    cloud, encoder = cloud.cuda().eval(), encoder.cuda().eval()
    options = config["relative_image_rollout"]
    cloud.cache_condition_kv = bool(options.get("cache_ms_kv", False))
    cloud.packed_condition_kv = bool(options.get("packed_ms_kv", False))
    runner = RaggedSpectrumTokenRunner(encoder).eval()
    rollout = RelativeImageRolloutCloud(cloud).eval()
    schedule = _make_schedule(manifest, config, torch.device("cuda"))
    assert float(schedule.alpha_bar[64]) == 0
    versions = [p._version for model in (cloud, encoder) for p in model.parameters()]
    frames = {}
    for record in records:
        index = record["index"]
        batch = collate_spectrum_field_batch([record["item"]], peak_chunk_size=256)
        clean = batch["field"].cuda()
        rng = torch.Generator(device="cuda").manual_seed(seed + index)
        epsilon = torch.randn(clean.shape, device="cuda", generator=rng)
        if schedule.zero_mean_noise:
            epsilon -= epsilon.mean((-2, -1), keepdim=True)
        initial = schedule.noise_scale * epsilon
        noise = torch.randn(1, steps, 5, cloud.noise_token_count, cloud.noise_token_dim,
                            device="cuda", generator=rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            condition, mask = _condition_tokens(runner, batch, torch.device("cuda"))
            fields = rollout(initial, condition, condition_mask=mask, steps=steps,
                             random_samples=4, noise=noise)
        assert torch.isfinite(fields).all()
        selected = [s for s in (1, 2, 4, 8, 12, 16, steps) if s <= steps]
        frames[index] = {"clean": clean[0].cpu(), "initial": initial[0].cpu(),
                         "selected_steps": selected,
                         "fields": fields[0, [s-1 for s in selected]].float().cpu()}
        print(json.dumps({"checkpoint_step": info["step"], "molecule": index, "done": True}), flush=True)
    assert versions == [p._version for model in (cloud, encoder) for p in model.parameters()]
    del cloud, encoder, runner, rollout, fields, clean, condition, mask
    gc.collect()
    torch.cuda.empty_cache()
    return frames, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=923001)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(args.samples, map_location="cpu", weights_only=False)
    records = [r for r in bundle["samples"] if r["split"] == "validation"][:args.count]
    started = time.perf_counter()
    old, old_info = infer(args.before, records, bundle["manifest"], args.seed, 23)
    new, new_info = infer(args.after, records, bundle["manifest"], args.seed, 23)
    assert old_info["model_config"] == new_info["model_config"]
    assert old_info["spectrum_config"] == new_info["spectrum_config"]
    for offset in range(0, len(records), 4):
        rows, labels, path_rows, path_labels = [], [], [], []
        for record in records[offset:offset+4]:
            index = record["index"]
            a, b = old[index], new[index]
            torch.testing.assert_close(a["initial"], b["initial"], rtol=0, atol=0)
            scale = float(a["clean"].abs().quantile(.995).clamp_min(1e-6))
            rows.append(([a["clean"], a["fields"][-1, 1], b["fields"][-1, 1]], scale))
            labels.append(f"Mol {index}\nRandom 1")
            for name, row in (("Before", a), ("After 126k", b)):
                path_rows.append(([a["clean"], *list(row["fields"][-1])], scale))
                path_labels.append(f"Mol {index}\n{name}")
        draw_grid(rows, labels, ["GT", "Before", "After 126k"],
                  args.output / f"overview_{offset//4+1}.png")
        draw_grid(path_rows, path_labels,
                  ["GT", "Zero", "Random 1", "Random 2", "Random 3", "Random 4"],
                  args.output / f"all_paths_{offset//4+1}.png")
    for record in records:
        index = record["index"]
        a, b = old[index], new[index]
        scale = float(a["clean"].abs().quantile(.995).clamp_min(1e-6))
        titles = ["GT", "Zero", "Random 1", "Random 2", "Random 3", "Random 4"]
        draw_grid([([a["clean"], *list(row["fields"][-1])], scale) for row in (a, b)],
                  ["Before", "After 126k"], titles, args.output / f"paths_{index}.png")
        rows, labels = [], []
        for path in (1, 2):
            for name, row in (("Before", a), ("After 126k", b)):
                rows.append(([a["clean"], a["initial"], *list(row["fields"][:, path])], scale))
                labels.append(f"{name}\nRandom {path}")
        draw_grid(rows, labels, ["GT", "Input L64", *[f"Step {s}" for s in a["selected_steps"]]],
                  args.output / f"trajectory_{index}.png")
    torch.save({"before": old, "after": new}, args.output / "frames.pt")
    report = {"before": old_info, "after": new_info, "count": len(records), "seed": args.seed,
              "molecules": [{"index": r["index"], "smiles": r["item"]["smiles"]} for r in records],
              "protocol": "First eligible validation records, unranked. Same MS, centered image noise, "
                          "and per-step Gaussian tokens. L64 pure noise, 23 model calls, Zero+4 Random "
                          "paths. Shared GT 99.5th percentile color scale per molecule, no variance "
                          "adjustment, no output clipping before display, no extra projection.",
              "device": torch.cuda.get_device_name(), "elapsed_seconds": time.perf_counter()-started,
              "parameters_unchanged": True}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"complete": True, "seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
