"""Evaluate x[k+1] = x[k] + Zero(Random(x[k])) locally, without training."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from compare_ustat_navigation_rollout import restore
from PIL import Image, ImageDraw
from rollout_alternating_zero import measure, tile
from train_absolute_cloud import RaggedSpectrumTokenRunner, _condition_tokens
from train_cloud import _make_schedule

from molai.data import collate_spectrum_field_batch


@torch.no_grad()
def accumulated_rollout(cloud, condition, mask, initial, noise):
    """Twenty R/Z pairs; add the full Zero image to the pre-Random state."""
    paths = noise.shape[2]
    current = initial.repeat_interleave(paths, dim=0)
    condition = condition.repeat_interleave(paths, dim=0)
    mask = mask.repeat_interleave(paths, dim=0)
    zeros = torch.zeros_like(noise[0, 0, :, None])
    states, random_outputs, zero_outputs = [], [], []
    for cycle in range(noise.shape[1]):
        before = current
        with torch.autocast(current.device.type, dtype=torch.bfloat16,
                            enabled=current.is_cuda):
            random = cloud(before, condition, samples=1, noise=noise[0, cycle, :, None],
                           condition_mask=mask)[0][:, 0]
            zero = cloud(random, condition, samples=1, noise=zeros,
                         condition_mask=mask)[0][:, 0]
        current = before.float() + zero.float()
        torch.testing.assert_close(current, before.float() + zero.float(), rtol=0, atol=0)
        assert torch.isfinite(current).all(), f"Non-finite state at cycle {cycle+1}"
        states.append(current.cpu())
        random_outputs.append(random.float().cpu())
        zero_outputs.append(zero.float().cpu())
    return {"accumulated": torch.stack(states), "random": torch.stack(random_outputs),
            "zero": torch.stack(zero_outputs)}


def render_evolution(data, destination):
    selected = [1, 2, 4, 8, 12, 16, 20]
    scale = float(data["clean"].abs().flatten().quantile(.995).clamp_min(1e-6))
    canvas = Image.new("RGB", (1248, 1056), "white")
    draw = ImageDraw.Draw(canvas)
    for col, title in enumerate(["GT", "Input", *[f"Cycle {s}" for s in selected]]):
        draw.text((100 + col * 128, 10), title, fill="black")
    for path in range(4):
        for kind_index, kind in enumerate(("accumulated", "zero")):
            row = path * 2 + kind_index
            fields = [data["clean"], data["initial"], *[data[kind][s-1, path] for s in selected]]
            for col, field in enumerate(fields):
                canvas.paste(tile(field, scale), (96 + col * 128, 32 + row * 128))
            draw.text((3, 85 + row * 128), f"P{path+1} {'State x' if kind_index == 0 else 'Addend z'}",
                      fill="black")
    canvas.save(destination)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(args.samples, map_location="cpu", weights_only=False)
    previous = torch.load(args.previous / "frames.pt", map_location="cpu", weights_only=False)
    previous_report = json.loads((args.previous / "report.json").read_text())
    records = [r for r in bundle["samples"] if r["index"] in (33, 88, 122, 177)]
    assert len(records) == 4
    cloud, encoder, metadata = restore(args.checkpoint)
    assert metadata["global_step"] == previous_report["checkpoint"]["global_step"]
    schedule = _make_schedule(bundle["manifest"], metadata["config"], torch.device("cuda"))
    stored = {}
    report = {
        "checkpoint": metadata, "device": torch.cuda.get_device_name(),
        "cycles": 20, "model_calls_per_path": 40, "coefficient": 1.0,
        "equation": "r=Random(x); z=Zero(r); x_next=x+z",
        "note": "Full Zero image added to the state BEFORE Random, not a residual difference. "
                "Local BF16 eval, no training, clipping, variance rescaling or extra mean projection. "
                "4 fixed validation molecules x 4 paths. Identical pure-noise initial state and "
                "matched Random-transition tokens to the previous experiment. "
                "No level embedding or noise schedule applied during updates. "
                "GT 99.5% color scale shared with previous experiment; display clipping only. "
                "20 cycles are 40 model calls, unlike previous 20 alternating calls (10 cycles).",
        "molecules": {},
    }
    started = time.perf_counter()
    for record in records:
        index = record["index"]
        batch = collate_spectrum_field_batch([record["item"]], peak_chunk_size=256)
        clean = batch["field"].cuda()
        torch.manual_seed(923001 + index)
        initial = torch.randn_like(clean)
        initial -= initial.mean((-2, -1), keepdim=True)
        initial *= schedule.noise_scale
        torch.testing.assert_close(initial[0].cpu(), previous[index]["initial"], rtol=0, atol=0)
        noise = torch.randn(1, 20, 4, cloud.noise_token_count, cloud.noise_token_dim, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            condition, mask = _condition_tokens(RaggedSpectrumTokenRunner(encoder), batch, torch.device("cuda"))
        data = accumulated_rollout(cloud, condition, mask, initial, noise)
        data.update(clean=clean[0].cpu(), initial=initial[0].cpu())
        stored[index] = data
        row = {kind: measure(data[kind], data["clean"]) for kind in ("accumulated", "random", "zero")}
        row["clean_rms"] = float(clean.square().mean().sqrt())
        report["molecules"][str(index)] = row
        render_evolution(data, args.output / f"evolution-{index}.png")
        print(json.dumps({"molecule": index, "seconds": time.perf_counter() - started,
                          "final": {kind: {key: row[kind][key][-1] for key in ("rms", "correlation", "mse")}
                                    for kind in ("accumulated", "zero")}}), flush=True)
    report["summary"] = {}
    for label, kind, cycle in (("accumulated10_20calls", "accumulated", 9),
                                ("zero10_20calls", "zero", 9),
                                ("accumulated20_40calls", "accumulated", 19),
                                ("zero20_40calls", "zero", 19)):
        report["summary"][label] = {
            key: sum(row[kind][key][cycle] for row in report["molecules"].values()) / 4
            for key in ("rms", "correlation", "best_correlation", "mse", "best_mse",
                        "diversity_variance", "max_abs_spatial_mean")}
    report["previous_summary"] = previous_report["summary"]
    report["seconds"] = time.perf_counter() - started
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    # One image per path avoids making each result unreadably small.
    for path in range(4):
        canvas = Image.new("RGB", (976, 544), "white")
        draw = ImageDraw.Draw(canvas)
        titles = ["GT", "Old alternate20", "Random-only20", "New state10",
                  "New Zero10", "New state20", "New Zero20"]
        for col, title in enumerate(titles):
            draw.text((84 + col * 128, 10), title, fill="black")
        for row, (index, data) in enumerate(stored.items()):
            scale = float(data["clean"].abs().flatten().quantile(.995).clamp_min(1e-6))
            panels = [data["clean"], previous[index]["random_zero"][19, path],
                      previous[index]["random_only"][19, path], data["accumulated"][9, path],
                      data["zero"][9, path], data["accumulated"][19, path], data["zero"][19, path]]
            for col, field in enumerate(panels):
                canvas.paste(tile(field, scale), (80 + col * 128, 32 + row * 128))
            draw.text((5, 85 + row * 128), f"Mol {index}", fill="black")
        canvas.save(args.output / f"comparison-path{path+1}.png")
    torch.save(stored, args.output / "frames.pt")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
