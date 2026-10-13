"""Sum Zero(Random(fresh pure noise)) samples; never feed back accumulated images."""
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
def fresh_noise_rollout(cloud, condition, mask, fresh_inputs, tokens):
    """Each round independently starts from fresh_inputs[k], then sums its Zero image."""
    paths = fresh_inputs.shape[1]
    condition = condition.repeat_interleave(paths, dim=0)
    mask = mask.repeat_interleave(paths, dim=0)
    zero_tokens = torch.zeros_like(tokens[0, 0, :, None])
    signal_sum = torch.zeros_like(fresh_inputs[0])
    sums, random_outputs, zero_outputs = [], [], []
    for cycle, fresh in enumerate(fresh_inputs):
        with torch.autocast(fresh.device.type, dtype=torch.bfloat16, enabled=fresh.is_cuda):
            random = cloud(fresh, condition, samples=1, noise=tokens[0, cycle, :, None],
                           condition_mask=mask)[0][:, 0]
            zero = cloud(random, condition, samples=1, noise=zero_tokens,
                         condition_mask=mask)[0][:, 0]
        signal_sum = signal_sum + zero.float()
        assert torch.isfinite(signal_sum).all(), f"Non-finite sum at cycle {cycle+1}"
        sums.append(signal_sum.cpu())
        random_outputs.append(random.float().cpu())
        zero_outputs.append(zero.float().cpu())
    result = {"signal_sum": torch.stack(sums), "random": torch.stack(random_outputs),
              "zero": torch.stack(zero_outputs), "fresh_inputs": fresh_inputs.cpu()}
    torch.testing.assert_close(result["signal_sum"], result["zero"].cumsum(0), rtol=1e-5, atol=1e-5)
    return result


def render_evolution(data, destination):
    selected = [1, 2, 4, 8, 12, 16, 20]
    scale = float(data["clean"].abs().flatten().quantile(.995).clamp_min(1e-6))
    canvas = Image.new("RGB", (1136, 1056), "white")
    draw = ImageDraw.Draw(canvas)
    for col, title in enumerate(["GT", *[f"Cycle {s}" for s in selected]]):
        draw.text((116 + col * 128, 10), title, fill="black")
    for path in range(4):
        for offset, kind in enumerate(("signal_sum", "zero")):
            row = path * 2 + offset
            for col, field in enumerate([data["clean"], *[data[kind][s-1, path] for s in selected]]):
                canvas.paste(tile(field, scale), (112 + col * 128, 32 + row * 128))
            draw.text((3, 85 + row * 128), f"P{path+1} {'Sum of Zero' if offset == 0 else 'Single Zero'}",
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
        "equation": "epsilon_k=fresh_centered_Gaussian*noise_scale; "
                    "z_k=Zero(Random(epsilon_k)); sum_k=sum_previous+z_k; x_k=initial+sum_k",
        "note": "No accumulated image is fed to the model. Pure-noise image inputs are independent "
                "across cycles and paths; MS condition fixed. Every cycle restarts at pure noise, "
                "so this is independent-sample accumulation, NOT a 20-level denoising trajectory. "
                "Both signal sum and initial-noise-plus-sum saved. No variance correction, "
                "averaging, image clipping or extra mean projection; display clipping only. "
                "Local BF16 eval, no training. 4 fixed validation molecules x 4 paths. "
                "GT 99.5% color scale shared across comparisons. Random tokens and initial "
                "noise exactly match the earlier experiment; additional image noises use the "
                "same seeded stream after token generation.",
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
        tokens = torch.randn(1, 20, 4, cloud.noise_token_count, cloud.noise_token_dim, device="cuda")
        fresh_inputs = torch.randn(20, 4, *clean.shape[1:], device="cuda")
        fresh_inputs -= fresh_inputs.mean((-2, -1), keepdim=True)
        fresh_inputs *= schedule.noise_scale
        with torch.autocast("cuda", dtype=torch.bfloat16):
            condition, mask = _condition_tokens(RaggedSpectrumTokenRunner(encoder), batch, torch.device("cuda"))
        data = fresh_noise_rollout(cloud, condition, mask, fresh_inputs, tokens)
        data.update(clean=clean[0].cpu(), initial=initial[0].cpu())
        data["accumulated"] = data["signal_sum"] + initial.cpu()
        stored[index] = data
        row = {kind: measure(data[kind], data["clean"])
               for kind in ("accumulated", "signal_sum", "zero", "random")}
        row["clean_rms"] = float(clean.square().mean().sqrt())
        report["molecules"][str(index)] = row
        render_evolution(data, args.output / f"evolution-{index}.png")
        print(json.dumps({"molecule": index, "seconds": time.perf_counter() - started,
                          "final": {kind: {key: row[kind][key][-1] for key in ("rms", "correlation", "mse")}
                                    for kind in ("accumulated", "signal_sum", "zero")}}), flush=True)
    report["summary"] = {}
    for kind in ("accumulated", "signal_sum", "zero"):
        report["summary"][kind] = {
            key: [sum(row[kind][key][cycle] for row in report["molecules"].values()) / 4
                  for cycle in range(20)]
            for key in ("rms", "correlation", "best_correlation", "mse", "best_mse",
                        "diversity_variance", "max_abs_spatial_mean")}
    report["seconds"] = time.perf_counter() - started
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    for path in range(4):
        canvas = Image.new("RGB", (976, 544), "white")
        draw = ImageDraw.Draw(canvas)
        titles = ["GT", "Old alternate20", "Random-only20", "Fresh sum4",
                  "Fresh sum10", "Fresh sum20", "Initial+sum20"]
        for col, title in enumerate(titles):
            draw.text((84 + col * 128, 10), title, fill="black")
        for row, (index, data) in enumerate(stored.items()):
            scale = float(data["clean"].abs().flatten().quantile(.995).clamp_min(1e-6))
            panels = [data["clean"], previous[index]["random_zero"][19, path],
                      previous[index]["random_only"][19, path], data["signal_sum"][3, path],
                      data["signal_sum"][9, path], data["signal_sum"][19, path], data["accumulated"][19, path]]
            for col, field in enumerate(panels):
                canvas.paste(tile(field, scale), (80 + col * 128, 32 + row * 128))
            draw.text((5, 85 + row * 128), f"Mol {index}", fill="black")
        canvas.save(args.output / f"comparison-path{path+1}.png")
    torch.save(stored, args.output / "frames.pt")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({kind: {key: values[-1] for key, values in metrics.items()}
                      for kind, metrics in report["summary"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
