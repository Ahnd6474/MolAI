"""Local matched Random/Zero image-feedback diagnostic, without weight updates."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import torch
from compare_ustat_navigation_rollout import restore
from PIL import Image, ImageDraw
from train_absolute_cloud import RaggedSpectrumTokenRunner, _condition_tokens
from train_cloud import _make_schedule, _transition_jumps

from molai.data import collate_spectrum_field_batch


def measure(fields, clean):
    flat = fields.float().flatten(2)
    truth = clean.float().flatten()
    centered = flat - flat.mean(-1, keepdim=True)
    truth_centered = truth - truth.mean()
    corr = (centered * truth_centered).sum(-1) / (
        centered.norm(dim=-1) * truth_centered.norm()).clamp_min(1e-12)
    mse = (flat - truth).square().mean(-1)
    return {
        "correlation": corr.mean(1).tolist(),
        "best_correlation": corr.max(1).values.tolist(),
        "mse": mse.mean(1).tolist(),
        "best_mse": mse.min(1).values.tolist(),
        "rms": flat.square().mean((1, 2)).sqrt().tolist(),
        "diversity_variance": flat.var(dim=1, unbiased=False).mean(-1).tolist(),
        "max_abs_spatial_mean": flat.mean(-1).abs().max(1).values.tolist(),
        "per_path_correlation": corr.tolist(),
    }


@torch.no_grad()
def rollout(cloud, condition, mask, initial, noise, config, alternating):
    current = initial.repeat_interleave(4, dim=0)
    condition = condition.repeat_interleave(4, dim=0)
    mask = mask.repeat_interleave(4, dim=0)
    frames, levels, operations = [], [], []
    level = torch.tensor([64], device=initial.device)
    random_index = 0
    for call in range(20):
        is_random = not alternating or call % 2 == 0
        tokens = noise[0, random_index, :, None] if is_random else torch.zeros_like(noise[0, 0, :, None])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            current = cloud(current, condition, samples=1, noise=tokens,
                            condition_mask=mask)[0][:, 0]
        if is_random:
            level = (level - _transition_jumps(level, config)).clamp_min(0)
            random_index += 1
        frames.append(current.float().cpu())
        levels.append(int(level.item()))
        operations.append("Random" if is_random else "Zero")
    frames = torch.stack(frames)
    assert torch.isfinite(frames).all(), "Non-finite inference output"
    return frames, levels, operations


def tile(field, scale):
    values = ((field[0].float() / scale).clamp(-1, 1) + 1) / 2
    rgb = (matplotlib.colormaps["coolwarm"](values.numpy())[..., :3] * 255).round().astype("uint8")
    return Image.fromarray(rgb)


def render_evolution(clean, initial, frames, levels, operations, destination):
    selected = [1, 2, 3, 4, 9, 10, 19, 20]
    scale = float(clean.abs().flatten().quantile(.995).clamp_min(1e-6))
    canvas = Image.new("RGB", (1360, 544), "white")
    draw = ImageDraw.Draw(canvas)
    titles = ["GT", "Input", *[f"{s}: {operations[s-1]} L{levels[s-1]}" for s in selected]]
    for col, title in enumerate(titles):
        draw.text((84 + 128 * col, 10), title, fill="black")
    for path in range(4):
        for col, field in enumerate([clean, initial, *[frames[s-1, path] for s in selected]]):
            canvas.paste(tile(field, scale), (80 + 128 * col, 32 + 128 * path))
        draw.text((5, 85 + 128 * path), f"Path {path+1}", fill="black")
    canvas.save(destination)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(args.samples, map_location="cpu", weights_only=False)
    records = [r for r in bundle["samples"] if r["index"] in (33, 88, 122, 177)]
    assert len(records) == 4
    cloud, encoder, metadata = restore(args.checkpoint)
    schedule = _make_schedule(bundle["manifest"], metadata["config"], torch.device("cuda"))
    report = {
        "checkpoint": metadata,
        "device": torch.cuda.get_device_name(),
        "note": "4 fixed validation molecules x 4 paths; local eval BF16, strict checkpoint restore. "
                "Random and Zero alternate for 20 model calls (10 Random transitions). "
                "Zero holds the nominal level. No variance correction or extra mean projection. "
                "Random tokens matched by Random-transition index across protocols. "
                "Common initial pure noise for all paths/protocols per molecule. "
                "Image scale fixed to GT absolute 99.5% quantile per molecule; shared protocols. "
                "Nominal levels are labels only: there is no level embedding. "
                "Zero removes VP noise, so subsequent Random input is outside its training marginal.",
        "molecules": {},
    }
    stored = {}
    started = time.perf_counter()
    for record in records:
        index = record["index"]
        batch = collate_spectrum_field_batch([record["item"]], peak_chunk_size=256)
        clean = batch["field"].cuda()
        torch.manual_seed(923001 + index)
        initial = torch.randn_like(clean)
        initial -= initial.mean((-2, -1), keepdim=True)
        initial *= schedule.noise_scale
        noise = torch.randn(1, 20, 4, cloud.noise_token_count, cloud.noise_token_dim, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            condition, mask = _condition_tokens(RaggedSpectrumTokenRunner(encoder), batch, torch.device("cuda"))
        outcomes, row = {}, {"clean_rms": float(clean.square().mean().sqrt())}
        for name, alternating in (("random_only", False), ("random_zero", True)):
            frames, levels, operations = rollout(cloud, condition, mask, initial, noise,
                                                 metadata["config"], alternating)
            row[name] = measure(frames, clean.cpu())
            row[name]["levels"] = levels
            row[name]["operations"] = operations
            nominal_mean = schedule.alpha_bar[torch.tensor(levels, device="cuda")].sqrt().cpu()
            target = nominal_mean[:, None, None, None, None] * clean.cpu()
            row[name]["nominal_signal_mse"] = (frames - target).square().mean((1, 2, 3, 4)).tolist()
            outcomes[name] = frames
            render_evolution(clean[0].cpu(), initial[0].cpu(), frames, levels, operations,
                             args.output / f"{name}-{index}.png")
        report["molecules"][str(index)] = row
        stored[index] = {"clean": clean[0].cpu(), "initial": initial[0].cpu(), **outcomes}
        print(json.dumps({"molecule": index, "seconds": time.perf_counter() - started,
                          "alt20_corr": row["random_zero"]["correlation"][-1],
                          "random10_corr": row["random_only"]["correlation"][9],
                          "random20_corr": row["random_only"]["correlation"][-1]}), flush=True)
    report["summary"] = {}
    for label, name, call in (("Random10_L12", "random_only", 9),
                              ("Alternate20_L12", "random_zero", 19),
                              ("Random20_L0", "random_only", 19)):
        report["summary"][label] = {
            key: sum(row[name][key][call] for row in report["molecules"].values()) / len(records)
            for key in ("correlation", "best_correlation", "mse", "best_mse", "rms",
                        "diversity_variance", "max_abs_spatial_mean", "nominal_signal_mse")}
    report["seconds"] = time.perf_counter() - started
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    # Four molecules, GT plus all four paths for each matched protocol.
    canvas = Image.new("RGB", (1744, 544), "white")
    draw = ImageDraw.Draw(canvas)
    titles = ["GT", *[f"Random10/{i+1}" for i in range(4)],
              *[f"Alternate20/{i+1}" for i in range(4)], *[f"Random20/{i+1}" for i in range(4)]]
    for col, title in enumerate(titles):
        draw.text((84 + col * 128, 10), title, fill="black")
    for row, (index, data) in enumerate(stored.items()):
        scale = float(data["clean"].abs().flatten().quantile(.995).clamp_min(1e-6))
        panels = [data["clean"], *data["random_only"][9],
                  *data["random_zero"][19], *data["random_only"][19]]
        for col, field in enumerate(panels):
            canvas.paste(tile(field, scale), (80 + col * 128, 32 + row * 128))
        draw.text((5, 85 + row * 128), f"Mol {index}", fill="black")
    canvas.save(args.output / "comparison.png")
    torch.save(stored, args.output / "frames.pt")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
