"""Paired L64/23-call versus oracle VP L62/22-call inference on local GPU."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnose_midstart_ms_ablation import measures
from rollout_alternating_zero import tile
from train_absolute_cloud import RaggedSpectrumTokenRunner, _condition_tokens
from train_cloud import _make_schedule, _transition_jumps
from train_relative_image_rollout import parallel_smiles_metrics

from molai.data import collate_spectrum_field_batch
from molai.models.parallel_smiles import ParallelFieldToSmiles
from molai.models.relative_rollout import RelativeImageRolloutCloud, restore_token_relative_models
from molai.models.smiles import SmilesTokenizer


def nominal_levels(start, steps, config, device):
    current = torch.tensor([start], device=device)
    result = [start]
    for _ in range(steps):
        current = (current - _transition_jumps(current, config)).clamp_min(0)
        result.append(int(current.item()))
    return result


def render(stored, levels, output):
    columns = ["GT", "L64 input", "L64 -> L62", "Oracle L62 input",
               "L64 final", "L62 final"]
    canvas = Image.new("RGB", (104 + 128 * len(columns), 32 + 128 * len(stored) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(columns):
        draw.text((104 + col * 128, 8), label, fill="black")
    for i, (index, row) in enumerate(stored.items()):
        scale = float(row["clean"].abs().quantile(.995).clamp_min(1e-6))
        for j, path in enumerate((0, 1)):
            y = 32 + (i * 2 + j) * 128
            draw.text((3, y + 45), f"Mol {index}\n" + ("Zero" if path == 0 else "Random 1"), fill="black")
            panels = [row["clean"], row["l64"][0, path], row["l64"][1, path],
                      row["l62"][0, path], row["l64"][-1, path], row["l62"][-1, path]]
            for col, panel in enumerate(panels):
                canvas.paste(tile(panel, scale), (104 + col * 128, y))
        selected = [62, 56, 32, 16, 8, 0]
        evolution = Image.new("RGB", (104 + 7 * 128, 32 + 4 * 128), "white")
        pen = ImageDraw.Draw(evolution)
        for col, label in enumerate(["GT", *[f"At L{x}" for x in selected]]):
            pen.text((104 + col * 128, 8), label, fill="black")
        for j, (mode, path) in enumerate((("l64", 0), ("l62", 0), ("l64", 1), ("l62", 1))):
            y = 32 + j * 128
            pen.text((3, y + 45), f"{mode.upper()} start\n" + ("Zero" if path == 0 else "Random 1"), fill="black")
            panels = [row["clean"], *[row[mode][levels[mode].index(level), path] for level in selected]]
            for col, panel in enumerate(panels):
                evolution.paste(tile(panel, scale), (104 + col * 128, y))
        evolution.save(output / f"evolution_{index}.png")
    canvas.save(output / "comparison.png")
    final = Image.new("RGB", (96 + 3 * 128, 32 + len(stored) * 128), "white")
    pen = ImageDraw.Draw(final)
    for col, label in enumerate(("GT", "L64 / 23 calls", "L62 / 22 calls")):
        pen.text((96 + col * 128, 8), label, fill="black")
    for i, (index, row) in enumerate(stored.items()):
        scale = float(row["clean"].abs().quantile(.995).clamp_min(1e-6))
        y = 32 + i * 128
        pen.text((3, y + 45), f"Mol {index}\nRandom 1", fill="black")
        for col, panel in enumerate((row["clean"], row["l64"][-1, 1], row["l62"][-1, 1])):
            final.paste(tile(panel, scale), (96 + col * 128, y))
    final.save(output / "final_random.png")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=923001)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    bundle = torch.load(args.samples, map_location="cpu", weights_only=False)
    records = [r for r in bundle["samples"] if r["split"] == "validation"][:args.count]
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = source["config"]
    options = config["relative_image_rollout"]
    cloud, encoder, audit = restore_token_relative_models(
        source, int(bundle["manifest"]["metadata_dim"]), gradient_checkpointing=False)
    cloud, encoder = cloud.to(device).eval(), encoder.to(device).eval()
    cloud.cache_condition_kv = bool(options.get("cache_ms_kv", False))
    cloud.packed_condition_kv = bool(options.get("packed_ms_kv", False))
    runner = RaggedSpectrumTokenRunner(encoder).eval()
    rollout = RelativeImageRolloutCloud(cloud).eval()
    tokenizer = SmilesTokenizer(source["smiles_vocabulary"])
    text = ParallelFieldToSmiles(
        vocab_size=len(tokenizer), max_length=int(options["smiles_max_length"]),
        dim=int(options["smiles_dim"]), heads=int(options["smiles_heads"]),
        layers=int(options["smiles_layers"]), image_grid_size=int(options["smiles_image_grid_size"]),
        ff_multiplier=float(options.get("smiles_ffn_ratio", 4)),
        dropout=float(options.get("smiles_dropout", 0)),
    )
    text.load_state_dict(source["parallel_smiles"], strict=True)
    text = text.to(device).eval()
    schedule = _make_schedule(bundle["manifest"], config, device)
    levels = {"l64": nominal_levels(64, 23, config, device),
              "l62": nominal_levels(62, 22, config, device)}
    assert levels["l64"][1:] == levels["l62"] and levels["l64"][-1] == 0
    assert float(schedule.alpha_bar[64]) == 0
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "epoch": source["epoch"] + 1,
        "global_step": source["global_step"], "device": torch.cuda.get_device_name(),
        "count": len(records), "seed": args.seed, "restore_audit": audit,
        "protocol": "Same epsilon in x64=noise_scale*epsilon and x62=sqrt(alpha62)*GT+"
                    "sqrt(1-alpha62)*noise_scale*epsilon. Shared token stream after the first "
                    "L64 call: L62 uses tokens[:,1:]. Persistent Zero+4 Random paths. "
                    "L62 input contains oracle GT information; this is a bottleneck diagnostic, "
                    "not generation from MS alone. No rescaling, clipping or extra projection.",
        "alpha_bar_62": float(schedule.alpha_bar[62]),
        "signal_amplitude_62": float(schedule.alpha_bar[62].sqrt()),
        "noise_scale_62": float(schedule.noise_scale * (1 - schedule.alpha_bar[62]).sqrt()),
        "nominal_levels": levels, "molecules": {},
    }
    del source
    versions = [p._version for module in (cloud, encoder, text) for p in module.parameters()]
    stored, finals = {}, {}
    started = time.perf_counter()
    for i, record in enumerate(records):
        index = record["index"]
        batch = collate_spectrum_field_batch([record["item"]], tokenizer=tokenizer,
                                            max_smiles_length=int(options["smiles_max_length"]))
        clean = batch["field"].to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed + index)
        epsilon = torch.randn(clean.shape, device=device, generator=generator)
        if schedule.zero_mean_noise:
            epsilon -= epsilon.mean((-2, -1), keepdim=True)
        tokens = torch.randn(1, 23, 5, cloud.noise_token_count, cloud.noise_token_dim,
                             device=device, generator=generator)
        alpha = schedule.alpha_bar[62]
        inputs = {"l64": schedule.noise_scale * epsilon,
                  "l62": alpha.sqrt() * clean + (1-alpha).sqrt() * schedule.noise_scale * epsilon}
        row = {"smiles": record["item"]["smiles"]}
        frames = {"clean": clean[0].cpu()}
        last = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            condition, mask = _condition_tokens(runner, batch, device)
            for mode, steps in (("l64", 23), ("l62", 22)):
                noise = tokens if mode == "l64" else tokens[:, 1:]
                output = rollout(inputs[mode], condition, condition_mask=mask, steps=steps,
                                 random_samples=4, noise=noise)
                assert torch.isfinite(output).all()
                image_sequence = torch.cat((inputs[mode][:, None, None].expand(-1, 1, 5, -1, -1, -1),
                                            output.float()), dim=1)[0].cpu()
                row[mode] = measures(image_sequence, clean[0].cpu())
                output_text = text(output[:, -1, 1:])
                row[mode]["text_final"] = {
                    key: float(value) for key, value in parallel_smiles_metrics(
                        output_text, batch["token_ids"].to(device), tokenizer.pad_id).items()
                }
                last[mode] = image_sequence[-1].clone()
                if i < 4:
                    frames[mode] = image_sequence
                del output, output_text, image_sequence
        if i < 4:
            stored[index] = frames
        finals[index] = last
        report["molecules"][str(index)] = row
        print(json.dumps({"done": i+1, "count": len(records), "index": index,
                          "seconds": round(time.perf_counter()-started, 1),
                          **{mode: {key: row[mode][key][-1] for key in
                                    ("random_correlation", "zero_correlation", "random_mse")}
                             for mode in ("l64", "l62")}}), flush=True)
        (args.output / "report_partial.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = list(report["molecules"].values())
    report["summary"] = {}
    for mode in ("l64", "l62"):
        report["summary"][mode] = {
            key: np.mean([row[mode][key] for row in rows], axis=0).tolist()
            for key in rows[0][mode] if key != "text_final"
        }
        report["summary"][mode]["text_final"] = {
            key: float(np.mean([row[mode]["text_final"][key] for row in rows]))
            for key in rows[0][mode]["text_final"]
        }
    rng = np.random.default_rng(7393)
    bootstrap = rng.integers(0, len(rows), (10000, len(rows)))
    report["paired_final_differences_l62_minus_l64"] = {}
    for key in ("zero_correlation", "random_correlation", "best_random_correlation",
                "zero_mse", "random_mse", "random_diversity"):
        delta = np.array([row["l62"][key][-1] - row["l64"][key][-1] for row in rows])
        report["paired_final_differences_l62_minus_l64"][key] = {
            "mean": float(delta.mean()),
            "molecule_bootstrap_95_percent_ci": np.quantile(delta[bootstrap].mean(1), [.025, .975]).tolist(),
            "improved_molecules": int((delta < 0 if "mse" in key else delta > 0).sum()),
        }
    render(stored, levels, args.output)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for mode, label in (("l64", "L64 start: 23 calls"), ("l62", "Oracle L62 start: 22 calls")):
        for ax, key in zip(axes, ("random_correlation", "random_mse")):
            ax.plot(levels[mode], report["summary"][mode][key], marker=".", label=label)
            ax.set_xlabel("Nominal VP level (model receives no level embedding)")
            ax.set_ylabel(key.replace("_", " "))
            ax.invert_xaxis() if mode == "l64" else None
            ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "metrics.png", dpi=140)
    plt.close(fig)
    assert versions == [p._version for module in (cloud, encoder, text) for p in module.parameters()]
    report.update(parameters_unchanged=True, elapsed_seconds=time.perf_counter()-started,
                  peak_cuda_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
    torch.save({"previews": stored, "finals": finals}, args.output / "frames.pt")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"complete": True, "seconds": report["elapsed_seconds"],
                      "paired": report["paired_final_differences_l62_minus_l64"]}), flush=True)


if __name__ == "__main__":
    main()
