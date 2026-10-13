"""Paired inference from partial/pure noise, with MS cross-attention ablated."""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train_absolute_cloud import RaggedSpectrumTokenRunner, _condition_tokens
from train_cloud import _make_schedule, _transition_jumps

from molai.data import collate_spectrum_field_batch
from molai.models.relative_rollout import RelativeImageRolloutCloud, restore_token_relative_models


@contextmanager
def without_ms_attention(cloud):
    """Remove the MS-dependent attention update, retaining each block's FFN."""
    def image_only(block, query, *_args, **_kwargs):
        return query + block.ff(query)

    if cloud.cache_condition_kv:
        raise ValueError("Disable condition caching before the diagnostic ablation")
    previous = []
    try:
        for block in cloud.condition_blocks:
            previous.append((block, block.__dict__.get("forward")))
            block.forward = MethodType(image_only, block)
        yield
    finally:
        for block, original in previous:
            if original is None:
                del block.forward
            else:
                block.forward = original


def measures(fields, clean):
    """One value per step, averaged over random paths where applicable."""
    flat = fields.float().flatten(2)
    truth = clean.float().flatten()
    centered = flat - flat.mean(-1, keepdim=True)
    centered_truth = truth - truth.mean()
    correlation = (centered * centered_truth).sum(-1) / (
        centered.norm(dim=-1) * centered_truth.norm()).clamp_min(1e-12)
    mse = (flat - truth).square().mean(-1)
    return {
        "zero_correlation": correlation[:, 0].tolist(),
        "random_correlation": correlation[:, 1:].mean(-1).tolist(),
        "best_random_correlation": correlation[:, 1:].max(-1).values.tolist(),
        "zero_mse": mse[:, 0].tolist(),
        "random_mse": mse[:, 1:].mean(-1).tolist(),
        "zero_rms": flat[:, 0].square().mean(-1).sqrt().tolist(),
        "random_rms": flat[:, 1:].square().mean((1, 2)).sqrt().tolist(),
        "random_diversity": fields[:, 1:].float().var(1, unbiased=False).flatten(1).mean(-1).tolist(),
    }


def render_overview(stored, level, output):
    entries = list(stored.items())
    fig, axes = plt.subplots(len(entries), 6, figsize=(11, 2 * len(entries)), squeeze=False)
    titles = ["GT", f"Input L{level}", "MS Zero / s20", "No MS Zero / s20",
              "MS Random1 / s20", "No MS Random1 / s20"]
    for row, (index, data) in enumerate(entries):
        case = data["levels"][level]
        panels = [data["clean"], case["initial"], case["ms"][-1, 0],
                  case["no_ms"][-1, 0], case["ms"][-1, 1], case["no_ms"][-1, 1]]
        scale = float(data["clean"].abs().quantile(.995).clamp_min(1e-6))
        for col, panel in enumerate(panels):
            axes[row, col].imshow(panel[0], cmap="coolwarm", vmin=-scale, vmax=scale)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=9)
        axes[row, 0].set_ylabel(f"Molecule {index}")
    fig.suptitle(f"Epoch 14 | L{level} start | 20-step paired rollout | GT color scale")
    fig.tight_layout()
    fig.savefig(output / f"level{level}_comparison.png", dpi=150)
    plt.close(fig)


def render_evolution(index, data, levels, output):
    selected = [1, 2, 4, 8, 12, 16, 20]
    fig, axes = plt.subplots(len(levels) * 4, 9, figsize=(15, len(levels) * 6.2), squeeze=False)
    scale = float(data["clean"].abs().quantile(.995).clamp_min(1e-6))
    for i, level in enumerate(levels):
        case = data["levels"][level]
        for j, (mode, path) in enumerate((("ms", 0), ("no_ms", 0), ("ms", 1), ("no_ms", 1))):
            row = i * 4 + j
            panels = [data["clean"], case["initial"]] + [case[mode][s-1, path] for s in selected]
            for col, panel in enumerate(panels):
                axes[row, col].imshow(panel[0], cmap="coolwarm", vmin=-scale, vmax=scale)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                if row == 0:
                    axes[row, col].set_title(["GT", "Input", *[f"Step {s}" for s in selected]][col])
            axes[row, 0].set_ylabel(f"L{level} {mode}\n{'Zero' if path == 0 else 'Random1'}", fontsize=9)
    fig.suptitle(f"Molecule {index}: middle vs pure-noise starts; fixed GT color scale")
    fig.tight_layout()
    fig.savefig(output / f"evolution_molecule{index}.png", dpi=130)
    plt.close(fig)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[16, 32, 48, 64])
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(args.samples, map_location="cpu", weights_only=False)
    records = [r for r in bundle["samples"] if r["split"] == "validation"][:args.count]
    if not records:
        raise ValueError("No validation records in the sample bundle")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cloud, encoder, audit = restore_token_relative_models(
        source, int(bundle["manifest"]["metadata_dim"]), gradient_checkpointing=False)
    checkpoint = {"path": str(args.checkpoint.resolve()), "epoch": source["epoch"] + 1,
                  "global_step": source["global_step"], "config": source["config"],
                  "restore_audit": audit}
    del source
    cloud, encoder = cloud.cuda().eval(), encoder.cuda().eval()
    cloud.cache_condition_kv = False
    cloud.packed_condition_kv = False
    runner = RaggedSpectrumTokenRunner(encoder).eval()
    rollout = RelativeImageRolloutCloud(cloud).eval()
    device = torch.device("cuda")
    schedule = _make_schedule(bundle["manifest"], checkpoint["config"], device)
    if any(level < 0 or level >= len(schedule.alpha_bar) for level in args.levels):
        raise ValueError("Requested level outside checkpoint schedule")
    report = {
        "checkpoint": checkpoint, "device": torch.cuda.get_device_name(),
        "count": len(records), "steps": 20, "random_paths": 4,
        "no_ms_definition": "MS cross-attention update removed; condition-block FFNs retained. "
                            "No MS encoder evaluated for this branch. This is an inference ablation, "
                            "not a separately trained unconditional model.",
        "pairing": "Same image-noise field and random-token stream across MS conditions and levels; "
                   "x_L=sqrt(alpha_L)*GT+sqrt(1-alpha_L)*noise_scale*epsilon.",
        "caveat": "20 repeated network applications; nominal target levels are bookkeeping only. "
                  "The model receives no level embedding. Iteration after nominal level zero is "
                  "reported separately from initial denoising. Four-molecule diagnostic, not full validation.",
        "levels": {str(level): {"alpha_bar": float(schedule.alpha_bar[level])} for level in args.levels},
        "molecules": {},
    }
    stored = {}
    started = time.perf_counter()
    parameter_versions = [p._version for p in cloud.parameters()]
    for record in records:
        index = record["index"]
        batch = collate_spectrum_field_batch([record["item"]], peak_chunk_size=256)
        clean = batch["field"].to(device)
        torch.manual_seed(923001 + index)
        epsilon = torch.randn_like(clean)
        if schedule.zero_mean_noise:
            epsilon -= epsilon.mean((-2, -1), keepdim=True)
        noise = torch.randn(1, 20, 5, cloud.noise_token_count, cloud.noise_token_dim, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            condition, mask = _condition_tokens(runner, batch, device)
        stored[index] = {"clean": clean[0].cpu(), "levels": {}}
        row = {"smiles": record["item"]["smiles"], "levels": {}}
        for level in args.levels:
            alpha = schedule.alpha_bar[level]
            initial = alpha.sqrt() * clean + (1-alpha).sqrt() * schedule.noise_scale * epsilon
            case = {"initial": initial[0].cpu()}
            baseline = measures(initial[:, None].expand(-1, 5, -1, -1, -1), clean)
            nominal, current = [], torch.tensor([level], device=device)
            for _ in range(20):
                current = (current - _transition_jumps(current, checkpoint["config"])).clamp_min(0)
                nominal.append(int(current[0]))
            results = {"input_correlation": baseline["zero_correlation"][0],
                       "input_mse": baseline["zero_mse"][0], "nominal_target_levels": nominal}
            for mode in ("ms", "no_ms"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if mode == "ms":
                        fields = rollout(initial, condition, condition_mask=mask, steps=20, noise=noise)
                    else:
                        dummy = torch.zeros(1, 1, condition.shape[-1], device=device, dtype=condition.dtype)
                        dummy_mask = torch.ones(1, 1, device=device, dtype=torch.bool)
                        with without_ms_attention(cloud):
                            fields = rollout(initial, dummy, condition_mask=dummy_mask, steps=20, noise=noise)
                if not torch.isfinite(fields).all():
                    raise RuntimeError(f"Non-finite rollout: {index}, {level}, {mode}")
                case[mode] = fields[0].float().cpu()
                results[mode] = measures(case[mode], stored[index]["clean"])
                del fields
            stored[index]["levels"][level] = case
            row["levels"][str(level)] = results
            print(json.dumps({"molecule": index, "level": level, "input_corr": results["input_correlation"],
                              **{mode: {key: results[mode][key][-1] for key in
                                 ("zero_correlation", "random_correlation", "random_mse")}
                                 for mode in ("ms", "no_ms")},
                              "seconds": time.perf_counter()-started}), flush=True)
        report["molecules"][str(index)] = row
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        torch.save(stored, args.output / "frames.pt")
    assert parameter_versions == [p._version for p in cloud.parameters()], "Parameters were modified"
    report["summary"] = {}
    for level in args.levels:
        rows = [row["levels"][str(level)] for row in report["molecules"].values()]
        summary = {key: sum(row[key] for row in rows)/len(rows)
                   for key in ("input_correlation", "input_mse")}
        for mode in ("ms", "no_ms"):
            summary[mode] = {key: [sum(row[mode][key][step] for row in rows)/len(rows)
                                  for step in range(20)] for key in rows[0][mode]}
        report["summary"][str(level)] = summary
        render_overview(stored, level, args.output)
    render_evolution(records[0]["index"], stored[records[0]["index"]], [32, 64], args.output)
    report["elapsed_seconds"] = time.perf_counter()-started
    report["parameters_unchanged"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Diagnostic complete", flush=True)


if __name__ == "__main__":
    main()
