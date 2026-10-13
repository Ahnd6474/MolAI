"""Plot data-derived diagnostic figures using unchanged shared field scales."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from diagnose_relative_rollout import autocast, encode, load_models, metrics
from train_cloud import _make_schedule
from train_relative_image_rollout import trajectory_targets

from molai.data import collate_spectrum_field_batch
from molai.models.relative_rollout import RelativeImageRolloutCloud


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--epoch", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    epoch = torch.load(args.epoch, map_location="cpu", weights_only=False)
    bundle = torch.load(args.samples, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    cloud, encoder = load_models(source, epoch, device, True)
    cloud.eval()
    encoder.eval()
    record = next(record for record in bundle["samples"] if record["split"] == "validation")
    batch = collate_spectrum_field_batch([record["item"]], peak_chunk_size=256)
    clean = batch["field"].to(device)
    schedule = _make_schedule(bundle["manifest"], epoch["config"], device)
    pictures, comparisons = [], []
    for level in (32, 64):
        torch.manual_seed(1729)
        initial, targets, levels = trajectory_targets(schedule, clean, torch.tensor([level], device=device), 4, 4, epoch["config"])
        torch.manual_seed(2718)
        # Preserve the exact four-step diagnostic noise, then extend its path.
        noise = torch.cat((torch.randn(1, 4, 5, 64, 64, device=device),
                           torch.randn(1, 16, 5, 64, 64, device=device)), dim=1)
        with torch.no_grad(), autocast("bf16"):
            condition, mask = encode(encoder, batch, device, True)
            frames = RelativeImageRolloutCloud(cloud)(initial, condition, condition_mask=mask, steps=20, noise=noise)
        comparisons.append({"input_level": level, "step4_target_level": int(levels[0, -1]),
            "step4_prediction": metrics(frames[:, 3], clean),
            "step4_exact_target": metrics(torch.cat((targets[:, -1, :1], targets[:, -1]), dim=1), clean),
            "step20_prediction": metrics(frames[:, -1], clean)})
        for branch in (0, 1):
            pictures.append((f"L{level} " + ("Zero" if branch == 0 else "Random 1"),
                [clean[0, 0].cpu(), initial[0, 0].cpu()] + [frames[0, step - 1, branch, 0].float().cpu() for step in (1, 4, 20)]))
    (args.output / "restoration_vs_generation.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(4, 5, figsize=(9, 7.3))
    scale = float(clean.abs().quantile(.995))
    for row, (label, panels) in enumerate(pictures):
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel, cmap="coolwarm", vmin=-scale, vmax=scale)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(("GT", "Input", "Step 1", "Step 4", "Step 20")[column])
        axes[row, 0].set_ylabel(label)
    figure.suptitle("Epoch 1: noisy-image restoration vs pure-noise generation\nSame molecule, fixed seed, one shared GT color scale")
    figure.tight_layout()
    figure.savefig(args.output / "noise_vs_generation.png", dpi=120)
    plt.close(figure)

    report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
    controls = json.loads((args.output / "controls.json").read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    series = {"Legacy E19 (L32)": controls["legacy19_level32"],
        "Aligned (L32)": report["gradients"]["aligned"]["validation_33"],
        "Epoch 1 (L32)": report["gradients"]["epoch1"]["validation_33"],
        "Epoch 1 (L64)": controls["epoch1_level64"]}
    for label, values in series.items():
        axes[0].plot((1, 2, 4), [values[f"prefix_{depth}"]["norm"] for depth in (1, 2, 4)], "o-", label=label)
    axes[0].set_yscale("log")
    axes[0].set_xticks((1, 2, 4))
    axes[0].set_xlabel("Supervised rollout length")
    axes[0].set_ylabel("Unclipped parameter gradient norm")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=.3)
    values = [report["gradients"]["epoch1"]["validation_33"]["prefix_4"]["norm"],
        controls["epoch1_stop_random_query_feedback_level32"]["prefix_4"]["norm"],
        report["gradients"]["epoch1"]["validation_detach_control"]["prefix_4"]["norm"]]
    axes[1].bar(("Full BP", "Stop random Q\nfeedback", "Stop inter-step\nfeedback"), values)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Gradient norm: same forward outputs (L32)")
    for index, value in enumerate(values):
        axes[1].text(index, value * 1.1, f"{value:.2f}", ha="center")
    figure.tight_layout()
    figure.savefig(args.output / "gradient_controls.png", dpi=140)
    plt.close(figure)
    print("Diagnostic summary figures saved.", flush=True)


if __name__ == "__main__":
    main()
