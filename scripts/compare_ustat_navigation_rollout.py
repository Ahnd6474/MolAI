"""Local, matched-input checkpoint comparison; no training or weight updates."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib
import torch
from PIL import Image, ImageDraw
from train_absolute_cloud import RaggedSpectrumTokenRunner, _condition_tokens
from train_cloud import _make_schedule

from molai.data import collate_spectrum_field_batch
from molai.models.relative_rollout import (
    RelativeImageRolloutCloud,
    build_fresh_token_relative_models,
)


def restore(path):
    source = torch.load(path, map_location="cpu", weights_only=False)
    config = source["config"]
    cloud, encoder, _ = build_fresh_token_relative_models(
        config, source["condition_encoder"]["metadata_embedding.0.weight"].shape[1],
        gradient_checkpointing=False,
    )
    # The token-converted legacy checkpoint retains its unused 384d readout;
    # fresh token models have a 64d readout. Preserve the saved topology exactly.
    width = source["cloud"]["readout.1.weight"].shape[0]
    if cloud.readout[1].out_features != width:
        cloud.readout[1] = torch.nn.Linear(cloud.dim, width)
    cloud.load_state_dict(source["cloud"], strict=True)
    encoder.load_state_dict(source["condition_encoder"], strict=True)
    metadata = {"checkpoint": str(path.resolve()), "saved_epoch_zero_based": source["epoch"],
                "global_step": source["global_step"], "config": config,
                "source_checkpoint": source.get("source_checkpoint"),
                "strict_load": True}
    return cloud.cuda().eval(), encoder.cuda().eval(), metadata


def scores(fields, clean):
    truth = clean.float().flatten()
    flattened = fields.float().flatten(2)
    mse = (flattened - truth).square().mean(-1)
    centered = flattened - flattened.mean(-1, keepdim=True)
    truth = truth - truth.mean()
    correlation = (centered * truth).sum(-1) / (centered.norm(dim=-1) * truth.norm()).clamp_min(1e-12)
    return {"zero_mse": mse[:, 0].tolist(), "random_mse": mse[:, 1:].mean(-1).tolist(),
            "best_random_mse": mse[:, 1:].min(-1).values.tolist(),
            "zero_correlation": correlation[:, 0].tolist(),
            "random_correlation": correlation[:, 1:].mean(-1).tolist(),
            "best_random_correlation": correlation[:, 1:].max(-1).values.tolist(),
            "zero_rms": flattened[:, 0].square().mean(-1).sqrt().tolist(),
            "random_rms": flattened[:, 1:].square().mean((1, 2)).sqrt().tolist(),
            "random_diversity": fields[:, 1:].float().var(dim=1, unbiased=False).flatten(1).mean(-1).tolist()}


def render(record, model_fields, initial, directory):
    clean = record["item"]["field"].float()
    scale = clean.abs().flatten().quantile(.995).clamp_min(1e-6)
    selected = [1, 2, 4, 8, 12, 16, 20]
    for name, fields in model_fields.items():
        canvas = Image.new("RGB", (1232, 672), "white")
        draw = ImageDraw.Draw(canvas)
        for col, title in enumerate(["GT", "Input", *[f"Step {s}" for s in selected]]):
            draw.text((80+128*col+30, 10), title, fill="black")
        for path in range(5):
            panels = torch.stack([clean, initial, *[fields[s-1, path] for s in selected]])[:, 0]
            values = ((panels / scale).clamp(-1, 1) + 1) / 2
            rgb = (matplotlib.colormaps["coolwarm"](values.numpy())[..., :3] * 255).round().astype("uint8")
            for col, tile in enumerate(rgb):
                canvas.paste(Image.fromarray(tile), (80+128*col, 32+128*path))
            draw.text((5, 85+128*path), "Zero" if path == 0 else f"Random {path}", fill="black")
        canvas.save(directory / f"{name}-molecule-{record['index']:03d}.png")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ustat", type=Path, required=True)
    parser.add_argument("--navigation", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    bundle = torch.load(args.samples, map_location="cpu", weights_only=False)
    records = [r for r in bundle["samples"] if r["index"] in (33, 88, 122, 177)]
    assert len(records) == 4
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"note": "4 fixed real validation molecules, not full-dataset validation. "
              "Both models strict-loaded, eval-mode local RTX 5060, BF16, no compile, "
              "same pure-noise initial image and random tokens for 20 steps, no variance correction. "
              "Different pretraining, architecture, LR and supervision prevent causal loss-speed attribution. "
              "Image colors fixed to each GT absolute 99.5% quantile, shared across models.",
              "models": {}, "molecules": {str(r["index"]): {} for r in records}}
    all_fields, all_initial = {}, {}
    reference_initial, reference_noise = {}, {}
    for name, checkpoint in (("ustat_mse", args.ustat), ("navigation_gated", args.navigation)):
        cloud, encoder, metadata = restore(checkpoint)
        report["models"][name] = metadata
        schedule = _make_schedule(bundle["manifest"], metadata["config"], torch.device("cuda"))
        for record in records:
            batch = collate_spectrum_field_batch([record["item"]], peak_chunk_size=256)
            clean = batch["field"].cuda()
            torch.manual_seed(923001 + record["index"])
            initial = torch.randn_like(clean)
            initial -= initial.mean((-2, -1), keepdim=True)
            initial *= schedule.noise_scale
            noise = torch.randn(1, 20, 5, cloud.noise_token_count, cloud.noise_token_dim, device="cuda")
            noise[:, :, 0].zero_()
            if name == "ustat_mse":
                reference_initial[record["index"]] = initial.cpu()
                reference_noise[record["index"]] = noise.cpu()
            else:
                torch.testing.assert_close(initial.cpu(), reference_initial[record["index"]], rtol=0, atol=0)
                torch.testing.assert_close(noise.cpu(), reference_noise[record["index"]], rtol=0, atol=0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                condition, mask = _condition_tokens(RaggedSpectrumTokenRunner(encoder), batch, torch.device("cuda"))
                fields = RelativeImageRolloutCloud(cloud)(initial, condition, condition_mask=mask, steps=20, noise=noise)
            fields = fields[0].float().cpu()
            assert torch.isfinite(fields).all()
            report["molecules"][str(record["index"])][name] = scores(fields, clean.cpu())
            all_fields.setdefault(record["index"], {})[name] = fields
            all_initial[record["index"]] = initial[0].float().cpu()
            print(json.dumps({"model": name, "index": record["index"], "final":
                              {k: v[-1] for k, v in report["molecules"][str(record["index"])][name].items()}}), flush=True)
        del cloud, encoder, fields, clean, initial, noise, condition
        gc.collect()
        torch.cuda.empty_cache()
    report["mean_over_4_molecules"] = {}
    for name in report["models"]:
        rows = [report["molecules"][str(r["index"])][name] for r in records]
        report["mean_over_4_molecules"][name] = {
            key: torch.tensor([r[key] for r in rows]).mean(0).tolist() for key in rows[0]}
    current, levels, radii = 64, [], []
    for _ in range(20):
        jump = 1 if current <= 8 else 2 if current <= 16 else 4 if current <= 32 else 8
        after = max(0, current-jump)
        x = torch.zeros(1, 1, 128, 128, device="cuda")
        _, variance = schedule.target_mean_and_variance(x, x, torch.tensor([current], device="cuda"),
                                                        torch.tensor([after], device="cuda"))
        radii.append(float(schedule.noise_scale * (variance * (1-1/128**2)).sqrt()[0]))
        levels.append(after)
        current = after
    report["after_levels"] = levels
    report["posterior_radius_rms_before_token_scaling"] = radii
    for record in records:
        render(record, all_fields[record["index"]], all_initial[record["index"]], args.output)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved " + str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
