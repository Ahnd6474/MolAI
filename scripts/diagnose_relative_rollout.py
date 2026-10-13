"""Local paired diagnostics: no optimizer, no checkpoint writes, no training edits."""
from __future__ import annotations

import argparse
import gc
import json
import math
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from train_absolute_cloud import RaggedSpectrumTokenRunner, _condition_tokens
from train_cloud import RaggedSpectrumConditionRunner, _condition_from_batch, _make_schedule
from train_relative_image_rollout import rollout_losses, trajectory_targets

from molai.data import collate_spectrum_field_batch
from molai.models.losses import FullBandEnergyDistance
from molai.models.relative_rollout import (
    RelativeImageRolloutCloud,
    build_relative_models,
    build_token_relative_models,
)


def autocast(precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def metrics(fields, clean):
    fields, clean = fields.float(), clean.float()
    truth = clean.flatten(1)
    prediction = fields.flatten(2)
    centered_truth = truth - truth.mean(-1, keepdim=True)
    centered_prediction = prediction - prediction.mean(-1, keepdim=True)
    corr = (centered_prediction * centered_truth[:, None]).sum(-1) / (
        centered_prediction.norm(dim=-1) * centered_truth.norm(dim=-1)[:, None]).clamp_min(1e-12)
    error = (fields - clean[:, None]).square().mean((2, 3, 4))
    return {"zero_mse": float(error[:, 0].mean()), "random_mse": float(error[:, 1:].mean()),
        "zero_corr": float(corr[:, 0].mean()), "random_corr": float(corr[:, 1:].mean()),
        "best_random_corr": float(corr[:, 1:].max(-1).values.mean()),
        "zero_rms": float(fields[:, 0].square().mean().sqrt()),
        "random_rms": float(fields[:, 1:].square().mean().sqrt()),
        "clean_rms": float(clean.square().mean().sqrt()),
        "diversity": float(fields[:, 1:].var(1, unbiased=False).mean())}


def batch_for(record):
    return collate_spectrum_field_batch([record["item"]], peak_chunk_size=256)


def encode(encoder, batch, device, token_mode):
    if token_mode:
        return _condition_tokens(RaggedSpectrumTokenRunner(encoder), batch, device)
    return _condition_from_batch(RaggedSpectrumConditionRunner(encoder), batch, device), None


def load_models(source, weights, device, token_mode):
    dim = source["condition_encoder"]["metadata_embedding.0.weight"].shape[1]
    if token_mode:
        cloud, encoder, _ = build_token_relative_models(source, dim, gradient_checkpointing=True)
        cloud.load_state_dict(weights["cloud"], strict=True)
        encoder.load_state_dict(weights["condition_encoder"], strict=True)
        cloud.cache_condition_kv = True
    else:
        cloud, encoder = build_relative_models(source, dim, gradient_checkpointing=True)
    return cloud.to(device), encoder.to(device)


def group(name):
    if name.startswith("encoder."):
        return "ms_encoder"
    return name.split(".")[1]


def gradient_summary(named, gradients):
    squared = defaultdict(float)
    parameters = {}
    for (name, _), gradient in zip(named, gradients, strict=True):
        if gradient is not None:
            norm = float(gradient.float().norm())
            squared[group(name)] += norm * norm
            parameters[name] = norm
    return {"norm": math.sqrt(sum(squared.values())),
        "by_module": {name: math.sqrt(value) for name, value in squared.items()},
        "largest_parameters": sorted(parameters.items(), key=lambda entry: -entry[1])[:10]}


def gradient_cosine(first, second):
    dot = sum(float((left.float() * right.float()).sum())
              for left, right in zip(first, second, strict=True) if left is not None and right is not None)
    a = sum(float(value.float().square().sum()) for value in first if value is not None)
    b = sum(float(value.float().square().sum()) for value in second if value is not None)
    return dot / max(math.sqrt(a * b), 1e-30)


def state_comparison(reference, current):
    grouped = defaultdict(lambda: [0.0, 0.0])
    for name, new in current.items():
        if name in reference and reference[name].shape == new.shape and new.is_floating_point():
            old = reference[name].float()
            key = name.split(".")[0]
            grouped[key][0] += float((new.float() - old).square().sum())
            grouped[key][1] += float(old.square().sum())
    return {key: math.sqrt(delta / max(base, 1e-30)) for key, (delta, base) in grouped.items()}


def write_report(output, report):
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def inference(cloud, encoder, records, schedule, config, device, output, label):
    cloud.eval()
    encoder.eval()
    token_mode = label != "legacy19"
    rollout = RelativeImageRolloutCloud(cloud)
    result, pictures = [], []
    for number, record in enumerate(records):
        batch = batch_for(record)
        clean = batch["field"].to(device)
        torch.manual_seed(100003 + number)
        initial, targets, levels = trajectory_targets(schedule, clean, torch.tensor([64], device=device), 20, 4, config)
        torch.manual_seed(314159 + number)
        noise = torch.randn(1, 20, 5, 64, 64, device=device)
        with torch.no_grad(), autocast("bf16"):
            condition, mask = encode(encoder, batch, device, token_mode)
            fields = rollout(initial, condition, condition_mask=mask, steps=20, noise=noise)
        rows = []
        distance = FullBandEnergyDistance(include_target_constant=True)
        for step in range(20):
            row = metrics(fields[:, step], clean)
            row.update({"step": step + 1, "target_level": int(levels[0, step]),
                "energy_with_target_constant": float(distance(fields[:, step, 1:], targets[:, step], initial)),
                "target_diversity": float(targets[:, step].var(1, unbiased=False).mean()),
                "conditional_mean_mse": float((fields[:, step, 1:].mean(1) - targets[:, step].mean(1)).square().mean())})
            rows.append(row)
        result.append({"index": record["index"], "smiles": record["item"]["smiles"], "steps": rows})
        pictures.append({"clean": clean.cpu(), "initial": initial.cpu(), "fields": fields.float().cpu()})
        print(json.dumps({"phase": "inference", "model": label, "sample": number,
            "step4": rows[3], "step20": rows[-1]}), flush=True)
        torch.save(pictures, output / f"{label}_frames.pt")
        del fields, targets, condition
    figure, axes = plt.subplots(len(records) * 2, 7, figsize=(12, 3.25 * len(records)), squeeze=False)
    for number, picture in enumerate(pictures):
        clean = picture["clean"][0, 0]
        scale = float(clean.abs().quantile(.995).clamp_min(1e-6))
        for branch in range(2):
            panels = [clean, picture["initial"][0, 0]] + [picture["fields"][0, step - 1, branch, 0]
                for step in (1, 4, 8, 12, 20)]
            row_index = number * 2 + branch
            for column, panel in enumerate(panels):
                axes[row_index, column].imshow(panel, cmap="coolwarm", vmin=-scale, vmax=scale)
                axes[row_index, column].set_xticks([])
                axes[row_index, column].set_yticks([])
                if row_index == 0:
                    axes[row_index, column].set_title(["GT", "Input L64", "Step 1", "Step 4", "Step 8", "Step 12", "Step 20"][column])
            axes[row_index, 0].set_ylabel(f"#{number + 1} " + ("Zero" if branch == 0 else "Random 1"))
    figure.suptitle(f"{label} | fixed inputs and seeds | unchanged GT color scale per molecule")
    figure.tight_layout()
    figure.savefig(output / f"{label}_trajectory.png", dpi=120)
    plt.close(figure)
    return result


def gradient_diagnostic(cloud, encoder, record, schedule, config, device, precision, detach=False, start_level=32):
    cloud.train()
    encoder.eval()  # Keep stochastic layers disabled for paired comparisons.
    batch = batch_for(record)
    clean = batch["field"].to(device)
    torch.manual_seed(1729)
    initial, targets, _ = trajectory_targets(schedule, clean, torch.tensor([start_level], device=device), 4, 4, config)
    initial.requires_grad_()
    torch.manual_seed(2718)
    noise = torch.randn(1, 4, 5, 64, 64, device=device)
    noise[:, :, 0] = 0
    named = [("cloud." + key, value) for key, value in cloud.named_parameters() if value.requires_grad]
    named += [("encoder." + key, value) for key, value in encoder.named_parameters() if value.requires_grad]
    parameters = [value for _, value in named]
    frames = []
    with autocast(precision):
        token_mode = hasattr(cloud, "prepare_condition")
        condition, mask = encode(encoder, batch, device, token_mode)
        prepared = cloud.prepare_condition(condition, mask) if token_mode else None
        kwargs = {"condition_mask": mask, "prepared_condition": prepared} if token_mode else {}
        current = initial
        for step in range(4):
            if step == 0:
                fields, _, _ = cloud(current, condition, samples=5, noise=noise[:, step], **kwargs)
            else:
                current = current.detach() if detach else current
                later_condition = condition if token_mode else condition.repeat_interleave(5, 0)
                fields, _, _ = cloud(current.flatten(0, 1), later_condition, samples=1,
                    noise=noise[:, step].reshape(5, 1, 64, 64), **kwargs)
                fields = fields[:, 0].reshape(1, 5, 1, 128, 128)
            frames.append(fields)
            current = fields
        stacked = torch.stack(frames, 1)
        losses = rollout_losses(stacked, targets, initial, clean,
            FullBandEnergyDistance(include_target_constant=False), .05)
    summaries = {}
    for prefix in ((4,) if detach else (1, 2, 4)):
        loss = (losses["step_u_statistic"][:prefix] + .05 * losses["step_zero_token_clean_mse"][:prefix]).mean()
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
        summaries[f"prefix_{prefix}"] = {"loss": float(loss.detach()), **gradient_summary(named, gradients)}
        del gradients
    if not detach:
        u_gradients = torch.autograd.grad(losses["u_statistic"], parameters, allow_unused=True, retain_graph=True)
        summaries["u_stat"] = gradient_summary(named, u_gradients)
        z_gradients = torch.autograd.grad(.05 * losses["zero_token_clean_mse"], parameters, allow_unused=True, retain_graph=True)
        summaries["weighted_zero_mse"] = gradient_summary(named, z_gradients)
        summaries["u_zero_gradient_cosine"] = gradient_cosine(u_gradients, z_gradients)
        del u_gradients, z_gradients
        frame_gradients = torch.autograd.grad(losses["step_u_statistic"][-1] + .05 * losses["step_zero_token_clean_mse"][-1],
            [initial, *frames], allow_unused=True)
        summaries["last_loss_input_and_frame_gradient_rms"] = [
            float(value.float().square().mean().sqrt()) if value is not None else None for value in frame_gradients]
    summaries["per_step_losses"] = {key: value.detach().float().cpu().tolist() for key, value in losses.items()
        if key.startswith("step_")}
    summaries["per_step_metrics"] = [metrics(frame.detach(), clean) for frame in frames]
    print(json.dumps({"phase": "gradients", "precision": precision, "detach": detach, "index": record["index"], **summaries}), flush=True)
    return summaries


def condition_sensitivity(cloud, encoder, records, schedule, device):
    cloud.eval()
    encoder.eval()
    torch.manual_seed(90125)
    initial = torch.randn(1, 1, 128, 128, device=device) * schedule.noise_scale
    initial -= initial.mean((-2, -1), keepdim=True)
    noise = torch.randn(1, 20, 5, 64, 64, device=device)
    outputs = []
    with torch.no_grad(), autocast("bf16"):
        rollout = RelativeImageRolloutCloud(cloud)
        for record in records:
            condition, mask = encode(encoder, batch_for(record), device, True)
            outputs.append(rollout(initial, condition, condition_mask=mask, steps=20, noise=noise).float().cpu())
    base = outputs[0]
    result = []
    for step in (1, 4, 20):
        deltas = [(value[:, step - 1] - base[:, step - 1]).square().mean().sqrt() for value in outputs[1:]]
        result.append({"step": step, "output_rms": float(base[:, step - 1].square().mean().sqrt()),
                       "changed_ms_output_delta_rms": [float(value) for value in deltas]})
    print(json.dumps({"phase": "changed_ms_same_input_same_noise", "result": result}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--epoch", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "inference", "gradients"), default="all")
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    aligned = torch.load(args.aligned, map_location="cpu", weights_only=False)
    epoch = torch.load(args.epoch, map_location="cpu", weights_only=False)
    bundle = torch.load(args.samples, map_location="cpu", weights_only=False)
    config = epoch["config"]
    validation = [record for record in bundle["samples"] if record["split"] == "validation"][:args.count]
    training = [record for record in bundle["samples"] if record["split"] == "train"]
    schedule = _make_schedule(bundle["manifest"], config, device)
    report = {"source_epoch": int(source["epoch"]) + 1, "epoch_global_step": epoch["global_step"],
        "torch": torch.__version__, "device": torch.cuda.get_device_name(),
        "note": "Local eager execution; paired BF16 seeds; FP32 and detach controls for gradients. No optimizer updates.",
        "cloud_weight_relative_change_aligned_to_epoch1": state_comparison(aligned["cloud"], epoch["cloud"]),
        "ms_weight_relative_change_aligned_to_epoch1": state_comparison(aligned["condition_encoder"], epoch["condition_encoder"])}
    report["random_gate"] = {key: float(value["cloud"]["noise_attention.gate"])
                             for key, value in (("legacy19", source), ("aligned", aligned), ("epoch1", epoch))}
    write_report(args.output, report)
    if args.phase in ("all", "inference"):
        for label, weights in (("legacy19", source), ("aligned", aligned), ("epoch1", epoch)):
            cloud, encoder = load_models(source, weights, device, label != "legacy19")
            report[label] = inference(cloud, encoder, validation, schedule, config, device, args.output, label)
            if label == "epoch1":
                report["ms_sensitivity"] = condition_sensitivity(cloud, encoder, validation, schedule, device)
            write_report(args.output, report)
            del cloud, encoder
            gc.collect()
            torch.cuda.empty_cache()
    if args.phase in ("all", "gradients"):
        report["gradients"] = {}
        for label, weights in (("aligned", aligned), ("epoch1", epoch)):
            cloud, encoder = load_models(source, weights, device, True)
            report["gradients"][label] = {}
            for record in (validation[0], training[0]):
                key = record["split"] + "_" + str(record["index"])
                report["gradients"][label][key] = gradient_diagnostic(cloud, encoder, record, schedule, config, device, "bf16")
                write_report(args.output, report)
            if label == "epoch1":
                report["gradients"][label]["validation_fp32"] = gradient_diagnostic(cloud, encoder, validation[0], schedule, config, device, "fp32")
                report["gradients"][label]["validation_detach_control"] = gradient_diagnostic(cloud, encoder, validation[0], schedule, config, device, "bf16", True)
            write_report(args.output, report)
            del cloud, encoder
            gc.collect()
            torch.cuda.empty_cache()
    print("Diagnostics completed: " + str(args.output / "report.json"), flush=True)


if __name__ == "__main__":
    main()
