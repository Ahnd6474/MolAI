"""Read-only local checkpoint gradient audit; no optimizer or training edits."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
from train_absolute_cloud import RaggedSpectrumTokenRunner, _condition_tokens
from train_cloud import _make_schedule
from train_relative_image_rollout import (
    rollout_initial_and_levels,
    sample_rollout_noise,
    step_posterior_targets,
)

from molai.data import collate_spectrum_field_batch
from molai.models.losses import vp_cloud_geometry_losses
from molai.models.relative_rollout import (
    RelativeImageRolloutCloud,
    build_fresh_token_relative_models,
)

TERMS = ("zero_mean_mse", "random_radius", "random_direction")


def vector_summary(vector, named):
    squared = defaultdict(float)
    largest, offset = [], 0
    for name, parameter in named:
        count = parameter.numel()
        norm2 = float(vector[offset:offset + count].square().sum())
        squared[".".join(name.split(".")[:2])] += norm2
        largest.append((name, math.sqrt(norm2)))
        offset += count
    return {"norm": float(vector.norm()),
            "by_module": {key: math.sqrt(value) for key, value in squared.items()},
            "largest_parameters": sorted(largest, key=lambda pair: -pair[1])[:6]}


def audit_case(cloud, encoder, record, schedule, config, level, seed, precision="bf16", detach=False):
    torch.manual_seed(seed)
    batch = collate_spectrum_field_batch([record["item"]], peak_chunk_size=256)
    device = next(cloud.parameters()).device
    clean = batch["field"].to(device)
    start = torch.tensor([level], device=device)
    initial, levels = rollout_initial_and_levels(schedule, clean, start, 2, config)
    noise = sample_rollout_noise(initial, cloud, 2, 4)
    named = [("cloud." + name, p) for name, p in cloud.named_parameters() if p.requires_grad]
    named += [("encoder." + name, p) for name, p in encoder.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    torch.cuda.reset_peak_memory_stats()
    frames = []

    def capture(_module, _inputs, output):
        frames.append(output[0])

    handle = cloud.register_forward_hook(capture)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
    with context:
        condition, mask = _condition_tokens(RaggedSpectrumTokenRunner(encoder), batch, device)
        if not detach:
            fields = RelativeImageRolloutCloud(cloud)(initial, condition, condition_mask=mask,
                                                     steps=2, noise=noise)
        else:
            first, _, _ = cloud(initial, condition, samples=5, noise=noise[:, 0], condition_mask=mask)
            second, _, _ = cloud(first.detach().flatten(0, 1), condition.repeat_interleave(5, 0),
                                samples=1, noise=noise[:, 1].reshape(5, 1, 64, 64),
                                condition_mask=mask.repeat_interleave(5, 0))
            fields = torch.stack((first, second[:, 0].reshape_as(first)), dim=1)
        _, mean, sigma = step_posterior_targets(schedule, clean, initial, fields, start, levels,
                                                sample_innovations=False)
        losses = vp_cloud_geometry_losses(fields, mean, sigma, noise[:, :, 1:])
    handle.remove()
    result = {"index": record["index"], "split": record["split"], "level": level,
              "target_levels": levels[0].tolist(), "sigma": sigma[0].tolist(),
              "precision": precision, "detach_feedback": detach, "seed": seed,
              "clean_mean": float(clean.mean()), "clean_rms": float(clean.square().mean().sqrt()),
              "losses": {key: float(losses[key].detach()) for key in TERMS},
              "per_step_losses": {key: losses["step_" + key].detach().cpu().tolist() for key in TERMS},
              "parameter_gradients": {}, "output_gradients": {}}
    direct_gradients = torch.autograd.grad(sum(losses[key] for key in TERMS), parameters,
                                           allow_unused=True, retain_graph=True)
    direct_vector = torch.cat([torch.zeros(p.numel()) if g is None else g.detach().float().flatten().cpu()
                               for p, g in zip(parameters, direct_gradients, strict=True)])
    result["parameter_gradients"]["direct_two_step_total"] = vector_summary(direct_vector, named)
    del direct_gradients, direct_vector
    second_loss = sum(losses["step_" + key][1] for key in TERMS)
    field_gradient, parent_gradient = torch.autograd.grad(second_loss, [fields, frames[0]],
                                                         allow_unused=True, retain_graph=True)
    result["second_loss_image_gradient"] = {
        "output_norm": float(field_gradient.float().norm()),
        "parent_norm": float(parent_gradient.float().norm()) if parent_gradient is not None else 0.,
        "parent_max_abs": float(parent_gradient.float().abs().max()) if parent_gradient is not None else 0.,
    }
    del field_gradient, parent_gradient
    first_vectors, averaged_vectors = {}, {}
    for step in range(2):
        for term in TERMS:
            loss = losses["step_" + term][step]
            gradients = torch.autograd.grad(loss, [fields, *parameters], allow_unused=True,
                                            retain_graph=True)
            output = gradients[0].detach().float()
            result["output_gradients"][f"step{step + 1}_{term}"] = {
                "norm": float(output.norm()), "max_abs": float(output.abs().max())}
            vector = torch.cat([torch.zeros(p.numel()) if g is None else g.detach().float().flatten().cpu()
                                for p, g in zip(parameters, gradients[1:], strict=True)])
            del gradients
            result["parameter_gradients"][f"step{step + 1}_{term}"] = vector_summary(vector, named)
            if step == 0:
                first_vectors[term] = vector
            else:
                averaged_vectors[term] = (first_vectors[term] + vector) * .5
    first_total = sum(first_vectors.values())
    second_total = sum(averaged_vectors.values()) * 2 - first_total
    full_total = sum(averaged_vectors.values())
    result["parameter_gradients"]["step1_total"] = vector_summary(first_total, named)
    result["parameter_gradients"]["step2_total"] = vector_summary(second_total, named)
    result["parameter_gradients"]["two_step_total"] = vector_summary(full_total, named)
    for term, vector in averaged_vectors.items():
        result["parameter_gradients"][f"two_step_{term}"] = vector_summary(vector, named)
    cosine = {}
    for i, term in enumerate(TERMS):
        for other in TERMS[i + 1:]:
            a, b = averaged_vectors[term], averaged_vectors[other]
            cosine[f"{term}__{other}"] = float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))
    result["term_gradient_cosines"] = cosine
    with torch.no_grad():
        offsets = (fields[:, :, 1:].float() - mean[:, :, 1:]).flatten(3)
        result["offset_rms_per_step_path"] = offsets.square().mean(-1).sqrt()[0].tolist()
        result["output_rms_per_step_path"] = fields.float().square().mean((3, 4, 5))[0].tolist()
    result["peak_allocated_GiB"] = torch.cuda.max_memory_allocated() / 2**30
    print(json.dumps({key: result[key] for key in ("index", "split", "level", "precision", "detach_feedback", "losses")}
                     | {"grad_norms": {key: value["norm"] for key, value in result["parameter_gradients"].items()
                                       if key.startswith("two_step") or key.endswith("total")}}), flush=True)
    frames.clear()
    del fields, losses, condition, first_vectors, averaged_vectors, first_total, second_total, full_total
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 4, 8, 16, 32, 48, 64])
    parser.add_argument("--controls", action="store_true", help="FP32 and detached-feedback controls at worst level")
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source.pop("optimizer", None)
    source.pop("scheduler", None)
    config = source["config"]
    if config["relative_image_rollout"]["loss_mode"] != "navigation":
        raise ValueError("expected the three-term-only Navigation checkpoint")
    bundle = torch.load(args.samples, map_location="cpu", weights_only=False)
    dim = source["condition_encoder"]["metadata_embedding.0.weight"].shape[1]
    cloud, encoder, _ = build_fresh_token_relative_models(config, dim, gradient_checkpointing=True)
    cloud.load_state_dict(source["cloud"], strict=True)
    encoder.load_state_dict(source["condition_encoder"], strict=True)
    with args.checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"epoch": int(source["epoch"]) + 1, "global_step": source["global_step"],
              "checkpoint": str(args.checkpoint.resolve()), "sha256": digest,
              "device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "note": "Local eager BF16, train mode incl. MS dropout, batch=1; no optimizer. No compiled/cache K/V. "
                      "Fixed per-molecule seed across levels. Original teacher detach retained. Not a population estimate.",
              "random_gate": float(cloud.noise_attention.gate.detach()), "cases": [], "controls": []}
    del source
    device = torch.device("cuda")
    cloud, encoder = cloud.to(device).train(), encoder.to(device).train()
    cloud.cache_condition_kv = False
    schedule = _make_schedule(bundle["manifest"], config, device)
    records = []
    for split in ("train", "validation"):
        records.extend([r for r in bundle["samples"] if r["split"] == split][:args.count])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for record in records:
        for level in args.levels:
            report["cases"].append(audit_case(cloud, encoder, record, schedule, config, level, 1729 + record["index"]))
            save()
    if args.controls:
        worst = max(report["cases"], key=lambda row: row["parameter_gradients"]["two_step_total"]["norm"])
        record = next(r for r in records if r["index"] == worst["index"])
        for precision, detach in (("fp32", False), ("bf16", True)):
            report["controls"].append(audit_case(cloud, encoder, record, schedule, config,
                                                  worst["level"], worst["seed"], precision, detach))
            save()
    print("Report saved: " + str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
