"""Layerwise local VJP and forward-identical backward ablations, checkpoint only.

No optimizer updates. All interventions apply only to the diagnostic process.
"""
from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
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


def norm(value):
    return float(value.detach().float().norm()) if value is not None else 0.


def measure(cloud, encoder, record, schedule, config, level, precision, intervention, trace=False):
    cloud.zero_grad(set_to_none=True)
    encoder.zero_grad(set_to_none=True)
    torch.manual_seed(1729 + record["index"])
    device = next(cloud.parameters()).device
    batch = collate_spectrum_field_batch([record["item"]], peak_chunk_size=256)
    clean = batch["field"].to(device)
    start = torch.tensor([level], device=device)
    initial, levels = rollout_initial_and_levels(schedule, clean, start, 2, config)
    noise = sample_rollout_noise(initial, cloud, 2, 4)
    stage = {"step": 0}
    calls, rows, handles = defaultdict(list), [], []
    replaced_forwards = []
    for name, module in cloud.named_modules():
        selected = name == "refine_blocks.2.attention.attention"
        all_refine = name.startswith("refine_blocks.") and name.endswith(".attention.attention")
        if ((intervention in ("math_last", "fp32_last") and selected)
                or (intervention == "math_all_refine" and all_refine)):
            original = module.forward
            replaced_forwards.append((module, original))

            def precise_forward(*args, original=original, **kwargs):
                if intervention.startswith("math"):
                    with sdpa_kernel(SDPBackend.MATH):
                        return original(*args, **kwargs)
                dtype = args[0].dtype
                with torch.autocast("cuda", enabled=False):
                    result = original(*(v.float() if isinstance(v, torch.Tensor) else v for v in args), **kwargs)
                return tuple(v.to(dtype) if isinstance(v, torch.Tensor) else v for v in result)

            module.forward = precise_forward

    def begin(_module, _args):
        stage["step"] += 1

    handles.append(cloud.register_forward_pre_hook(begin))

    def intervene(name, module):
        def before(_module, args):
            if stage["step"] != 2:
                return None
            if intervention == "detach_reencode" and name == "field_encoder":
                return (args[0].detach(), *args[1:])
            if intervention == "detach_noise_amplitude" and name == "noise_attention":
                return (*args[:2], args[2].detach())
            if intervention.startswith("detach_norm:") and name == intervention.split(":", 1)[1]:
                return (args[0].detach(), *args[1:])
            if intervention == "detach_all_norm_inputs" and isinstance(module, torch.nn.LayerNorm):
                return (args[0].detach(), *args[1:])
            if intervention.startswith("qscale:"):
                _, selected, factor = intervention.split(":")
                if name == selected:
                    value = args[0]
                    scaled = value.detach() + float(factor) * (value - value.detach())
                    return (scaled, *args[1:])
            if intervention.startswith("detach_kv:") and name == intervention.split(":", 1)[1]:
                return (args[0], args[1].detach(), args[2].detach(), *args[3:])
            return None
        return before

    def forward(name):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(value, torch.Tensor):
                return
            row = {"module": name or "cloud", "type": type(module).__name__, "step": stage["step"],
                   "input_rms": norm(inputs[0]) / inputs[0].numel()**.5,
                   "output_rms": norm(value) / value.numel()**.5}
            if isinstance(module, torch.nn.LayerNorm):
                dims = tuple(range(inputs[0].ndim - len(module.normalized_shape), inputs[0].ndim))
                variance = inputs[0].detach().float().var(dims, unbiased=False)
                inv = (variance + module.eps).rsqrt().flatten()
                row.update(eps=module.eps, variance_min=float(variance.min()),
                           inv_std_median=float(inv.median()), inv_std_p99=float(inv.quantile(.99)),
                           inv_std_max=float(inv.max()), gamma_max=float(module.weight.detach().abs().max())
                           if module.weight is not None else 1.)
            if isinstance(module, torch.nn.MultiheadAttention) and name == "refine_blocks.2.attention.attention":
                with torch.no_grad():
                    dim = module.embed_dim
                    q, k, v = inputs[:3]
                    stride = max(1, q.shape[1] // 1024)
                    q = q[:, ::stride]
                    bias = module.in_proj_bias
                    q = F.linear(q, module.in_proj_weight[:dim], None if bias is None else bias[:dim])
                    k = F.linear(k, module.in_proj_weight[dim:2*dim], None if bias is None else bias[dim:2*dim])
                    v = F.linear(v, module.in_proj_weight[2*dim:], None if bias is None else bias[2*dim:])
                    heads = module.num_heads
                    q = q.reshape(q.shape[0], -1, heads, dim//heads).transpose(1, 2).float()
                    k = k.reshape(k.shape[0], -1, heads, dim//heads).transpose(1, 2).float()
                    v = v.reshape(v.shape[0], -1, heads, dim//heads).transpose(1, 2).float()
                    with torch.autocast("cuda", enabled=False):
                        logits = (q @ k.transpose(-1, -2)) / (dim//heads)**.5
                    probability = logits.softmax(-1)
                    entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1)
                    row["sampled_attention_stats"] = {
                        "queries_sampled_per_path": q.shape[2], "kv_tokens": k.shape[2],
                        "q_rms": float(q.square().mean().sqrt()), "k_rms": float(k.square().mean().sqrt()),
                        "v_rms": float(v.square().mean().sqrt()),
                        "logit_std_across_keys_mean": float(logits.std(-1).mean()),
                        "top1_probability_mean": float(probability.max(-1).values.mean()),
                        "entropy_mean": float(entropy.mean()),
                        "projection_spectral_norms": [float(torch.linalg.matrix_norm(weight.detach().float(), ord=2))
                                                      for weight in module.in_proj_weight.chunk(3)]
                    }
            calls[name].append(row)
            rows.append(row)
        return hook

    def backward(name):
        def hook(_module, grad_inputs, grad_outputs):
            row = calls[name].pop()
            row["input_grad_norms"] = [norm(g) for g in grad_inputs]
            row["output_grad_norms"] = [norm(g) for g in grad_outputs]
            row["first_input_vjp_gain"] = row["input_grad_norms"][0] / max(row["output_grad_norms"][0], 1e-30)
            if intervention.startswith("cap_norm_vjp:") and row["step"] == 2:
                selected = intervention.split(":", 1)[1]
                if name == selected and grad_inputs[0] is not None:
                    factor = min(1., row["output_grad_norms"][0] / max(row["input_grad_norms"][0], 1e-30))
                    row["diagnostic_vjp_scale"] = factor
                    return (grad_inputs[0] * factor, *grad_inputs[1:])
            return None
        return hook

    for name, module in cloud.named_modules():
        if name:
            handles.append(module.register_forward_pre_hook(intervene(name, module)))
        if trace or intervention.startswith("cap_norm_vjp:"):
            if isinstance(module, (torch.nn.ModuleList, torch.nn.ModuleDict)):
                continue
            handles.append(module.register_forward_hook(forward(name)))
            handles.append(module.register_full_backward_hook(backward(name)))
    context = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
    with context:
        condition, mask = _condition_tokens(RaggedSpectrumTokenRunner(encoder), batch, device)
        fields = RelativeImageRolloutCloud(cloud)(initial, condition, condition_mask=mask, steps=2, noise=noise)
        _, mean, sigma = step_posterior_targets(schedule, clean, initial, fields, start, levels,
                                                sample_innovations=False)
        losses = vp_cloud_geometry_losses(fields, mean, sigma, noise[:, :, 1:])
        total = sum(losses[key] for key in TERMS)
    total.backward()
    for handle in handles:
        handle.remove()
    for module, original in replaced_forwards:
        module.forward = original
    names = [("cloud." + name, p) for name, p in cloud.named_parameters()]
    names += [("encoder." + name, p) for name, p in encoder.named_parameters()]
    grouped = defaultdict(float)
    largest = []
    for name, parameter in names:
        n = norm(parameter.grad)
        grouped[".".join(name.split(".")[:2])] += n*n
        largest.append((name, n))
    result = {"index": record["index"], "level": level, "precision": precision, "intervention": intervention,
              "loss": float(total.detach()), "losses": {key: float(losses[key].detach()) for key in TERMS},
              "grad_norm": sum(grouped.values())**.5,
              "without_random_gate_grad_norm": max(0., sum(grouped.values()) - norm(cloud.noise_attention.gate.grad)**2)**.5,
              "by_module": {key: value**.5 for key, value in grouped.items()},
              "largest_parameters": sorted(largest, key=lambda pair: -pair[1])[:8],
              "fields": fields.detach().float().cpu(), "layers": rows}
    print(json.dumps({key: result[key] for key in ("index", "level", "precision", "intervention", "loss", "grad_norm")}), flush=True)
    del total, losses, fields, condition
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=33)
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--backend-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = source["config"]
    dim = source["condition_encoder"]["metadata_embedding.0.weight"].shape[1]
    cloud, encoder, _ = build_fresh_token_relative_models(config, dim, gradient_checkpointing=False)
    cloud.load_state_dict(source["cloud"], strict=True)
    encoder.load_state_dict(source["condition_encoder"], strict=True)
    del source
    cloud, encoder = cloud.cuda().train(), encoder.cuda().train()
    bundle = torch.load(args.samples, map_location="cpu", weights_only=False)
    record = next(r for r in bundle["samples"] if r["index"] == args.index)
    schedule = _make_schedule(bundle["manifest"], config, torch.device("cuda"))
    report = {"checkpoint": str(args.checkpoint.resolve()), "note": "No optimizer; local eager train-mode, batch 1. "
              "Checkpointing disabled for layer hooks; baseline is checked against prior audit. "
              "Backward interventions preserve forward values; not proposed fixes.", "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def add(precision, intervention, trace=False):
        row = measure(cloud, encoder, record, schedule, config, args.level, precision, intervention, trace)
        fields = row.pop("fields")
        if report["cases"] and precision == "bf16":
            row["max_forward_difference_from_baseline"] = float((fields - baseline_fields).abs().max())
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return fields

    baseline_fields = add("bf16", "none", True)
    add("fp32", "none", True)
    if args.backend_only:
        for intervention in ("math_last", "math_all_refine", "fp32_last"):
            add("bf16", intervention, True)
        print("Report saved: " + str(args.output.resolve()), flush=True)
        return
    for intervention in ("detach_reencode", "detach_noise_amplitude", "detach_norm:noise_attention.query_norm",
                         "detach_all_norm_inputs"):
        add("bf16", intervention)
    candidates = sorted((r for r in report["cases"][0]["layers"]
                         if r["step"] == 2 and r["type"] == "LayerNorm" and "first_input_vjp_gain" in r
                         and r["output_grad_norms"][0] > 1e-8), key=lambda r: -r["first_input_vjp_gain"])[:4]
    for row in candidates:
        add("bf16", "cap_norm_vjp:" + row["module"])
    selected = "refine_blocks.2.attention.attention"
    for factor in (0., .1, .03):
        add("bf16", f"qscale:{selected}:{factor}")
    add("bf16", "detach_kv:" + selected)
    add("bf16", "qscale:refine_blocks.1.attention.attention:0")
    print("Report saved: " + str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
