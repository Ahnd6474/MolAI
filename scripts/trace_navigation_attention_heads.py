"""Checkpoint-only headwise SDPA VJP audit; never updates model weights.

Q-gradient ablations are diagnostic only. Forward values stay unchanged; a
projected Q head's backward path (including its Q projection weights) is cut.
"""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn import functional as F
from trace_navigation_gradient_path import measure
from train_cloud import _make_schedule

from molai.models.relative_rollout import build_fresh_token_relative_models

TARGET = "refine_blocks.2.attention.attention"


@contextmanager
def probe_heads(cloud, rows, cut_heads=()):
    state = {"step": 0, "module": None}
    original_sdpa = F.scaled_dot_product_attention
    originals = []

    def begin(_module, _args):
        state["step"] += 1

    handle = cloud.register_forward_pre_hook(begin)
    for name, module in cloud.named_modules():
        if not (name.startswith("refine_blocks.") and name.endswith(".attention.attention")):
            continue
        original = module.forward
        originals.append((module, original))

        def forward(*args, name=name, module=module, original=original, **kwargs):
            previous = state["module"]
            state["module"] = (name, module)
            try:
                return original(*args, **kwargs)
            finally:
                state["module"] = previous

        module.forward = forward

    def sdpa(q, k, v, *args, **kwargs):
        if state["module"] is None:
            return original_sdpa(q, k, v, *args, **kwargs)
        name, module = state["module"]
        step = state["step"]
        row = {"module": name, "step": step, "heads": module.num_heads,
               "head_dim": q.shape[-1], "queries": q.shape[-2], "keys": k.shape[-2]}
        rows.append(row)
        head_dim = q.shape[-1]
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            sampled_q = q.detach()[:, :, ::max(1, q.shape[-2] // 1024)].float()
            keys = k.detach().float()
            logits = sampled_q @ keys.transpose(-1, -2) / head_dim**.5
            probability = logits.softmax(-1)
            entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1)
            row["forward_per_head"] = [{
                "head": head,
                "q_rms": float(q[:, head].detach().float().square().mean().sqrt()),
                "k_rms": float(keys[:, head].square().mean().sqrt()),
                "v_rms": float(v[:, head].detach().float().square().mean().sqrt()),
                "logit_std": float(logits[:, head].std(-1).mean()),
                "top1_probability": float(probability[:, head].max(-1).values.mean()),
                "entropy_nats": float(entropy[:, head].mean()),
            } for head in range(module.num_heads)]

        def capture_q(gradient):
            g = gradient.detach().float()
            norms = g.square().sum((0, 2, 3)).sqrt()
            row["q_grad_norm_per_head"] = norms.tolist()
            # Wq^T maps each head's Q-gradient to the same pre-projection query
            # space. Contributions can then be compared and summed correctly.
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                weights = module.in_proj_weight[:module.embed_dim].detach().float()
                contributions = [g[:, h] @ weights[h*head_dim:(h+1)*head_dim]
                                 for h in range(module.num_heads)]
                flat = torch.stack([value.flatten() for value in contributions])
                lengths = flat.norm(dim=1)
                cosine = (flat @ flat.T) / (lengths[:, None] * lengths[None, :]).clamp_min(1e-30)
                row["preprojection_query_contribution_norms"] = lengths.tolist()
                row["query_contribution_cosine"] = cosine.tolist()
                row["preprojection_query_combined_norm"] = float(flat.sum(0).norm())
            if name == TARGET and step == 2 and cut_heads:
                result = gradient.clone()
                result[:, list(cut_heads)] = 0
                return result
            return None

        def capture_input(label):
            def capture(gradient):
                row[label + "_grad_norm_per_head"] = gradient.detach().float().square().sum((0, 2, 3)).sqrt().tolist()
            return capture

        q.register_hook(capture_q)
        k.register_hook(capture_input("k"))
        v.register_hook(capture_input("v"))
        output = original_sdpa(q, k, v, *args, **kwargs)
        output.register_hook(capture_input("attention_output"))
        return output

    F.scaled_dot_product_attention = sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original_sdpa
        handle.remove()
        for module, original in originals:
            module.forward = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    schedule = _make_schedule(bundle["manifest"], config, torch.device("cuda"))
    report = {"checkpoint": str(args.checkpoint.resolve()), "note":
              "Local eager train-mode batch 1, fixed per-record seed, no optimizer. "
              "Head indices are zero-based. Interventions cut projected Q gradients "
              "only in final CvT of step 2; include Q projection parameter path. "
              "VJP gains refer to observed loss directions, not Jacobian spectral norms.",
              "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, level, ablate in ((33, 4, True), (0, 0, True), (88, 32, False), (122, 64, False)):
        record = next(r for r in bundle["samples"] if r["index"] == index)
        baseline_fields = None
        modes = [("bf16", ())]
        if ablate:
            modes += [("bf16", (head,)) for head in range(4)]
            modes += [("bf16", (0, 1, 2, 3)), ("fp32", ())]
            modes += [("fp32", (head,)) for head in range(4)]
        for precision, cut_heads in modes:
            heads = []
            with probe_heads(cloud, heads, cut_heads):
                row = measure(cloud, encoder, record, schedule, config, level, precision, "none")
            fields = row.pop("fields")
            row["head_trace"] = heads
            row["cut_projected_q_heads_step2"] = list(cut_heads)
            if not cut_heads:
                baseline_fields = fields
            row["max_forward_difference_from_precision_baseline"] = float((fields-baseline_fields).abs().max())
            if not heads:
                raise RuntimeError("Target SDPA was not observed")
            report["cases"].append(row)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps({"index": index, "level": level, "precision": precision,
                              "cut_heads": cut_heads, "grad_norm": row["grad_norm"],
                              "forward_difference": row["max_forward_difference_from_precision_baseline"]}), flush=True)
    print("Report saved: " + str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
