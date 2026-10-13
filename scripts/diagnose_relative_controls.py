"""Forward-identical derivative controls for relative image recurrence."""
import argparse
import gc
import json
from pathlib import Path

import torch
from diagnose_relative_rollout import gradient_diagnostic, load_models
from train_cloud import _make_schedule


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
    config = epoch["config"]
    schedule = _make_schedule(bundle["manifest"], config, device)
    record = next(record for record in bundle["samples"] if record["split"] == "validation")
    report = {}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    cloud, encoder = load_models(source, source, device, False)
    report["legacy19_level32"] = gradient_diagnostic(cloud, encoder, record, schedule, config, device, "bf16")
    save()
    del cloud, encoder
    gc.collect()
    torch.cuda.empty_cache()
    cloud, encoder = load_models(source, epoch, device, True)
    for level in (0, 64):
        report[f"epoch1_level{level}"] = gradient_diagnostic(cloud, encoder, record, schedule, config, device, "bf16", start_level=level)
        save()
    original_norm = cloud.noise_attention.query_norm.forward

    def stop_query_feedback(value):
        # Same forward values and attention-parameter derivatives. Only cut dQ/dx.
        return original_norm(value.detach())

    cloud.noise_attention.query_norm.forward = stop_query_feedback
    report["epoch1_stop_random_query_feedback_level32"] = gradient_diagnostic(cloud, encoder, record, schedule, config, device, "bf16")
    cloud.noise_attention.query_norm.forward = original_norm
    save()
    print("Derivative controls complete: " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
