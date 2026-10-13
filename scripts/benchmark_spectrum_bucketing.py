"""Report exact padding operation counts, not a throughput prediction."""

import argparse
import json
from pathlib import Path

import torch
from train_absolute_cloud import _postpool_peak_token_counts

from molai.data import ShardShuffleSampler, SpectrumFieldDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--replicas", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    dataset = SpectrumFieldDataset(args.data)
    counts = _postpool_peak_token_counts(dataset, 2, torch.device("cpu"), False, 0)
    seed = int(dataset.manifest.get("seed", 0))
    for bucketed in (False, True):
        total_tokens = 0
        squared_tokens = 0
        padded_tokens = 0
        used_indices = set()
        for rank in range(args.replicas):
            sampler = ShardShuffleSampler(
                dataset, seed=seed, rank=rank, replicas=args.replicas,
                batch_size=args.batch_size, split="train", validation_fraction=0.02,
                split_seed=seed + 17, eligible_mask=counts <= 8192,
                bucket_lengths=counts if bucketed else None,
            )
            sampler.set_epoch(3)
            indices = list(sampler)
            used_indices.update(indices)
            for start in range(0, len(indices), args.batch_size):
                lengths = counts[indices[start:start + args.batch_size]]
                maximum = int(lengths.max())
                total_tokens += int(lengths.sum())
                padded_tokens += len(lengths) * maximum
                squared_tokens += len(lengths) * maximum**2
        print(json.dumps({
            "bucketed": bucketed, "unique_molecules": len(used_indices),
            "actual_tokens": total_tokens, "padded_tokens": padded_tokens,
            "padding_fraction": 1 - total_tokens / padded_tokens,
            "self_attention_B_K_squared": squared_tokens,
        }), flush=True)


if __name__ == "__main__":
    main()
