"""Export a small, exact train/validation sample bundle without GPU work."""
import argparse
from pathlib import Path

import torch

from molai.data import ShardShuffleSampler, SpectrumFieldDataset


def clone_record(item):
    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in item.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    del checkpoint
    dataset = SpectrumFieldDataset(args.data)
    seed = int(dataset.manifest.get("seed", 0))
    samples = []
    for split in ("validation", "train"):
        sampler = ShardShuffleSampler(dataset, split=split, shuffle=False, seed=seed,
            validation_fraction=float(config["training"]["validation_fraction"]), split_seed=seed + 17)
        count = 0
        for index in sampler:
            item = dataset[index]
            counts = item["peak_offsets"].diff()
            for _ in range(2):
                counts = (counts + 1) // 2
            if int(counts.sum()) > 8192:
                continue
            samples.append({"index": index, "split": split, "postpool_tokens": int(counts.sum()),
                            "item": clone_record(item)})
            count += 1
            if count == args.count:
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"manifest": dataset.manifest, "samples": samples}, args.output)
    print([(sample["split"], sample["index"], sample["postpool_tokens"]) for sample in samples], flush=True)


if __name__ == "__main__":
    main()
