"""Random-access loader for generated pseudo-DFT field shards."""

from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from molai.models.smiles import SmilesTokenizer


class FieldShardDataset(Dataset[dict[str, object]]):
    def __init__(self, root: Path | str, cache_shards: int = 2) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.shards = self.manifest["shards"]
        self.ends: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["records"])
            self.ends.append(total)
        self.cache_shards = cache_shards
        self.cache: OrderedDict[int, dict[str, object]] = OrderedDict()
        self.field_key = str(self.manifest.get("channel", "field"))

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def _load_shard(self, shard_index: int) -> dict[str, object]:
        if shard_index in self.cache:
            payload = self.cache.pop(shard_index)
            self.cache[shard_index] = payload
            return payload
        path = self.root / self.shards[shard_index]["file"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.cache[shard_index] = payload
        while len(self.cache) > self.cache_shards:
            self.cache.popitem(last=False)
        return payload

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.ends, index)
        start = 0 if shard_index == 0 else self.ends[shard_index - 1]
        local_index = index - start
        payload = self._load_shard(shard_index)
        field_key = self.field_key if self.field_key in payload else "field"
        smiles_key = "canonical_smiles" if "canonical_smiles" in payload else "smiles"
        key_name = "molecule_key" if "molecule_key" in payload else "inchikey14"
        molecule_key = str(payload[key_name][local_index])
        return {
            "field": payload[field_key][local_index].float(),
            "smiles": payload[smiles_key][local_index],
            "molecule_key": molecule_key,
            "inchikey14": molecule_key[:14],
            "electron_count": payload["electron_count"][local_index],
            "formal_charge": payload["formal_charge"][local_index],
        }

    def iter_smiles(self) -> Iterator[str]:
        for shard_index in range(len(self.shards)):
            payload = self._load_shard(shard_index)
            key = "canonical_smiles" if "canonical_smiles" in payload else "smiles"
            yield from payload[key]


class ShardShuffleSampler(Sampler[int]):
    """Shuffle records while visiting each large tensor shard only once per epoch."""

    def __init__(
        self,
        dataset: FieldShardDataset,
        *,
        seed: int = 0,
        rank: int = 0,
        replicas: int = 1,
    ) -> None:
        if replicas < 1 or not 0 <= rank < replicas:
            raise ValueError("rank must be in [0, replicas)")
        self.dataset = dataset
        self.seed = seed
        self.rank = rank
        self.replicas = replicas
        self.epoch = 0
        self.samples_per_rank = (len(dataset) + replicas - 1) // replicas

    def __len__(self) -> int:
        return self.samples_per_rank

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        shard_order = torch.randperm(len(self.dataset.shards), generator=generator).tolist()
        indices: list[int] = []
        start = 0
        shard_starts: list[int] = []
        for end in self.dataset.ends:
            shard_starts.append(start)
            start = end
        for shard_index in shard_order:
            shard_start = shard_starts[shard_index]
            count = self.dataset.ends[shard_index] - shard_start
            local_order = torch.randperm(count, generator=generator).tolist()
            indices.extend(shard_start + local_index for local_index in local_order)

        total_size = self.samples_per_rank * self.replicas
        if len(indices) < total_size:
            indices.extend(indices[: total_size - len(indices)])
        rank_start = self.rank * self.samples_per_rank
        return iter(indices[rank_start : rank_start + self.samples_per_rank])


def collate_field_batch(
    batch: list[dict[str, object]],
    tokenizer: SmilesTokenizer,
) -> dict[str, Tensor | list[str]]:
    encoded = [tokenizer.encode(str(item["smiles"])) for item in batch]
    max_length = max(map(len, encoded))
    token_ids = torch.full((len(batch), max_length), tokenizer.pad_id, dtype=torch.long)
    for index, values in enumerate(encoded):
        token_ids[index, : len(values)] = torch.tensor(values)
    return {
        "field": torch.stack([item["field"] for item in batch]),
        "token_ids": token_ids,
        "smiles": [str(item["smiles"]) for item in batch],
        "inchikey14": [str(item["inchikey14"]) for item in batch],
    }
