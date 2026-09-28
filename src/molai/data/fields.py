"""Random-access loader for generated pseudo-DFT field shards."""

from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

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
        return {
            "field": payload["field"][local_index].float(),
            "smiles": payload["smiles"][local_index],
            "inchikey14": payload["inchikey14"][local_index],
            "electron_count": payload["electron_count"][local_index],
            "formal_charge": payload["formal_charge"][local_index],
        }

    def iter_smiles(self) -> Iterator[str]:
        for shard_index in range(len(self.shards)):
            yield from self._load_shard(shard_index)["smiles"]


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

