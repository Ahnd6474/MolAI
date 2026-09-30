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


class NoisyFieldDataset(Dataset[dict[str, object]]):
    """A zero-copy view that assigns geometric noise levels to clean field records."""

    def __init__(self, root: Path | str, cache_shards: int = 2) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format") != "molai-noise-view-v1":
            raise ValueError("not a molai noise-view dataset")
        source_path = Path(str(self.manifest["source"])).expanduser()
        if not source_path.is_absolute():
            source_path = (self.root / source_path).resolve()
        self.source = FieldShardDataset(source_path, cache_shards=cache_shards)
        self.variants = int(self.manifest.get("variants_per_field", 1))
        self.levels = int(self.manifest["noise_schedule"]["levels"])
        if self.variants < 1 or self.levels < 2:
            raise ValueError("noise variants must be positive and levels must be at least two")
        self.shards = [
            {**shard, "records": int(shard["records"]) * self.variants}
            for shard in self.source.shards
        ]
        self.ends: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["records"])
            self.ends.append(total)

    def __len__(self) -> int:
        return len(self.source) * self.variants

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_index, _ = divmod(index, self.variants)
        return dict(self.source[source_index])

    def iter_smiles(self) -> Iterator[str]:
        return self.source.iter_smiles()


class SpectrumFieldDataset(Dataset[dict[str, object]]):
    """Aligned raw fields and compact, prepacked replicate MS/MS spectra."""

    def __init__(self, root: Path | str, cache_shards: int = 2) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format") != "molai-spectrum-field-v1":
            raise ValueError("not a molai spectrum-field dataset")
        field_path = Path(str(self.manifest["field_source"])).expanduser()
        if not field_path.is_absolute():
            field_path = (self.root / field_path).resolve()
        self.fields = open_field_dataset(field_path, cache_shards=cache_shards)
        self.shards = self.manifest["shards"]
        self.ends: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["records"])
            self.ends.append(total)
        if total != len(self.fields):
            raise ValueError("spectrum and field dataset sizes differ")
        self.cache_shards = cache_shards
        self.cache: OrderedDict[int, dict[str, object]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.fields)

    def _load_shard(self, shard_index: int) -> dict[str, object]:
        if shard_index in self.cache:
            payload = self.cache.pop(shard_index)
            self.cache[shard_index] = payload
            return payload
        payload = torch.load(
            self.root / self.shards[shard_index]["file"],
            map_location="cpu",
            weights_only=False,
        )
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
        item = dict(self.fields[index])
        spectrum_start = int(payload["spectrum_offsets"][local_index])
        spectrum_end = int(payload["spectrum_offsets"][local_index + 1])
        peak_offsets = payload["peak_offsets"][spectrum_start : spectrum_end + 1].long()
        peak_start = int(peak_offsets[0])
        peak_end = int(peak_offsets[-1])
        peak_offsets = peak_offsets - peak_start
        peaks = torch.stack(
            (
                payload["mz"][peak_start:peak_end].float(),
                payload["intensity"][peak_start:peak_end].float(),
            ),
            dim=-1,
        )
        item.update(
            {
                "peaks": peaks,
                "peak_offsets": peak_offsets,
                "metadata": payload["metadata"][spectrum_start:spectrum_end].float(),
                "precursor_mz": payload["precursor_mz"][spectrum_start:spectrum_end].float(),
            }
        )
        return item

    def iter_smiles(self) -> Iterator[str]:
        return self.fields.iter_smiles()


def open_field_dataset(
    root: Path | str, cache_shards: int = 2
) -> FieldShardDataset | NoisyFieldDataset | SpectrumFieldDataset:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") == "molai-spectrum-field-v1":
        return SpectrumFieldDataset(root, cache_shards=cache_shards)
    if manifest.get("format") == "molai-noise-view-v1":
        return NoisyFieldDataset(root, cache_shards=cache_shards)
    return FieldShardDataset(root, cache_shards=cache_shards)


class ShardShuffleSampler(Sampler[int]):
    """Shuffle records while visiting each large tensor shard only once per epoch."""

    def __init__(
        self,
        dataset: FieldShardDataset | NoisyFieldDataset | SpectrumFieldDataset,
        *,
        seed: int = 0,
        rank: int = 0,
        replicas: int = 1,
        batch_size: int | None = None,
        split: str = "all",
        validation_fraction: float = 0.0,
        split_seed: int = 17,
        shuffle: bool = True,
    ) -> None:
        if replicas < 1 or not 0 <= rank < replicas:
            raise ValueError("rank must be in [0, replicas)")
        if split not in {"all", "train", "validation"}:
            raise ValueError("split must be all, train, or validation")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if split != "all" and validation_fraction <= 0.0:
            raise ValueError("train/validation splits require a positive validation fraction")
        self.dataset = dataset
        self.seed = seed
        self.rank = rank
        self.replicas = replicas
        self.shuffle = shuffle
        self.epoch = 0
        self.local_indices: list[list[int]] = []
        start = 0
        selected_count = 0
        validation_threshold = int(validation_fraction * (1 << 64))
        for end in dataset.ends:
            selected: list[int] = []
            for index in range(start, end):
                mixed = (
                    (index + 1) * 0x9E3779B185EBCA87 + split_seed
                ) & ((1 << 64) - 1)
                is_validation = mixed < validation_threshold
                if split == "all" or (split == "validation") == is_validation:
                    selected.append(index)
            self.local_indices.append(selected)
            selected_count += len(selected)
            start = end
        self.selected_count = selected_count
        if selected_count < 1:
            raise ValueError(f"{split} split is empty")
        self.samples_per_rank = (selected_count + replicas - 1) // replicas
        if batch_size is not None:
            if batch_size < 1:
                raise ValueError("batch_size must be positive")
            self.samples_per_rank = (
                (self.samples_per_rank + batch_size - 1) // batch_size * batch_size
            )

    def __len__(self) -> int:
        return self.samples_per_rank

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.shuffle:
            shard_order = torch.randperm(len(self.dataset.shards), generator=generator).tolist()
        else:
            shard_order = list(range(len(self.dataset.shards)))
        indices: list[int] = []
        for shard_index in shard_order:
            shard_indices = self.local_indices[shard_index]
            if self.shuffle:
                local_order = torch.randperm(len(shard_indices), generator=generator).tolist()
                indices.extend(shard_indices[local_index] for local_index in local_order)
            else:
                indices.extend(shard_indices)

        total_size = self.samples_per_rank * self.replicas
        if len(indices) < total_size:
            missing = total_size - len(indices)
            repeats = (missing + len(indices) - 1) // len(indices)
            indices.extend((indices * repeats)[:missing])
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
    output: dict[str, Tensor | list[str]] = {
        "field": torch.stack([item["field"] for item in batch]),
        "token_ids": token_ids,
        "smiles": [str(item["smiles"]) for item in batch],
        "inchikey14": [str(item["inchikey14"]) for item in batch],
    }
    if "noise_level" in batch[0]:
        output["noise_level"] = torch.tensor(
            [int(item["noise_level"]) for item in batch], dtype=torch.long
        )
    return output


def collate_spectrum_field_batch(
    batch: list[dict[str, object]],
    peak_chunk_size: int = 256,
) -> dict[str, Tensor | list[str]]:
    """Collate every spectrum and split every peak sequence into bounded chunks."""

    if peak_chunk_size < 1:
        raise ValueError("peak_chunk_size must be positive")
    chunks: list[Tensor] = []
    chunk_masks: list[Tensor] = []
    chunk_to_spectrum: list[int] = []
    spectrum_to_molecule: list[int] = []
    metadata_parts: list[Tensor] = []
    precursor_parts: list[Tensor] = []
    spectrum_index = 0
    for molecule_index, item in enumerate(batch):
        offsets = item["peak_offsets"]
        spectra = len(offsets) - 1
        if spectra < 1:
            raise ValueError("every field must have at least one spectrum")
        metadata_parts.append(item["metadata"])
        precursor_parts.append(item["precursor_mz"])
        spectrum_to_molecule.extend([molecule_index] * spectra)
        for local_spectrum in range(spectra):
            start = int(offsets[local_spectrum])
            end = int(offsets[local_spectrum + 1])
            values = item["peaks"][start:end]
            order = values[:, 0].argsort()
            values = values[order]
            for chunk_start in range(0, len(values), peak_chunk_size):
                chunk_values = values[chunk_start : chunk_start + peak_chunk_size]
                chunk = torch.zeros(peak_chunk_size, 2)
                mask = torch.zeros(peak_chunk_size, dtype=torch.bool)
                chunk[: len(chunk_values)] = chunk_values
                mask[: len(chunk_values)] = True
                chunks.append(chunk)
                chunk_masks.append(mask)
                chunk_to_spectrum.append(spectrum_index)
            spectrum_index += 1

    return {
        "field": torch.stack([item["field"] for item in batch]),
        "peak_chunks": torch.stack(chunks),
        "peak_mask": torch.stack(chunk_masks),
        "chunk_to_spectrum": torch.tensor(chunk_to_spectrum, dtype=torch.long),
        "spectrum_to_molecule": torch.tensor(spectrum_to_molecule, dtype=torch.long),
        "metadata": torch.cat(metadata_parts),
        "precursor_mz": torch.cat(precursor_parts),
        "smiles": [str(item["smiles"]) for item in batch],
        "inchikey14": [str(item["inchikey14"]) for item in batch],
    }
