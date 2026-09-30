import json
from itertools import pairwise

import torch

from molai.data import (
    FieldShardDataset,
    NoisyFieldDataset,
    ShardShuffleSampler,
    SpectrumFieldDataset,
    collate_spectrum_field_batch,
)


def test_field_dataset_reads_expected_charge_shards(tmp_path) -> None:
    payload = {
        "electrostatic_potential": torch.randn(2, 1, 8, 8).half(),
        "canonical_smiles": ["CO", "CCO"],
        "molecule_key": ["ABCDEFGHIJKLMN-ONE", "NOPQRSTUVWXYZ-TWO"],
        "electron_count": torch.tensor([14, 20]),
        "formal_charge": torch.tensor([0, 0]),
    }
    torch.save(payload, tmp_path / "fields-000000.pt")
    manifest = {
        "channel": "electrostatic_potential",
        "shards": [{"file": "fields-000000.pt", "records": 2}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    dataset = FieldShardDataset(tmp_path)

    assert len(dataset) == 2
    assert dataset[0]["field"].dtype == torch.float32
    assert dataset[0]["smiles"] == "CO"
    assert dataset[0]["molecule_key"] == "ABCDEFGHIJKLMN-ONE"
    assert dataset[0]["inchikey14"] == "ABCDEFGHIJKLMN"
    assert list(dataset.iter_smiles()) == ["CO", "CCO"]


def test_shard_shuffle_sampler_preserves_shard_locality(tmp_path) -> None:
    shards = []
    for shard_index, records in enumerate((3, 2, 4)):
        name = f"fields-{shard_index:06d}.pt"
        torch.save(
            {
                "field": torch.zeros(records, 1, 2, 2),
                "smiles": ["C"] * records,
                "inchikey14": [f"KEY{shard_index}"] * records,
                "electron_count": torch.ones(records),
                "formal_charge": torch.zeros(records),
            },
            tmp_path / name,
        )
        shards.append({"file": name, "records": records})
    (tmp_path / "manifest.json").write_text(
        json.dumps({"shards": shards}), encoding="utf-8"
    )
    dataset = FieldShardDataset(tmp_path)
    sampler = ShardShuffleSampler(dataset, seed=11)

    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)

    assert sorted(first) == list(range(9))
    assert first != second
    shard_ids = [0 if index < 3 else 1 if index < 5 else 2 for index in first]
    assert len([1 for left, right in pairwise(shard_ids) if left != right]) == 2


def test_noise_view_preserves_raw_fields_without_copying(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    torch.save(
        {
            "field": torch.tensor([[[[-4.0, 0.0], [4.0, 12.0]]]]),
            "smiles": ["CO"],
            "inchikey14": ["ABCDEFGHIJKLMN"],
            "electron_count": torch.tensor([14]),
            "formal_charge": torch.tensor([0]),
        },
        source / "fields-000000.pt",
    )
    (source / "manifest.json").write_text(
        json.dumps({"shards": [{"file": "fields-000000.pt", "records": 1}]}),
        encoding="utf-8",
    )
    view = tmp_path / "noise"
    view.mkdir()
    (view / "manifest.json").write_text(
        json.dumps(
            {
                "format": "molai-noise-view-v1",
                "source": "../source",
                "variants_per_field": 1,
                "seed": 7,
                "value_space": "raw",
                "scale_reference": {"type": "rms", "value": 4.0},
                "noise_schedule": {
                    "type": "geometric_ve",
                    "levels": 64,
                    "sigma_min": 0.04,
                    "sigma_max": 4.0,
                },
            }
        ),
        encoding="utf-8",
    )

    dataset = NoisyFieldDataset(view)
    item = dataset[0]

    torch.testing.assert_close(item["field"], torch.tensor([[[-4.0, 0.0], [4.0, 12.0]]]))
    assert "noise_level" not in item


def test_shard_sampler_pads_each_rank_to_full_batches(tmp_path) -> None:
    records = 17
    torch.save(
        {
            "field": torch.zeros(records, 1, 2, 2),
            "smiles": ["C"] * records,
            "inchikey14": ["ABCDEFGHIJKLMN"] * records,
            "electron_count": torch.ones(records),
            "formal_charge": torch.zeros(records),
        },
        tmp_path / "fields-000000.pt",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"shards": [{"file": "fields-000000.pt", "records": records}]}),
        encoding="utf-8",
    )
    dataset = FieldShardDataset(tmp_path)
    rank_zero = ShardShuffleSampler(dataset, replicas=2, rank=0, batch_size=4)
    rank_one = ShardShuffleSampler(dataset, replicas=2, rank=1, batch_size=4)

    combined = list(rank_zero) + list(rank_one)

    assert len(rank_zero) == len(rank_one) == 12
    assert set(combined) == set(range(records))
    assert len(combined) - len(set(combined)) == 7


def test_shard_sampler_molecule_split_is_disjoint(tmp_path) -> None:
    records = 100
    torch.save(
        {
            "field": torch.zeros(records, 1, 2, 2),
            "smiles": ["C"] * records,
            "inchikey14": ["ABCDEFGHIJKLMN"] * records,
            "electron_count": torch.ones(records),
            "formal_charge": torch.zeros(records),
        },
        tmp_path / "fields-000000.pt",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"shards": [{"file": "fields-000000.pt", "records": records}]}),
        encoding="utf-8",
    )
    dataset = FieldShardDataset(tmp_path)
    train = ShardShuffleSampler(
        dataset, split="train", validation_fraction=0.2, split_seed=23
    )
    validation = ShardShuffleSampler(
        dataset,
        split="validation",
        validation_fraction=0.2,
        split_seed=23,
        shuffle=False,
    )

    train_indices = set(train)
    validation_indices = set(validation)

    assert train_indices.isdisjoint(validation_indices)
    assert train_indices | validation_indices == set(range(records))
    assert 10 <= len(validation_indices) <= 30


def test_spectrum_field_dataset_decodes_and_collates(tmp_path) -> None:
    fields = tmp_path / "fields"
    fields.mkdir()
    torch.save(
        {
            "field": torch.zeros(2, 1, 4, 4),
            "smiles": ["CO", "CCO"],
            "inchikey14": ["ABCDEFGHIJKLMN", "NOPQRSTUVWXYZ"],
            "electron_count": torch.tensor([14, 20]),
            "formal_charge": torch.tensor([0, 0]),
        },
        fields / "fields-000000.pt",
    )
    (fields / "manifest.json").write_text(
        json.dumps({"shards": [{"file": "fields-000000.pt", "records": 2}]}),
        encoding="utf-8",
    )

    spectra = tmp_path / "spectra"
    spectra.mkdir()
    torch.save(
        {
            "mz": torch.tensor([100.0, 200.0, 150.0, 250.0]),
            "intensity": torch.tensor([0.5, 1.0, 0.8, 0.6], dtype=torch.float16),
            "peak_offsets": torch.tensor([0, 2, 3, 4]),
            "spectrum_offsets": torch.tensor([0, 1, 3]),
            "metadata": torch.zeros(3, 6, dtype=torch.float16),
            "precursor_mz": torch.tensor([300.0, 400.0, 450.0]),
        },
        spectra / "spectra-000000.pt",
    )
    (spectra / "manifest.json").write_text(
        json.dumps(
            {
                "format": "molai-spectrum-field-v1",
                "field_source": "../fields",
                "records": 2,
                "metadata_dim": 6,
                "shards": [{"file": "spectra-000000.pt", "records": 2}],
            }
        ),
        encoding="utf-8",
    )

    dataset = SpectrumFieldDataset(spectra)
    batch = collate_spectrum_field_batch([dataset[0], dataset[1]], peak_chunk_size=2)

    assert batch["peak_chunks"].shape == (3, 2, 2)
    assert batch["peak_mask"].sum() == 4
    assert batch["chunk_to_spectrum"].tolist() == [0, 1, 2]
    assert batch["spectrum_to_molecule"].tolist() == [0, 1, 1]
    assert batch["metadata"].shape == (3, 6)
    assert float(batch["peak_chunks"][0, 1, 1]) == 1.0
