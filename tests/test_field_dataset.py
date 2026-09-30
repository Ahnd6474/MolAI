import json
from itertools import pairwise

import torch

from molai.data import FieldShardDataset, NoisyFieldDataset, ShardShuffleSampler


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
