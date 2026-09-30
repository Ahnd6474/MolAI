import json
from itertools import pairwise

import torch

from molai.data import FieldShardDataset, ShardShuffleSampler


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
