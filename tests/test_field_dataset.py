import json

import torch

from molai.data import FieldShardDataset


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
