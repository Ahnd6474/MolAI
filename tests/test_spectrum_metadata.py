import importlib
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from molai.data.spectrum_metadata import SpectrumMetadataSchema, adduct_mass_features
from molai.models.condition import SpectrumTokenEncoder, bind_spectrum_metadata


@pytest.fixture
def schema(tmp_path):
    rows = [
        {"ionization_mode": "positive", "adduct": "[M+H]+", "instrument_type": "timsTOF",
         "collision_energy_orig_units": "eV", "collision_energy_ev": [20., 40., 60.],
         "collision_energy_orig": "20,40,60"},
        {"ionization_mode": "negative", "adduct": "[M-H]-", "instrument_type": "Orbitrap",
         "collision_energy_orig_units": "NCE", "collision_energy_ev": [25.7],
         "collision_energy_orig": "30% (nominal)"},
    ]
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return SpectrumMetadataSchema.fit_training_parquet(path)


def test_metadata_roundtrip_missing_unseen_and_no_label_features(schema):
    restored = SpectrumMetadataSchema.from_dict(schema.to_dict())
    assert restored.feature_names == schema.feature_names
    row = {"instrument_type": "new instrument", "precursor_mz": 375.0709,
           "base_peak_intensity": 0.0}
    value = restored.encode(row, 10)
    assert value.dtype == torch.float32
    assert value.shape == (len(schema.feature_names),)
    assert value[schema.feature_names.index("instrument_type:unseen")] == 1
    assert value[schema.feature_names.index("adduct:missing")] == 1
    assert value[schema.feature_names.index("base_peak_intensity_present")] == 1
    assert restored.encode({}, 0)[schema.feature_names.index("base_peak_intensity_present")] == 0
    for key in ("precursor_error_ppm", "molecular_formula", "normalized_smiles", "ingest_lib",
                "inchikey", "inchikey14", "molecule_id", "spectrum_id", "adduct_orig"):
        changed = restored.encode({**row, key: "should never be a condition"}, 10)
        torch.testing.assert_close(value, changed)


def test_collision_energy_values_not_just_mean_and_variance(schema):
    # These different lists have identical population mean and variance.
    first = [30. - math.sqrt(75.), 30. - math.sqrt(75.), 30. + 2 * math.sqrt(75.)]
    second = [15., 30., 45.]
    assert sum(first) == pytest.approx(sum(second))
    assert sum((x - 30.) ** 2 for x in first) == pytest.approx(sum((x - 30.) ** 2 for x in second))
    assert not torch.equal(schema.encode({"collision_energy_ev": first}, 3),
                           schema.encode({"collision_energy_ev": second}, 3))
    torch.testing.assert_close(schema.encode({"collision_energy_ev": first}, 3),
                               schema.encode({"collision_energy_ev": first[::-1]}, 3))
    zero = schema.encode({"collision_energy_ev": [0.]}, 3)
    missing = schema.encode({}, 3)
    assert not torch.equal(zero, missing)
    with pytest.raises(ValueError, match="refusing truncation"):
        schema.encode({"collision_energy_ev": [1., 2., 3., 4.]}, 4)


def test_original_energy_units_and_values_preserved(schema):
    value = schema.encode({"collision_energy_orig_units": "NCE",
                           "collision_energy_orig": "30% (nominal)"}, 2)
    assert value[schema.feature_names.index("collision_orig_0_over_100")] == pytest.approx(.3)
    assert value[schema.feature_names.index("collision_energy_orig_units:nce")] == 1
    assert torch.isfinite(schema.encode({"precursor_mz": float("nan"),
                                        "base_peak_intensity": float("inf"),
                                        "collision_energy_ev": [None, float("nan")]}, 0)).all()


@pytest.mark.parametrize("adduct,n,z,shift", [
    ("[M+H]+", 1, 1, 1.007825032), ("[M-H]-", 1, -1, -1.007825032),
    ("[2M+Na]+", 2, 1, 22.98976928), ("[M+2H]2+", 1, 2, 2.015650064),
    ("[M+CH2O2-H]-", 1, -1, 44.997654272),
])
def test_adduct_derived_neutral_mass(adduct, n, z, shift):
    from molai.data.spectrum_metadata import ELECTRON_MASS
    precursor = (300. * n + shift - z * ELECTRON_MASS) / abs(z)
    features = adduct_mass_features(adduct, precursor)
    assert features[0] == z
    assert features[1] == n
    assert features[3] * 1000 == pytest.approx(300., abs=1e-6)
    assert features[4] == 1
    assert adduct_mass_features("[Cat]2+", precursor) == [0.] * 5


def test_rich_metadata_reaches_condition_tokens_and_receives_gradient(schema):
    torch.manual_seed(7)
    encoder = SpectrumTokenEncoder(
        metadata_dim=len(schema.feature_names), dim=16, heads=4, dropout=0.,
        peak_position_dim=4, mz_bin_width=1., mz_upper_bound=1000.,
        metadata_fusion="preconv_cross", metadata_feature_names=schema.feature_names,
    )
    peaks = torch.tensor([[[100., 1.], [150., .5], [200., .2], [250., .1]]])
    row = {"adduct": "[M+H]+", "instrument_type": "timsTOF", "precursor_mz": 300.}
    first = schema.encode(row, 4)
    second = schema.encode({**row, "adduct": "[M-H]-"}, 4)

    def run(metadata):
        return encoder.forward_ragged_tokens(
            peaks, torch.ones(1, 4, dtype=torch.bool), torch.tensor([0]), torch.tensor([0]),
            metadata[None], torch.tensor([300.]), 1,
        )[0]

    tokens = run(first)
    assert not torch.allclose(tokens, run(second))
    tokens.square().sum().backward()
    gradient = encoder.metadata_tokenizer.projections[1][0].weight.grad
    adduct_column = schema.feature_names.index("adduct:[m+h]+")
    local_column = encoder.metadata_tokenizer.indices_1.tolist().index(adduct_column)
    assert gradient[:, local_column].abs().sum() > 0
    assert torch.isfinite(encoder.metadata_cross.gate.grad)


def test_packer_dataset_collator_encoder_use_same_metadata_schema(tmp_path, monkeypatch):
    from molai.data import SpectrumFieldDataset, collate_spectrum_field_batch

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    packer = importlib.import_module("prepare_spectrum_field_dataset")
    fields, output = tmp_path / "fields", tmp_path / "packed"
    fields.mkdir()
    (fields / "manifest.json").write_text(json.dumps({
        "shards": [{"file": "fields-000000.pt", "records": 1}],
    }))
    (fields / "molecule_keys.txt").write_text("A\n")
    torch.save({"field": torch.zeros(1, 1, 4, 4), "canonical_smiles": ["C"],
                "molecule_key": ["A"], "electron_count": torch.tensor([10.]),
                "formal_charge": torch.tensor([0.])}, fields / "fields-000000.pt")
    spectra = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"inchikey": "A", "ms2_mzs": [100., 120., 150., 175.],
         "ms2_normalized_intensities": [1., .3, .2, .1], "precursor_mz": 200.,
         "ionization_mode": "positive", "adduct": "[M+H]+", "instrument_type": "timsTOF",
         "collision_energy_orig_units": "eV", "base_peak_intensity": 120000.,
         "collision_energy_ev": [20., 40., 60.], "collision_energy_orig": "20,40,60",
         "precursor_error_ppm": 123456.},
    ]), spectra)
    monkeypatch.setattr("sys.argv", ["prepare", "--fields", str(fields), "--spectra",
                                     str(spectra), "--output", str(output)])
    packer.main()
    dataset = SpectrumFieldDataset(output)
    manifest = dataset.manifest
    assert "precursor_error_ppm" in manifest["excluded_condition_columns"]
    assert "precursor_error_ppm" not in " ".join(manifest["metadata_features"])
    assert dataset[0]["metadata"].dtype == torch.float32
    batch = collate_spectrum_field_batch([dataset[0]], peak_chunk_size=8)
    encoder = SpectrumTokenEncoder(metadata_dim=manifest["metadata_dim"], dim=16, heads=4,
                                   dropout=0., peak_position_dim=4, mz_bin_width=1.,
                                   mz_upper_bound=1000.,
                                   **bind_spectrum_metadata({"metadata_fusion": "preconv_cross"},
                                                            manifest))
    tokens, mask = encoder.forward_ragged_tokens(
        *(batch[name] for name in ("peak_chunks", "peak_mask", "chunk_to_spectrum",
                                  "spectrum_to_molecule", "metadata", "precursor_mz")), 1,
    )
    assert tokens.shape[-1] == 16 and mask.any()
    tokens.square().sum().backward()
    assert encoder.metadata_tokenizer.projections[1][0].weight.grad is not None
    validator = importlib.import_module("validate_spectrum_repack")
    report = validator.validate(output, output, spectra)
    assert report["verified"] and report["sampled_molecules_matching_raw"] == 1
    packed_path = output / "spectra-000000.pt"
    payload = torch.load(packed_path, weights_only=False)
    payload["metadata"][0, 0] += 1.
    torch.save(payload, packed_path)
    with pytest.raises(ValueError, match="differ from raw Parquet"):
        validator.validate(output, output, spectra)
