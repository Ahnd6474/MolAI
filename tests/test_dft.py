import numpy as np
import pytest
import torch

from molai.dft import DFT2DConfig, OrbitalFreeDFT2D, canonical_nuclei, collate_nuclei


def test_canonical_layout_is_deterministic_and_counts_implicit_hydrogen() -> None:
    first = canonical_nuclei("CCO")
    second = canonical_nuclei("OCC")

    assert first.canonical_smiles == second.canonical_smiles
    np.testing.assert_array_equal(first.coordinates, second.coordinates)
    assert first.electron_count == pytest.approx(20.0)


@pytest.mark.parametrize(("smiles", "expected_charge"), [("O", 0.0), ("[NH4+]", 1.0)])
def test_solver_conserves_charge(smiles: str, expected_charge: float) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    molecule = canonical_nuclei(smiles)
    config = DFT2DConfig(resolution=32, steps=12, convergence_patience=3)
    result = OrbitalFreeDFT2D(config, device).solve(collate_nuclei([molecule], device))

    assert result.field.shape == (1, 1, 32, 32)
    assert torch.isfinite(result.field).all()
    assert float(result.field.abs().max()) <= 1.0
    assert float(result.integrated_charge[0]) == pytest.approx(expected_charge, abs=2e-4)
