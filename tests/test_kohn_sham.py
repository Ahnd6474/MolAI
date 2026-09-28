import pytest
import torch

from molai.dft import KohnSham2D, KohnSham2DConfig, canonical_nuclei, collate_nuclei


def test_kohn_sham_orbitals_are_orthonormal_and_conserve_charge() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    molecule = canonical_nuclei("CCO")
    config = KohnSham2DConfig(resolution=32, scf_iterations=3, orbital_steps=2)
    result = KohnSham2D(config, device).solve(collate_nuclei([molecule], device))

    orbitals = result.orbitals[0].flatten(1)
    spacing = 2.0 * config.extent / (config.resolution - 1)
    overlap = orbitals @ orbitals.T * spacing**2
    torch.testing.assert_close(
        overlap, torch.eye(len(orbitals), device=device), atol=2e-4, rtol=2e-4
    )
    assert float(result.integrated_charge[0]) == pytest.approx(0.0, abs=2e-4)
    assert result.field.shape == (1, 1, 32, 32)
    assert float(result.field.abs().max()) <= 1.0
    assert torch.isfinite(result.electron_localization).all()
    assert torch.isfinite(result.bond_order_density).all()
    nuclear_charge = result.core_density.sum() * spacing**2
    electron_charge = result.electron_density.sum() * spacing**2
    assert float(nuclear_charge) == pytest.approx(float(molecule.effective_charges.sum()), abs=2e-4)
    assert float(electron_charge) == pytest.approx(molecule.electron_count, abs=2e-4)
