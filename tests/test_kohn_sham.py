import pytest
import torch

from molai.dft import KohnSham2D, KohnSham2DConfig, canonical_nuclei, collate_nuclei


def test_kohn_sham_orbitals_are_orthonormal_and_conserve_charge() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    molecule = canonical_nuclei("CCO")
    config = KohnSham2DConfig(
        resolution=32, scf_iterations=3, reference_scf_iterations=3, orbital_steps=2
    )
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
    promolecule_charge = result.promolecule_density.sum() * spacing**2
    fused_electron_charge = result.fused_electron_density.sum() * spacing**2
    deformation_charge = result.deformation_density.sum() * spacing**2
    fused_charge = result.fused_signed_density.sum() * spacing**2
    assert float(nuclear_charge) == pytest.approx(float(molecule.effective_charges.sum()), abs=2e-4)
    assert float(electron_charge) == pytest.approx(molecule.electron_count, abs=2e-4)
    assert float(promolecule_charge) == pytest.approx(molecule.electron_count, abs=2e-4)
    assert float(fused_electron_charge) == pytest.approx(molecule.electron_count, abs=2e-4)
    assert float(deformation_charge) == pytest.approx(0.0, abs=2e-4)
    assert float(fused_charge) == pytest.approx(0.0, abs=2e-4)


def test_isolated_atom_promolecule_preserves_each_population() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    molecule = canonical_nuclei("CO")
    config = KohnSham2DConfig(
        resolution=32, scf_iterations=1, reference_scf_iterations=1, orbital_steps=1
    )
    solver = KohnSham2D(config, device)
    batch = collate_nuclei([molecule], device)

    promolecule, atom_densities, reference_state = solver._promolecule_density(batch)
    total = promolecule.sum() * solver.pixel_area
    atom_populations = atom_densities.sum(dim=(-2, -1)) * solver.pixel_area

    assert float(total) == pytest.approx(molecule.electron_count, abs=2e-4)
    assert torch.isfinite(atom_densities).all()
    assert reference_state.iterations >= 1
    torch.testing.assert_close(
        atom_populations[0], batch.effective_charges[0], atol=2e-4, rtol=2e-4
    )
    maximum = torch.nonzero(promolecule[0] == promolecule[0].max(), as_tuple=False)[0]
    peak_xy = solver.backend.grid[maximum[0], maximum[1]]
    distances = torch.linalg.vector_norm(batch.coordinates[0] - peak_xy, dim=-1)
    assert float(distances.min()) <= 1.5 * solver.spacing


def test_invalid_feature_weights_are_rejected() -> None:
    config = KohnSham2DConfig(
        resolution=16,
        deformation_weight=0.5,
        localization_weight=0.4,
        bond_order_weight=0.2,
    )
    with pytest.raises(ValueError, match="feature weights"):
        KohnSham2D(config, "cpu")
