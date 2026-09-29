import pytest
import torch

from molai.fields import ExpectedCharge2D, ExpectedChargeConfig


@pytest.fixture(scope="module")
def renderer() -> ExpectedCharge2D:
    return ExpectedCharge2D(
        ExpectedChargeConfig(resolution=80, explicit_hydrogens=True), "cpu"
    )


@pytest.mark.parametrize(
    ("smiles", "formal_charge"),
    (("O", 0), ("C[NH3+]", 1), ("CC(=O)[O-]", -1), ("c1ccccc1", 0)),
)
def test_expected_charge_conserves_electrons_and_formal_charge(
    renderer: ExpectedCharge2D, smiles: str, formal_charge: int
) -> None:
    result = renderer.render(smiles)

    assert result.integrated_electron_count == pytest.approx(
        result.expected_electron_count, abs=2e-4
    )
    assert result.integrated_charge == pytest.approx(formal_charge, abs=2e-4)
    assert result.integrated_nuclear_charge - result.integrated_electron_count == pytest.approx(
        formal_charge, abs=2e-4
    )
    assert torch.isfinite(result.field).all()
    assert float(result.field.abs().max()) <= 1.0


def test_components_partition_the_electron_budget(renderer: ExpectedCharge2D) -> None:
    result = renderer.render("CC(=O)Oc1ccccc1C(=O)O")
    area = renderer.pixel_area
    component_electrons = float(
        (
            result.bond_density.sum()
            + result.lone_pair_density.sum()
            + result.delocalized_density.sum()
        )
        * area
    )

    assert component_electrons == pytest.approx(result.expected_electron_count, abs=2e-4)
    assert float(result.lone_pair_density.max()) > 0.0
    assert float(result.delocalized_density.max()) > 0.0


def test_aromatic_pi_electrons_are_redistributed_not_added(
    renderer: ExpectedCharge2D,
) -> None:
    benzene = renderer.render("c1ccccc1")
    delocalized_electrons = float(benzene.delocalized_density.sum() * renderer.pixel_area)

    assert delocalized_electrons == pytest.approx(6.0, abs=2e-4)
    assert benzene.integrated_electron_count == pytest.approx(
        benzene.expected_electron_count, abs=2e-4
    )


def test_bond_density_polarizes_without_changing_electron_count() -> None:
    renderer = ExpectedCharge2D(
        ExpectedChargeConfig(resolution=96, explicit_hydrogens=False), "cpu"
    )
    result = renderer.render("CO")
    weights = result.bond_density[0]
    center = (renderer.grid * weights[..., None]).sum(dim=(0, 1)) / weights.sum()
    carbon = result.coordinates[result.atomic_numbers == 6][0]
    oxygen = result.coordinates[result.atomic_numbers == 8][0]

    assert float((center - oxygen).norm()) < float((center - carbon).norm())
    assert float(weights.sum() * renderer.pixel_area) == pytest.approx(2.0, abs=2e-4)


def test_batched_render_matches_single_render(renderer: ExpectedCharge2D) -> None:
    smiles = ["CCO", "c1ccccc1", "CC(=O)[O-]"]
    batch = renderer.render_batch(smiles)

    assert batch.field.shape == (3, 1, 80, 80)
    for index, value in enumerate(smiles):
        single = renderer.render(value)
        torch.testing.assert_close(batch.field[index], single.field)
        torch.testing.assert_close(batch.signed_charge[index], single.signed_charge)
