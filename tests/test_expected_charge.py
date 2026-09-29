import numpy as np
import pytest
import torch

from molai.fields import ExpectedCharge2D, ExpectedChargeConfig, TrainingChannel

ATP4_SMILES = (
    "C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H](O3)"
    "COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])O)O)N"
)


@pytest.fixture(scope="module")
def renderer() -> ExpectedCharge2D:
    return ExpectedCharge2D(
        ExpectedChargeConfig(resolution=80, explicit_hydrogens=True), "cpu"
    )


@pytest.mark.parametrize(
    ("smiles", "formal_charge"),
    (
        ("O", 0),
        ("C[NH3+]", 1),
        ("CC(=O)[O-]", -1),
        ("c1ccccc1", 0),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 0),
        (ATP4_SMILES, -4),
    ),
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


@pytest.mark.parametrize(
    ("channel", "attribute"),
    (
        ("field", "field"),
        ("signed_charge", "signed_charge"),
        ("electrostatic_potential", "electrostatic_potential"),
    ),
)
def test_training_fast_path_matches_diagnostic_channels(
    renderer: ExpectedCharge2D, channel: TrainingChannel, attribute: str
) -> None:
    smiles = ["CCO", "c1ccccc1", "CC(=O)[O-]"]
    diagnostic = renderer.render_batch(smiles)
    training = renderer.render_training_batch(smiles, channel)

    torch.testing.assert_close(training.channel, getattr(diagnostic, attribute))
    assert training.canonical_smiles == diagnostic.canonical_smiles
    assert len(training.molecule_keys) == len(smiles)


def test_single_atom_molecule_supports_empty_primitive_groups(
    renderer: ExpectedCharge2D,
) -> None:
    result = renderer.render_training_batch(["[He]"], "signed_charge")

    assert result.channel.shape == (1, 1, 80, 80)
    assert torch.isfinite(result.channel).all()


def test_molecule_key_preserves_charge_state(renderer: ExpectedCharge2D) -> None:
    neutral = renderer.compile_molecule("CC(=O)O")
    anion = renderer.compile_molecule("CC(=O)[O-]")

    assert neutral.molecule_key != anion.molecule_key


def test_electrostatic_potential_matches_direct_g_times_q() -> None:
    renderer = ExpectedCharge2D(ExpectedChargeConfig(resolution=12), "cpu")
    charge = torch.zeros(1, 12, 12)
    source_y, source_x = 3, 7
    charge[0, source_y, source_x] = 1.0

    potential = renderer._electrostatic_potential(charge)[0]
    indices = torch.arange(12, dtype=torch.float32)
    offset_y, offset_x = torch.meshgrid(
        indices - source_y, indices - source_x, indexing="ij"
    )
    radius_squared = (
        offset_x.square() + offset_y.square()
    ) * renderer.grid_spacing**2
    expected = torch.rsqrt(
        radius_squared + renderer.config.coulomb_softening**2
    ) * renderer.pixel_area
    expected = expected - expected.mean()

    torch.testing.assert_close(potential, expected, atol=2e-6, rtol=2e-6)


def test_crossing_optimizer_preserves_crossing_free_layout() -> None:
    original = ExpectedCharge2D(
        ExpectedChargeConfig(resolution=32, optimize_layout=False), "cpu"
    ).compile_molecule("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    optimized = ExpectedCharge2D(
        ExpectedChargeConfig(resolution=32, optimize_layout=True), "cpu"
    ).compile_molecule("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

    assert original.bond_crossings == optimized.bond_crossings == 0
    np.testing.assert_array_equal(original.coordinates, optimized.coordinates)


def test_crossing_optimizer_removes_atp_crossing() -> None:
    original = ExpectedCharge2D(
        ExpectedChargeConfig(resolution=32, optimize_layout=False), "cpu"
    ).compile_molecule(ATP4_SMILES)
    optimized = ExpectedCharge2D(
        ExpectedChargeConfig(resolution=32, optimize_layout=True), "cpu"
    ).compile_molecule(ATP4_SMILES)

    assert original.bond_crossings == 1
    assert optimized.bond_crossings == 0
