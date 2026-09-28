import torch

from molai.fields import ElectronCloud2D, ElectronCloudConfig


def test_electron_cloud_is_deterministic_bounded_and_bond_sensitive() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ElectronCloudConfig(resolution=64, explicit_hydrogens=False)
    renderer = ElectronCloud2D(config, device)
    ethane = renderer.render("CC")
    ethane_reversed = renderer.render("CC")
    ethene = renderer.render("C=C")

    torch.testing.assert_close(ethane.field, ethane_reversed.field)
    assert ethane.field.shape == (1, 64, 64)
    assert torch.isfinite(ethane.field).all()
    assert float(ethane.field.abs().max()) <= 1.0
    assert float(ethene.bond_cloud.max()) > float(ethane.bond_cloud.max())
    water = ElectronCloud2D(
        ElectronCloudConfig(resolution=64, explicit_hydrogens=True), device
    ).render("O")
    assert water.atom_count == 3
    assert water.hydrogen_count == 2


def test_bond_cloud_polarizes_toward_more_electronegative_atom() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    renderer = ElectronCloud2D(ElectronCloudConfig(resolution=96, explicit_hydrogens=False), device)
    result = renderer.render("CO")
    weights = result.bond_cloud[0]
    center = (renderer.grid * weights[..., None]).sum(dim=(0, 1)) / weights.sum()
    carbon = result.coordinates[result.atomic_numbers == 6][0]
    oxygen = result.coordinates[result.atomic_numbers == 8][0]
    assert float((center - oxygen).norm()) < float((center - carbon).norm())
    assert float(result.partial_charges[result.atomic_numbers == 8][0]) < 0.0


def test_delocalization_and_stereochemistry_are_encoded() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    renderer = ElectronCloud2D(ElectronCloudConfig(resolution=80), device)

    benzene = renderer.render("c1ccccc1")
    cyclohexane = renderer.render("C1CCCCC1")
    assert float(benzene.delocalized_cloud.max()) > 0.0
    assert float(cyclohexane.delocalized_cloud.max()) == 0.0

    clockwise = renderer.render("F[C@](Cl)(Br)I")
    counterclockwise = renderer.render("F[C@@](Cl)(Br)I")
    assert not torch.allclose(clockwise.field, counterclockwise.field)
    assert not torch.equal(
        ((clockwise.field + 1.0) * 127.5).round().to(torch.uint8),
        ((counterclockwise.field + 1.0) * 127.5).round().to(torch.uint8),
    )
    assert float(clockwise.stereo_field.abs().max()) > 0.0

    trans = renderer.render("F/C=C/F")
    cis = renderer.render("F/C=C\\F")
    assert not torch.allclose(trans.field, cis.field)


def test_batched_render_matches_single_render() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    renderer = ElectronCloud2D(
        ElectronCloudConfig(resolution=64, primitive_chunk_size=7), device
    )
    smiles = ["CCO", "c1ccccc1", "F[C@](Cl)(Br)I"]
    batched = renderer.render_batch(smiles)

    assert batched.field.shape == (3, 1, 64, 64)
    for index, value in enumerate(smiles):
        single = renderer.render(value)
        torch.testing.assert_close(batched.field[index], single.field)
        assert batched.item(index).canonical_smiles == single.canonical_smiles
