"""Canonical RDKit layouts and valence-only pseudo-nuclei."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.MolStandardize import rdMolStandardize
from torch import Tensor


@dataclass(frozen=True, slots=True)
class MoleculeNuclei:
    canonical_smiles: str
    inchikey14: str
    coordinates: np.ndarray
    effective_charges: np.ndarray
    softening_widths: np.ndarray
    electron_count: float
    formal_charge: int


@dataclass(frozen=True, slots=True)
class NuclearBatch:
    coordinates: Tensor
    effective_charges: Tensor
    softening_widths: Tensor
    atom_mask: Tensor
    electron_counts: Tensor
    formal_charges: Tensor
    canonical_smiles: tuple[str, ...]
    inchikey14: tuple[str, ...]


def _standardize(smiles: str) -> tuple[Chem.Mol, str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    molecule = rdMolStandardize.Cleanup(molecule)
    molecule = rdMolStandardize.TautomerEnumerator().Canonicalize(molecule)
    Chem.RemoveStereochemistry(molecule)
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:  # pragma: no cover
        raise RuntimeError("RDKit failed to parse its canonical SMILES")
    return molecule, canonical


def _normalize_coordinates(
    molecule: Chem.Mol,
    extent: float,
    margin: float,
    target_bond_length: float,
) -> np.ndarray:
    molecule = Chem.Mol(molecule)
    rdDepictor.Compute2DCoords(molecule, canonOrient=True, clearConfs=True)
    conformer = molecule.GetConformer()
    coordinates = np.asarray(
        [
            [conformer.GetAtomPosition(index).x, conformer.GetAtomPosition(index).y]
            for index in range(molecule.GetNumAtoms())
        ],
        dtype=np.float32,
    )
    coordinates -= coordinates.mean(axis=0, keepdims=True)
    if molecule.GetNumBonds():
        lengths = [
            np.linalg.norm(coordinates[bond.GetBeginAtomIdx()] - coordinates[bond.GetEndAtomIdx()])
            for bond in molecule.GetBonds()
        ]
        median_length = max(float(np.median(lengths)), 1e-6)
    else:
        median_length = 1.0
    max_abs = max(float(np.abs(coordinates).max(initial=0.0)), 1e-6)
    scale = min(target_bond_length / median_length, extent * margin / max_abs)
    return coordinates * scale


def canonical_nuclei(
    smiles: str,
    *,
    extent: float = 1.0,
    margin: float = 0.82,
    target_bond_length: float = 0.22,
    collapse_hydrogens: bool = True,
) -> MoleculeNuclei:
    """Convert a molecule to deterministic valence-only 2D pseudo-nuclei.

    With collapsed hydrogens, each implicit hydrogen contributes one effective
    valence charge at its parent heavy-atom coordinate. This keeps hydrogen count
    while avoiding an unreadably dense explicit-H depiction.
    """

    molecule, canonical = _standardize(smiles)
    if not collapse_hydrogens:
        molecule = Chem.AddHs(molecule)
    coordinates = _normalize_coordinates(molecule, extent, margin, target_bond_length)
    periodic_table = Chem.GetPeriodicTable()
    charges: list[float] = []
    widths: list[float] = []
    for atom in molecule.GetAtoms():
        valence = float(periodic_table.GetNOuterElecs(atom.GetAtomicNum()))
        if collapse_hydrogens:
            valence += float(atom.GetTotalNumHs(includeNeighbors=True))
        charges.append(valence)
        # Element-dependent width distinguishes equal-valence rows of the table.
        covalent_radius = float(periodic_table.GetRcovalent(atom.GetAtomicNum()))
        widths.append(0.035 + 0.018 * covalent_radius)

    formal_charge = int(Chem.GetFormalCharge(molecule))
    electron_count = float(sum(charges) - formal_charge)
    if electron_count <= 0:
        raise ValueError("molecule has no valence electrons")
    inchikey = Chem.MolToInchiKey(molecule)
    return MoleculeNuclei(
        canonical_smiles=canonical,
        inchikey14=inchikey.split("-")[0],
        coordinates=coordinates,
        effective_charges=np.asarray(charges, dtype=np.float32),
        softening_widths=np.asarray(widths, dtype=np.float32),
        electron_count=electron_count,
        formal_charge=formal_charge,
    )


def collate_nuclei(
    molecules: list[MoleculeNuclei],
    device: torch.device | str,
) -> NuclearBatch:
    if not molecules:
        raise ValueError("cannot collate an empty molecule batch")
    max_atoms = max(len(molecule.effective_charges) for molecule in molecules)
    batch_size = len(molecules)
    coordinates = torch.zeros(batch_size, max_atoms, 2, device=device)
    charges = torch.zeros(batch_size, max_atoms, device=device)
    widths = torch.ones(batch_size, max_atoms, device=device)
    mask = torch.zeros(batch_size, max_atoms, device=device, dtype=torch.bool)
    for index, molecule in enumerate(molecules):
        atoms = len(molecule.effective_charges)
        coordinates[index, :atoms] = torch.from_numpy(molecule.coordinates).to(device)
        charges[index, :atoms] = torch.from_numpy(molecule.effective_charges).to(device)
        widths[index, :atoms] = torch.from_numpy(molecule.softening_widths).to(device)
        mask[index, :atoms] = True
    return NuclearBatch(
        coordinates=coordinates,
        effective_charges=charges,
        softening_widths=widths,
        atom_mask=mask,
        electron_counts=torch.tensor(
            [molecule.electron_count for molecule in molecules],
            device=device,
            dtype=torch.float32,
        ),
        formal_charges=torch.tensor(
            [molecule.formal_charge for molecule in molecules],
            device=device,
            dtype=torch.float32,
        ),
        canonical_smiles=tuple(molecule.canonical_smiles for molecule in molecules),
        inchikey14=tuple(molecule.inchikey14 for molecule in molecules),
    )
