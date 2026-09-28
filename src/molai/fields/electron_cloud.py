"""Chemistry-aware, batched pseudo-electron-cloud molecular images."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import NamedTuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdPartialCharges
from torch import Tensor

_PAULING_ELECTRONEGATIVITY = {
    1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 34: 2.55, 35: 2.96, 53: 2.66,
}


@dataclass(frozen=True, slots=True)
class ElectronCloudConfig:
    resolution: int = 192
    extent: float = 1.0
    margin: float = 0.82
    target_bond_length: float = 0.22
    core_sigma: float = 0.012
    atom_sigma_reference: float = 0.068
    vdw_radius_reference: float = 1.70
    bond_sigma_perpendicular: float = 0.030
    bond_sigma_parallel_fraction: float = 0.43
    electronegativity_shift: float = 0.12
    partial_charge_shift: float = 0.18
    max_bond_shift_fraction: float = 0.20
    lone_pair_offset_fraction: float = 0.48
    lone_pair_perpendicular_fraction: float = 0.52
    aromatic_width_multiplier: float = 1.85
    conjugated_width_multiplier: float = 1.45
    stereo_sigma: float = 0.020
    stereo_offset: float = 0.034
    atom_cloud_weight: float = 0.70
    bond_cloud_weight: float = 0.70
    delocalized_cloud_weight: float = 0.36
    stereo_weight: float = 0.30
    core_weight: float = 1.0
    softsign_scale: float = 2.0
    explicit_hydrogens: bool = True
    primitive_chunk_size: int = 128

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(slots=True)
class ElectronCloudResult:
    field: Tensor
    core: Tensor
    atom_cloud: Tensor
    bond_cloud: Tensor
    delocalized_cloud: Tensor
    stereo_field: Tensor
    canonical_smiles: str
    atom_count: int
    hydrogen_count: int
    coordinates: Tensor
    atomic_numbers: Tensor
    electronegativities: Tensor
    partial_charges: Tensor


@dataclass(slots=True)
class ElectronCloudBatchResult:
    field: Tensor
    core: Tensor
    atom_cloud: Tensor
    bond_cloud: Tensor
    delocalized_cloud: Tensor
    stereo_field: Tensor
    canonical_smiles: list[str]
    atom_count: Tensor
    hydrogen_count: Tensor
    coordinates: list[Tensor]
    atomic_numbers: list[Tensor]
    electronegativities: list[Tensor]
    partial_charges: list[Tensor]

    def item(self, index: int) -> ElectronCloudResult:
        """Extract one molecule without copying its raster channels."""
        return ElectronCloudResult(
            field=self.field[index],
            core=self.core[index],
            atom_cloud=self.atom_cloud[index],
            bond_cloud=self.bond_cloud[index],
            delocalized_cloud=self.delocalized_cloud[index],
            stereo_field=self.stereo_field[index],
            canonical_smiles=self.canonical_smiles[index],
            atom_count=int(self.atom_count[index]),
            hydrogen_count=int(self.hydrogen_count[index]),
            coordinates=self.coordinates[index],
            atomic_numbers=self.atomic_numbers[index],
            electronegativities=self.electronegativities[index],
            partial_charges=self.partial_charges[index],
        )


class _Gaussian(NamedTuple):
    center_x: float
    center_y: float
    sigma: float
    amplitude: float


class _Elliptical(NamedTuple):
    center_x: float
    center_y: float
    direction_x: float
    direction_y: float
    sigma_parallel: float
    sigma_perpendicular: float
    amplitude: float


@dataclass(slots=True)
class _CompiledMolecule:
    canonical_smiles: str
    coordinates: np.ndarray
    atomic_numbers: np.ndarray
    electronegativities: np.ndarray
    partial_charges: np.ndarray
    core: list[_Gaussian]
    atom_cloud: list[_Elliptical]
    bond_cloud: list[_Elliptical]
    delocalized_cloud: list[_Elliptical]
    stereo_field: list[_Gaussian]


class ElectronCloud2D:
    """Render one or many graph-faithful scalar fields.

    RDKit graph preparation stays on CPU. All Gaussian and elliptical primitives from
    a batch are then rasterized in bounded chunks on the selected Torch device.
    """

    def __init__(self, config: ElectronCloudConfig, device: torch.device | str) -> None:
        self.config = config
        self.device = torch.device(device)
        axis = torch.linspace(-config.extent, config.extent, config.resolution, device=self.device)
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        self.grid = torch.stack((grid_x, grid_y), dim=-1)

    def _prepare_molecule(self, smiles: str) -> tuple[Chem.Mol, str, np.ndarray]:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"invalid SMILES: {smiles!r}")
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None:  # pragma: no cover
            raise RuntimeError("RDKit could not reconstruct canonical molecule")
        if self.config.explicit_hydrogens:
            molecule = Chem.AddHs(molecule)
        rdDepictor.Compute2DCoords(molecule, canonOrient=True, clearConfs=True)
        conformer = molecule.GetConformer()
        coordinates = np.asarray(
            [[conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y]
             for i in range(molecule.GetNumAtoms())],
            dtype=np.float32,
        )
        coordinates -= coordinates.mean(axis=0, keepdims=True)
        lengths = [
            np.linalg.norm(coordinates[b.GetBeginAtomIdx()] - coordinates[b.GetEndAtomIdx()])
            for b in molecule.GetBonds()
        ]
        median_length = max(float(np.median(lengths)), 1e-6) if lengths else 1.0
        max_abs = max(float(np.abs(coordinates).max(initial=0.0)), 1e-6)
        scale = min(
            self.config.target_bond_length / median_length,
            self.config.extent * self.config.margin / max_abs,
        )
        return molecule, canonical, coordinates * scale

    @staticmethod
    def _gasteiger_charges(molecule: Chem.Mol) -> np.ndarray:
        rdPartialCharges.ComputeGasteigerCharges(molecule, nIter=12, throwOnParamFailure=False)
        charges: list[float] = []
        for atom in molecule.GetAtoms():
            try:
                charge = float(atom.GetProp("_GasteigerCharge"))
            except (KeyError, ValueError):
                charge = float(atom.GetFormalCharge())
            charges.append(charge if math.isfinite(charge) else float(atom.GetFormalCharge()))
        return np.asarray(charges, dtype=np.float32)

    @staticmethod
    def _rotate(direction: np.ndarray, angle: float) -> np.ndarray:
        cosine, sine = math.cos(angle), math.sin(angle)
        return np.asarray((
            cosine * direction[0] - sine * direction[1],
            sine * direction[0] + cosine * direction[1],
        ), dtype=np.float32)

    @staticmethod
    def _lobe_angles(count: int) -> list[float]:
        if count <= 1:
            return [0.0]
        if count == 2:
            return [-0.55, 0.55]
        if count == 3:
            return [-0.80, 0.0, 0.80]
        return np.linspace(-1.0, 1.0, count).tolist()

    @staticmethod
    def _away_direction(atom: Chem.Atom, coordinates: np.ndarray) -> np.ndarray:
        center = coordinates[atom.GetIdx()]
        directions = []
        for neighbor in atom.GetNeighbors():
            vector = coordinates[neighbor.GetIdx()] - center
            directions.append(vector / max(float(np.linalg.norm(vector)), 1e-6))
        if not directions:
            return np.asarray((1.0, 0.0), dtype=np.float32)
        away = -np.stack(directions).sum(axis=0)
        if float(np.linalg.norm(away)) < 1e-4:
            away = np.asarray((-directions[0][1], directions[0][0]), dtype=np.float32)
        return away / max(float(np.linalg.norm(away)), 1e-6)

    def _compile_stereo(self, molecule: Chem.Mol, coordinates: np.ndarray) -> list[_Gaussian]:
        stereo: list[_Gaussian] = []
        ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True, includeChirality=True))
        for atom in molecule.GetAtoms():
            tag = atom.GetChiralTag()
            if tag not in (
                Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
            ):
                continue
            neighbors = sorted(atom.GetNeighbors(), key=lambda candidate: ranks[candidate.GetIdx()])
            if not neighbors:
                continue
            direction = coordinates[neighbors[0].GetIdx()] - coordinates[atom.GetIdx()]
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            normal = np.asarray((-direction[1], direction[0]), dtype=np.float32)
            sign = 1.0 if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW else -1.0
            center = coordinates[atom.GetIdx()]
            positive = center + sign * self.config.stereo_offset * normal
            negative = center - sign * self.config.stereo_offset * normal
            stereo.append(_Gaussian(*positive, self.config.stereo_sigma, 1.0))
            stereo.append(_Gaussian(*negative, self.config.stereo_sigma, -1.0))
        for bond in molecule.GetBonds():
            bond_stereo = bond.GetStereo()
            if bond_stereo not in (
                Chem.BondStereo.STEREOE,
                Chem.BondStereo.STEREOZ,
                Chem.BondStereo.STEREOCIS,
                Chem.BondStereo.STEREOTRANS,
            ):
                continue
            start = coordinates[bond.GetBeginAtomIdx()]
            end = coordinates[bond.GetEndAtomIdx()]
            direction = end - start
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            normal = np.asarray((-direction[1], direction[0]), dtype=np.float32)
            sign = 1.0 if bond_stereo in (
                Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS
            ) else -1.0
            center = 0.5 * (start + end)
            positive = center + sign * self.config.stereo_offset * normal
            negative = center - sign * self.config.stereo_offset * normal
            stereo.append(_Gaussian(*positive, self.config.stereo_sigma, 1.0))
            stereo.append(_Gaussian(*negative, self.config.stereo_sigma, -1.0))
        return stereo

    def _compile_molecule(self, smiles: str) -> _CompiledMolecule:
        molecule, canonical, coordinates = self._prepare_molecule(smiles)
        periodic_table = Chem.GetPeriodicTable()
        atomic_numbers = np.asarray(
            [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.int64
        )
        electronegativities = np.asarray(
            [_PAULING_ELECTRONEGATIVITY.get(int(number), 2.20) for number in atomic_numbers],
            dtype=np.float32,
        )
        partial_charges = self._gasteiger_charges(molecule)
        core: list[_Gaussian] = []
        atom_cloud: list[_Elliptical] = []
        atom_sigmas: list[float] = []

        for index, atom in enumerate(molecule.GetAtoms()):
            valence = float(periodic_table.GetNOuterElecs(atom.GetAtomicNum()))
            bonded = sum(float(attached.GetBondTypeAsDouble()) for attached in atom.GetBonds())
            nonbonding = max(
                valence - float(atom.GetFormalCharge()) - bonded - float(partial_charges[index]),
                0.0,
            )
            atom_sigma = self.config.atom_sigma_reference * (
                float(periodic_table.GetRvdw(atom.GetAtomicNum()))
                / self.config.vdw_radius_reference
            )
            atom_sigmas.append(atom_sigma)
            core.append(_Gaussian(*coordinates[index], self.config.core_sigma, valence))
            if nonbonding <= 1e-6:
                continue
            lobe_count = max(1, min(4, math.ceil(nonbonding / 2.0)))
            away = self._away_direction(atom, coordinates)
            hybrid_scale = {
                Chem.HybridizationType.SP: 1.10,
                Chem.HybridizationType.SP2: 1.00,
                Chem.HybridizationType.SP3: 0.90,
            }.get(atom.GetHybridization(), 1.0)
            for angle in self._lobe_angles(lobe_count):
                direction = self._rotate(away, angle)
                center = coordinates[index] + (
                    direction * atom_sigma * self.config.lone_pair_offset_fraction
                )
                atom_cloud.append(_Elliptical(
                    *center, *direction, atom_sigma * hybrid_scale,
                    atom_sigma * self.config.lone_pair_perpendicular_fraction,
                    nonbonding / lobe_count,
                ))

        bond_cloud: list[_Elliptical] = []
        delocalized_cloud: list[_Elliptical] = []
        for bond in molecule.GetBonds():
            start_index, end_index = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            start, end = coordinates[start_index], coordinates[end_index]
            vector = end - start
            length = max(float(np.linalg.norm(vector)), 1e-6)
            direction = vector / length
            shift = np.clip(
                self.config.electronegativity_shift
                * (electronegativities[end_index] - electronegativities[start_index])
                + self.config.partial_charge_shift
                * (partial_charges[start_index] - partial_charges[end_index]),
                -self.config.max_bond_shift_fraction,
                self.config.max_bond_shift_fraction,
            )
            center = 0.5 * (start + end) + shift * vector
            sigma_parallel = self.config.bond_sigma_parallel_fraction * length
            sigma_perpendicular = self.config.bond_sigma_perpendicular * math.sqrt(
                (atom_sigmas[start_index] + atom_sigmas[end_index])
                / (2.0 * self.config.atom_sigma_reference)
            )
            bond_cloud.append(_Elliptical(
                *center, *direction, sigma_parallel, sigma_perpendicular,
                2.0 * float(bond.GetBondTypeAsDouble()),
            ))
            midpoint = 0.5 * (start + end)
            if bond.GetIsAromatic():
                delocalized_cloud.append(_Elliptical(
                    *midpoint, *direction, sigma_parallel * 1.08,
                    sigma_perpendicular * self.config.aromatic_width_multiplier, 1.0,
                ))
            elif bond.GetIsConjugated():
                delocalized_cloud.append(_Elliptical(
                    *midpoint, *direction, sigma_parallel,
                    sigma_perpendicular * self.config.conjugated_width_multiplier, 0.65,
                ))
        return _CompiledMolecule(
            canonical, coordinates, atomic_numbers, electronegativities, partial_charges,
            core, atom_cloud, bond_cloud, delocalized_cloud,
            self._compile_stereo(molecule, coordinates),
        )

    def _rasterize_gaussians(
        self,
        batch_size: int,
        owners: list[int],
        primitives: list[_Gaussian],
    ) -> Tensor:
        output = torch.zeros(
            batch_size, self.config.resolution, self.config.resolution, device=self.device
        )
        chunk_size = self.config.primitive_chunk_size
        for offset in range(0, len(primitives), chunk_size):
            chunk = primitives[offset:offset + chunk_size]
            values = torch.tensor(chunk, device=self.device, dtype=torch.float32)
            centers = values[:, None, None, :2]
            sigmas = values[:, None, None, 2]
            amplitudes = values[:, None, None, 3]
            radius_squared = (self.grid[None] - centers).square().sum(dim=-1)
            images = amplitudes * torch.exp(-0.5 * radius_squared / sigmas.square())
            indices = torch.tensor(
                owners[offset:offset + len(chunk)], device=self.device, dtype=torch.long
            )
            output.index_add_(0, indices, images)
        return output

    def _rasterize_ellipticals(
        self,
        batch_size: int,
        owners: list[int],
        primitives: list[_Elliptical],
    ) -> Tensor:
        output = torch.zeros(
            batch_size, self.config.resolution, self.config.resolution, device=self.device
        )
        chunk_size = self.config.primitive_chunk_size
        for offset in range(0, len(primitives), chunk_size):
            chunk = primitives[offset:offset + chunk_size]
            values = torch.tensor(chunk, device=self.device, dtype=torch.float32)
            centers = values[:, None, None, :2]
            directions = values[:, None, None, 2:4]
            normals = torch.stack((-directions[..., 1], directions[..., 0]), dim=-1)
            relative = self.grid[None] - centers
            parallel = (relative * directions).sum(dim=-1)
            perpendicular = (relative * normals).sum(dim=-1)
            images = values[:, None, None, 6] * torch.exp(
                -0.5 * (parallel / values[:, None, None, 4]).square()
                - 0.5 * (perpendicular / values[:, None, None, 5]).square()
            )
            indices = torch.tensor(
                owners[offset:offset + len(chunk)], device=self.device, dtype=torch.long
            )
            output.index_add_(0, indices, images)
        return output

    @staticmethod
    def _flatten_primitives(
        compiled: list[_CompiledMolecule],
        attribute: str,
    ) -> tuple[list[int], list[_Gaussian] | list[_Elliptical]]:
        owners: list[int] = []
        primitives: list[_Gaussian] | list[_Elliptical] = []
        for owner, molecule in enumerate(compiled):
            values = getattr(molecule, attribute)
            owners.extend([owner] * len(values))
            primitives.extend(values)
        return owners, primitives

    def render_batch(self, smiles: list[str] | tuple[str, ...]) -> ElectronCloudBatchResult:
        """Compile a molecular batch on CPU and rasterize it efficiently on the device."""
        if not smiles:
            raise ValueError("render_batch requires at least one SMILES")
        compiled = [self._compile_molecule(value) for value in smiles]
        batch_size = len(compiled)
        core_owners, core_primitives = self._flatten_primitives(compiled, "core")
        atom_owners, atom_primitives = self._flatten_primitives(compiled, "atom_cloud")
        bond_owners, bond_primitives = self._flatten_primitives(compiled, "bond_cloud")
        deloc_owners, deloc_primitives = self._flatten_primitives(
            compiled, "delocalized_cloud"
        )
        stereo_owners, stereo_primitives = self._flatten_primitives(compiled, "stereo_field")
        core = self._rasterize_gaussians(batch_size, core_owners, core_primitives)
        atom_cloud = self._rasterize_ellipticals(batch_size, atom_owners, atom_primitives)
        bond_cloud = self._rasterize_ellipticals(batch_size, bond_owners, bond_primitives)
        delocalized_cloud = self._rasterize_ellipticals(
            batch_size, deloc_owners, deloc_primitives
        )
        stereo_field = self._rasterize_gaussians(
            batch_size, stereo_owners, stereo_primitives
        )
        raw = (
            self.config.core_weight * core
            - self.config.atom_cloud_weight * atom_cloud
            - self.config.bond_cloud_weight * bond_cloud
            - self.config.delocalized_cloud_weight * delocalized_cloud
            + self.config.stereo_weight * stereo_field
        )
        field = raw / (self.config.softsign_scale + raw.abs())
        return ElectronCloudBatchResult(
            field=field[:, None], core=core[:, None], atom_cloud=atom_cloud[:, None],
            bond_cloud=bond_cloud[:, None], delocalized_cloud=delocalized_cloud[:, None],
            stereo_field=stereo_field[:, None],
            canonical_smiles=[molecule.canonical_smiles for molecule in compiled],
            atom_count=torch.tensor(
                [len(molecule.atomic_numbers) for molecule in compiled], device=self.device
            ),
            hydrogen_count=torch.tensor(
                [int((molecule.atomic_numbers == 1).sum()) for molecule in compiled],
                device=self.device,
            ),
            coordinates=[torch.as_tensor(molecule.coordinates, device=self.device)
                         for molecule in compiled],
            atomic_numbers=[torch.as_tensor(molecule.atomic_numbers, device=self.device)
                            for molecule in compiled],
            electronegativities=[torch.as_tensor(molecule.electronegativities, device=self.device)
                                 for molecule in compiled],
            partial_charges=[torch.as_tensor(molecule.partial_charges, device=self.device)
                             for molecule in compiled],
        )

    def render(self, smiles: str) -> ElectronCloudResult:
        """Render one molecule through the same code path used for batches."""
        return self.render_batch([smiles]).item(0)
