"""Charge-conserving graph renderer for smooth molecular images."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Literal, NamedTuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdPartialCharges
from torch import Tensor
from torch.nn import functional

_PAULING_ELECTRONEGATIVITY = {
    1: 2.20,
    5: 2.04,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    14: 1.90,
    15: 2.19,
    16: 2.58,
    17: 3.16,
    34: 2.55,
    35: 2.96,
    53: 2.66,
}


@dataclass(frozen=True, slots=True)
class ExpectedChargeConfig:
    """Geometry and rasterization controls for :class:`ExpectedCharge2D`."""

    resolution: int = 192
    extent: float = 1.0
    margin: float = 0.82
    target_bond_length: float = 0.22
    nucleus_sigma_reference: float = 0.018
    atom_sigma_reference: float = 0.068
    covalent_radius_reference: float = 0.76
    vdw_radius_reference: float = 1.70
    bond_sigma_perpendicular: float = 0.030
    bond_sigma_parallel_fraction: float = 0.40
    pi_lobe_offset: float = 0.038
    pi_lobe_width_multiplier: float = 0.82
    electronegativity_shift: float = 0.12
    partial_charge_shift: float = 0.18
    max_bond_shift_fraction: float = 0.20
    lone_pair_offset_fraction: float = 0.58
    lone_pair_perpendicular_fraction: float = 0.48
    aromatic_width_multiplier: float = 1.90
    softsign_scale: float = 32.0
    coulomb_softening: float = 0.045
    explicit_hydrogens: bool = True
    hydrogen_influence: float = 1.0
    optimize_layout: bool = True
    layout_candidates: int = 4
    layout_samples_per_candidate: int = 8
    primitive_chunk_size: int = 128

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(slots=True)
class ExpectedChargeResult:
    field: Tensor
    signed_charge: Tensor
    nuclear_density: Tensor
    electron_density: Tensor
    bond_density: Tensor
    lone_pair_density: Tensor
    delocalized_density: Tensor
    electrostatic_potential: Tensor
    canonical_smiles: str
    atom_count: int
    hydrogen_count: int
    formal_charge: int
    expected_electron_count: float
    integrated_nuclear_charge: float
    integrated_electron_count: float
    integrated_charge: float
    coordinates: Tensor
    atomic_numbers: Tensor
    electronegativities: Tensor
    partial_charges: Tensor


@dataclass(slots=True)
class ExpectedChargeBatchResult:
    field: Tensor
    signed_charge: Tensor
    nuclear_density: Tensor
    electron_density: Tensor
    bond_density: Tensor
    lone_pair_density: Tensor
    delocalized_density: Tensor
    electrostatic_potential: Tensor
    canonical_smiles: list[str]
    atom_count: Tensor
    hydrogen_count: Tensor
    formal_charge: Tensor
    expected_electron_count: Tensor
    integrated_nuclear_charge: Tensor
    integrated_electron_count: Tensor
    integrated_charge: Tensor
    coordinates: list[Tensor]
    atomic_numbers: list[Tensor]
    electronegativities: list[Tensor]
    partial_charges: list[Tensor]

    def item(self, index: int) -> ExpectedChargeResult:
        return ExpectedChargeResult(
            field=self.field[index],
            signed_charge=self.signed_charge[index],
            nuclear_density=self.nuclear_density[index],
            electron_density=self.electron_density[index],
            bond_density=self.bond_density[index],
            lone_pair_density=self.lone_pair_density[index],
            delocalized_density=self.delocalized_density[index],
            electrostatic_potential=self.electrostatic_potential[index],
            canonical_smiles=self.canonical_smiles[index],
            atom_count=int(self.atom_count[index]),
            hydrogen_count=int(self.hydrogen_count[index]),
            formal_charge=int(self.formal_charge[index]),
            expected_electron_count=float(self.expected_electron_count[index]),
            integrated_nuclear_charge=float(self.integrated_nuclear_charge[index]),
            integrated_electron_count=float(self.integrated_electron_count[index]),
            integrated_charge=float(self.integrated_charge[index]),
            coordinates=self.coordinates[index],
            atomic_numbers=self.atomic_numbers[index],
            electronegativities=self.electronegativities[index],
            partial_charges=self.partial_charges[index],
        )


TrainingChannel = Literal["field", "signed_charge", "electrostatic_potential"]


@dataclass(slots=True)
class ExpectedChargeTrainingBatchResult:
    """Memory-efficient single-channel output for large-scale pretraining data."""

    channel: Tensor
    channel_name: TrainingChannel
    canonical_smiles: list[str]
    molecule_keys: list[str]
    formal_charge: Tensor
    expected_electron_count: Tensor
    integrated_charge: Tensor


class _Gaussian(NamedTuple):
    center_x: float
    center_y: float
    sigma: float
    electrons: float


class _Elliptical(NamedTuple):
    center_x: float
    center_y: float
    direction_x: float
    direction_y: float
    sigma_parallel: float
    sigma_perpendicular: float
    electrons: float


class _LayoutScore(NamedTuple):
    total: float
    crossings: int
    nonbonded_overlap: float
    atom_bond_overlap: float
    aspect_penalty: float
    ring_distortion: float


@dataclass(slots=True)
class CompiledExpectedChargeMolecule:
    canonical_smiles: str
    molecule_key: str
    coordinates: np.ndarray
    atomic_numbers: np.ndarray
    electronegativities: np.ndarray
    partial_charges: np.ndarray
    bonds: np.ndarray
    bond_orders: np.ndarray
    layout_score: float
    bond_crossings: int
    formal_charge: int
    expected_electron_count: float
    nuclear_density: list[_Gaussian]
    bond_density: list[_Elliptical]
    lone_pair_density: list[_Elliptical]
    delocalized_density: list[_Elliptical]


class ExpectedCharge2D:
    """Render a normalized valence-charge expectation field from a molecular graph.

    Every primitive is normalized on the finite raster before its coefficient is
    applied. Consequently, coefficients have units of electrons and the raw field
    obeys ``integral(nuclear_density - electron_density) == formal_charge`` up to
    floating-point error.
    """

    def __init__(self, config: ExpectedChargeConfig, device: torch.device | str) -> None:
        if not 0.0 <= config.hydrogen_influence <= 1.0:
            raise ValueError("hydrogen_influence must be between 0 and 1")
        if config.layout_candidates < 1 or config.layout_samples_per_candidate < 1:
            raise ValueError("layout candidate and sample counts must be positive")
        self.config = config
        self.device = torch.device(device)
        axis = torch.linspace(-config.extent, config.extent, config.resolution, device=self.device)
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        self.grid = torch.stack((grid_x, grid_y), dim=-1)
        self.grid_spacing = 2.0 * config.extent / (config.resolution - 1)
        self.pixel_area = self.grid_spacing**2
        padded_resolution = 2 * config.resolution
        displacement_indices = torch.arange(padded_resolution, device=self.device)
        displacements = torch.where(
            displacement_indices < config.resolution,
            displacement_indices,
            displacement_indices - padded_resolution,
        ).to(torch.float32)
        displacements = displacements * self.grid_spacing
        displacement_y, displacement_x = torch.meshgrid(
            displacements, displacements, indexing="ij"
        )
        radius_squared = displacement_x.square() + displacement_y.square()
        coulomb_kernel = torch.rsqrt(radius_squared + config.coulomb_softening**2)
        self._coulomb_fft_shape = (padded_resolution, padded_resolution)
        self._coulomb_kernel_fft = torch.fft.rfft2(coulomb_kernel)

    @staticmethod
    def _coordinates(molecule: Chem.Mol) -> np.ndarray:
        conformer = molecule.GetConformer()
        return np.asarray(
            [
                [conformer.GetAtomPosition(index).x, conformer.GetAtomPosition(index).y]
                for index in range(molecule.GetNumAtoms())
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _segments_cross(
        start_a: np.ndarray,
        end_a: np.ndarray,
        start_b: np.ndarray,
        end_b: np.ndarray,
    ) -> bool:
        def orientation(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
            first = end - start
            second = point - start
            return float(first[0] * second[1] - first[1] * second[0])

        first = orientation(start_a, end_a, start_b)
        second = orientation(start_a, end_a, end_b)
        third = orientation(start_b, end_b, start_a)
        fourth = orientation(start_b, end_b, end_a)
        tolerance = 1e-7
        return first * second < -tolerance and third * fourth < -tolerance

    @classmethod
    def _score_layout(cls, molecule: Chem.Mol, coordinates: np.ndarray) -> _LayoutScore:
        bonds = [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in molecule.GetBonds()
        ]
        if not bonds:
            return _LayoutScore(0.0, 0, 0.0, 0.0, 0.0, 0.0)
        crossings = 0
        for first_index, (start_a, end_a) in enumerate(bonds):
            for start_b, end_b in bonds[first_index + 1 :]:
                if {start_a, end_a} & {start_b, end_b}:
                    continue
                if cls._segments_cross(
                    coordinates[start_a],
                    coordinates[end_a],
                    coordinates[start_b],
                    coordinates[end_b],
                ):
                    crossings += 1
        return _LayoutScore(float(crossings), crossings, 0.0, 0.0, 0.0, 0.0)

    def _layout_heavy_atoms(
        self, molecule: Chem.Mol, canonical: str
    ) -> tuple[Chem.Mol, _LayoutScore]:
        baseline = Chem.Mol(molecule)
        rdDepictor.Compute2DCoords(
            baseline,
            canonOrient=True,
            clearConfs=True,
            useRingTemplates=True,
        )
        best_molecule = (
            Chem.AddHs(baseline, addCoords=True)
            if self.config.explicit_hydrogens
            else baseline
        )
        best_score = self._score_layout(
            best_molecule, self._coordinates(best_molecule)
        )
        if best_score.crossings == 0 or self.config.layout_candidates <= 1:
            return best_molecule, best_score

        seed = int.from_bytes(hashlib.sha256(canonical.encode("utf-8")).digest()[:4], "big")
        seed &= 0x7FFF_FFFF
        for candidate_index in range(1, self.config.layout_candidates):
            candidate = Chem.Mol(molecule)
            rdDepictor.Compute2DCoords(
                candidate,
                canonOrient=True,
                clearConfs=True,
                nFlipsPerSample=3,
                nSample=self.config.layout_samples_per_candidate,
                sampleSeed=(seed + candidate_index) % 0x7FFF_FFFF,
                permuteDeg4Nodes=True,
                useRingTemplates=True,
            )
            final_candidate = (
                Chem.AddHs(candidate, addCoords=True)
                if self.config.explicit_hydrogens
                else candidate
            )
            candidate_score = self._score_layout(
                final_candidate, self._coordinates(final_candidate)
            )
            if candidate_score.crossings < best_score.crossings:
                best_molecule = final_candidate
                best_score = candidate_score
                if best_score.crossings == 0:
                    break
        return best_molecule, best_score

    def _prepare_molecule(
        self, smiles: str
    ) -> tuple[Chem.Mol, str, np.ndarray, _LayoutScore]:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"invalid SMILES: {smiles!r}")
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None:  # pragma: no cover
            raise RuntimeError("RDKit could not reconstruct canonical molecule")
        heavy_molecule = molecule
        molecule = (
            Chem.AddHs(heavy_molecule)
            if self.config.explicit_hydrogens
            else Chem.Mol(heavy_molecule)
        )
        rdDepictor.Compute2DCoords(molecule, canonOrient=True, clearConfs=True)
        layout_score = self._score_layout(molecule, self._coordinates(molecule))
        if self.config.optimize_layout and layout_score.crossings > 0:
            optimized_molecule, optimized_score = self._layout_heavy_atoms(
                heavy_molecule, canonical
            )
            if optimized_score.crossings < layout_score.crossings:
                molecule = optimized_molecule
                layout_score = optimized_score
        coordinates = self._coordinates(molecule)
        heavy_mask = np.asarray(
            [atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()], dtype=bool
        )
        center_coordinates = coordinates[heavy_mask] if heavy_mask.any() else coordinates
        coordinates -= center_coordinates.mean(axis=0, keepdims=True)
        lengths = [
            np.linalg.norm(coordinates[bond.GetBeginAtomIdx()] - coordinates[bond.GetEndAtomIdx()])
            for bond in molecule.GetBonds()
            if (
                molecule.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum() > 1
                and molecule.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum() > 1
            )
        ]
        if not lengths:
            lengths = [
                np.linalg.norm(
                    coordinates[bond.GetBeginAtomIdx()] - coordinates[bond.GetEndAtomIdx()]
                )
                for bond in molecule.GetBonds()
            ]
        median_length = max(float(np.median(lengths)), 1e-6) if lengths else 1.0
        max_abs = max(float(np.abs(coordinates).max(initial=0.0)), 1e-6)
        scale = min(
            self.config.target_bond_length / median_length,
            self.config.extent * self.config.margin / max_abs,
        )
        return molecule, canonical, coordinates * scale, layout_score

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
        return np.asarray(
            (
                cosine * direction[0] - sine * direction[1],
                sine * direction[0] + cosine * direction[1],
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _lobe_angles(count: int) -> list[float]:
        if count <= 1:
            return [0.0]
        if count == 2:
            return [-0.62, 0.62]
        if count == 3:
            return [-0.92, 0.0, 0.92]
        return np.linspace(-1.15, 1.15, count).tolist()

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

    def _compile_molecule(self, smiles: str) -> CompiledExpectedChargeMolecule:
        molecule, canonical, coordinates, layout_score = self._prepare_molecule(smiles)
        molecule_key = Chem.MolToInchiKey(molecule)
        if not molecule_key:
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            molecule_key = f"SMILES-SHA256-{digest}"
        periodic_table = Chem.GetPeriodicTable()
        atomic_numbers = np.asarray(
            [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.int64
        )
        electronegativities = np.asarray(
            [_PAULING_ELECTRONEGATIVITY.get(int(number), 2.20) for number in atomic_numbers],
            dtype=np.float32,
        )
        partial_charges = self._gasteiger_charges(molecule)
        atom_sigmas: list[float] = []
        nuclear_density: list[_Gaussian] = []
        lone_pair_density: list[_Elliptical] = []
        valence_electrons: list[float] = []
        atom_weights = np.asarray(
            [
                self.config.hydrogen_influence if atom.GetAtomicNum() == 1 else 1.0
                for atom in molecule.GetAtoms()
            ],
            dtype=np.float32,
        )

        for index, atom in enumerate(molecule.GetAtoms()):
            valence = (
                float(periodic_table.GetNOuterElecs(atom.GetAtomicNum()))
                * float(atom_weights[index])
            )
            valence_electrons.append(valence)
            covalent_radius = float(periodic_table.GetRcovalent(atom.GetAtomicNum()))
            nucleus_sigma = self.config.nucleus_sigma_reference * math.sqrt(
                max(covalent_radius, 0.20) / self.config.covalent_radius_reference
            )
            atom_sigma = self.config.atom_sigma_reference * (
                float(periodic_table.GetRvdw(atom.GetAtomicNum()))
                / self.config.vdw_radius_reference
            )
            atom_sigmas.append(atom_sigma)
            if valence > 1e-8:
                nuclear_density.append(_Gaussian(*coordinates[index], nucleus_sigma, valence))

            bonded_share = float(atom_weights[index]) * sum(
                float(bond.GetBondTypeAsDouble()) for bond in atom.GetBonds()
            )
            nonbonding = max(valence - float(atom.GetFormalCharge()) - bonded_share, 0.0)
            if nonbonding <= 1e-6:
                continue
            lobe_count = max(1, min(4, math.ceil(nonbonding / 2.0)))
            away = self._away_direction(atom, coordinates)
            hybrid_scale = {
                Chem.HybridizationType.SP: 1.12,
                Chem.HybridizationType.SP2: 1.02,
                Chem.HybridizationType.SP3: 0.92,
            }.get(atom.GetHybridization(), 1.0)
            remaining = nonbonding
            for lobe_index, angle in enumerate(self._lobe_angles(lobe_count)):
                direction = self._rotate(away, angle)
                center = coordinates[index] + (
                    direction * atom_sigma * self.config.lone_pair_offset_fraction
                )
                lobes_left = lobe_count - lobe_index
                electrons = min(2.0, remaining) if lobes_left > 1 else remaining
                remaining -= electrons
                lone_pair_density.append(
                    _Elliptical(
                        *center,
                        *direction,
                        atom_sigma * hybrid_scale,
                        atom_sigma * self.config.lone_pair_perpendicular_fraction,
                        electrons,
                    )
                )

        bond_density: list[_Elliptical] = []
        delocalized_density: list[_Elliptical] = []
        for bond in molecule.GetBonds():
            start_index, end_index = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            start, end = coordinates[start_index], coordinates[end_index]
            vector = end - start
            length = max(float(np.linalg.norm(vector)), 1e-6)
            direction = vector / length
            normal = np.asarray((-direction[1], direction[0]), dtype=np.float32)
            start_weight = float(atom_weights[start_index])
            end_weight = float(atom_weights[end_index])
            combined_weight = max(start_weight + end_weight, 1e-8)
            shift = np.clip(
                self.config.electronegativity_shift
                * (electronegativities[end_index] - electronegativities[start_index])
                + self.config.partial_charge_shift
                * (partial_charges[start_index] - partial_charges[end_index]),
                -self.config.max_bond_shift_fraction,
                self.config.max_bond_shift_fraction,
            )
            center = (
                start_weight * start + end_weight * end
            ) / combined_weight + min(start_weight, end_weight) * shift * vector
            sigma_parallel = self.config.bond_sigma_parallel_fraction * length
            sigma_perpendicular = self.config.bond_sigma_perpendicular * math.sqrt(
                (atom_sigmas[start_index] + atom_sigmas[end_index])
                / (2.0 * self.config.atom_sigma_reference)
            )

            sigma_electrons = combined_weight
            bond_density.append(
                _Elliptical(
                    *center,
                    *direction,
                    sigma_parallel,
                    sigma_perpendicular,
                    sigma_electrons,
                )
            )
            extra_electrons = max(
                combined_weight * (float(bond.GetBondTypeAsDouble()) - 1.0), 0.0
            )
            if extra_electrons <= 1e-6:
                continue
            if bond.GetIsAromatic():
                # Aromatic pi density replaces, rather than augments, the extra
                # electron implied by RDKit's 1.5 bond order.
                midpoint = 0.5 * (start + end)
                delocalized_density.append(
                    _Elliptical(
                        *midpoint,
                        *direction,
                        sigma_parallel * 1.08,
                        sigma_perpendicular * self.config.aromatic_width_multiplier,
                        extra_electrons,
                    )
                )
                continue

            # Non-aromatic pi electrons form two lobes on either side of the
            # bond axis. Their combined integral equals the remaining bond count.
            for sign in (-1.0, 1.0):
                pi_center = center + sign * self.config.pi_lobe_offset * normal
                bond_density.append(
                    _Elliptical(
                        *pi_center,
                        *direction,
                        sigma_parallel,
                        sigma_perpendicular * self.config.pi_lobe_width_multiplier,
                        extra_electrons / 2.0,
                    )
                )

        formal_charge = int(Chem.GetFormalCharge(molecule))
        expected_electron_count = float(sum(valence_electrons) - formal_charge)
        assigned_electron_count = sum(
            primitive.electrons
            for primitives in (bond_density, lone_pair_density, delocalized_density)
            for primitive in primitives
        )
        if assigned_electron_count <= 0.0 and expected_electron_count > 0.0:
            raise ValueError(f"could not assign valence electrons for {canonical!r}")
        electron_scale = expected_electron_count / max(assigned_electron_count, 1e-12)
        bond_density = [
            primitive._replace(electrons=primitive.electrons * electron_scale)
            for primitive in bond_density
        ]
        lone_pair_density = [
            primitive._replace(electrons=primitive.electrons * electron_scale)
            for primitive in lone_pair_density
        ]
        delocalized_density = [
            primitive._replace(electrons=primitive.electrons * electron_scale)
            for primitive in delocalized_density
        ]
        return CompiledExpectedChargeMolecule(
            canonical_smiles=canonical,
            molecule_key=molecule_key,
            coordinates=coordinates,
            atomic_numbers=atomic_numbers,
            electronegativities=electronegativities,
            partial_charges=partial_charges,
            bonds=np.asarray(
                [
                    (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    for bond in molecule.GetBonds()
                ],
                dtype=np.int64,
            ).reshape(-1, 2),
            bond_orders=np.asarray(
                [bond.GetBondTypeAsDouble() for bond in molecule.GetBonds()],
                dtype=np.float32,
            ),
            layout_score=layout_score.total,
            bond_crossings=layout_score.crossings,
            formal_charge=formal_charge,
            expected_electron_count=expected_electron_count,
            nuclear_density=nuclear_density,
            bond_density=bond_density,
            lone_pair_density=lone_pair_density,
            delocalized_density=delocalized_density,
        )

    def compile_molecule(self, smiles: str) -> CompiledExpectedChargeMolecule:
        """Run the CPU/RDKit preparation stage without rasterizing the molecule."""
        return self._compile_molecule(smiles)

    def _rasterize_gaussians(
        self,
        batch_size: int,
        owners: list[int],
        primitives: list[_Gaussian],
    ) -> Tensor:
        output = torch.zeros(
            batch_size, self.config.resolution, self.config.resolution, device=self.device
        )
        self._accumulate_gaussians(output, owners, primitives)
        return output

    def _accumulate_gaussians(
        self,
        output: Tensor,
        owners: list[int],
        primitives: list[_Gaussian],
        coefficient: float = 1.0,
    ) -> None:
        if not primitives:
            return
        chunk_size = self.config.primitive_chunk_size
        for offset in range(0, len(primitives), chunk_size):
            chunk = primitives[offset : offset + chunk_size]
            values = torch.tensor(chunk, device=self.device, dtype=torch.float32)
            centers = values[:, None, None, :2]
            sigmas = values[:, None, None, 2]
            radius_squared = (self.grid[None] - centers).square().sum(dim=-1)
            kernels = torch.exp(-0.5 * radius_squared / sigmas.square())
            normalizers = (kernels.sum(dim=(1, 2)) * self.pixel_area).clamp_min(1e-12)
            images = values[:, None, None, 3] * kernels / normalizers[:, None, None]
            indices = torch.tensor(
                owners[offset : offset + len(chunk)], device=self.device, dtype=torch.long
            )
            output.index_add_(0, indices, images, alpha=coefficient)

    def _rasterize_ellipticals(
        self,
        batch_size: int,
        owners: list[int],
        primitives: list[_Elliptical],
    ) -> Tensor:
        output = torch.zeros(
            batch_size, self.config.resolution, self.config.resolution, device=self.device
        )
        self._accumulate_ellipticals(output, owners, primitives)
        return output

    def _accumulate_ellipticals(
        self,
        output: Tensor,
        owners: list[int],
        primitives: list[_Elliptical],
        coefficient: float = 1.0,
    ) -> None:
        if not primitives:
            return
        chunk_size = self.config.primitive_chunk_size
        for offset in range(0, len(primitives), chunk_size):
            chunk = primitives[offset : offset + chunk_size]
            values = torch.tensor(chunk, device=self.device, dtype=torch.float32)
            centers = values[:, None, None, :2]
            directions = values[:, None, None, 2:4]
            normals = torch.stack((-directions[..., 1], directions[..., 0]), dim=-1)
            relative = self.grid[None] - centers
            parallel = (relative * directions).sum(dim=-1)
            perpendicular = (relative * normals).sum(dim=-1)
            kernels = torch.exp(
                -0.5 * (parallel / values[:, None, None, 4]).square()
                - 0.5 * (perpendicular / values[:, None, None, 5]).square()
            )
            normalizers = (kernels.sum(dim=(1, 2)) * self.pixel_area).clamp_min(1e-12)
            images = values[:, None, None, 6] * kernels / normalizers[:, None, None]
            indices = torch.tensor(
                owners[offset : offset + len(chunk)], device=self.device, dtype=torch.long
            )
            output.index_add_(0, indices, images, alpha=coefficient)

    @staticmethod
    def _flatten_primitives(
        compiled: list[CompiledExpectedChargeMolecule], attribute: str
    ) -> tuple[list[int], list[_Gaussian] | list[_Elliptical]]:
        owners: list[int] = []
        primitives: list[_Gaussian] | list[_Elliptical] = []
        for owner, molecule in enumerate(compiled):
            values = getattr(molecule, attribute)
            owners.extend([owner] * len(values))
            primitives.extend(values)
        return owners, primitives

    def _electrostatic_potential(self, signed_charge: Tensor) -> Tensor:
        resolution = self.config.resolution
        padded_charge = functional.pad(signed_charge, (0, resolution, 0, resolution))
        transformed_charge = torch.fft.rfft2(padded_charge)
        padded_potential = torch.fft.irfft2(
            transformed_charge * self._coulomb_kernel_fft,
            s=self._coulomb_fft_shape,
        ) * self.pixel_area
        potential = padded_potential[..., :resolution, :resolution]
        return potential - potential.mean(dim=(-2, -1), keepdim=True)

    def render_training_compiled_batch(
        self,
        compiled: list[CompiledExpectedChargeMolecule],
        channel: TrainingChannel = "electrostatic_potential",
    ) -> ExpectedChargeTrainingBatchResult:
        """Rasterize only one training channel and discard diagnostic components.

        Component densities are accumulated into one electron-density buffer. This
        avoids retaining the eight visualization channels used by ``render_batch``.
        """
        if not compiled:
            raise ValueError("render_training_compiled_batch requires at least one molecule")
        if channel not in {"field", "signed_charge", "electrostatic_potential"}:
            raise ValueError(f"unsupported training channel: {channel}")

        batch_size = len(compiled)
        nuclear_owners, nuclear_primitives = self._flatten_primitives(
            compiled, "nuclear_density"
        )
        signed_charge = torch.zeros(
            batch_size, self.config.resolution, self.config.resolution, device=self.device
        )
        self._accumulate_gaussians(
            signed_charge, nuclear_owners, nuclear_primitives
        )
        electron_owners: list[int] = []
        electron_primitives: list[_Elliptical] = []
        for owner, molecule in enumerate(compiled):
            for primitives in (
                molecule.bond_density,
                molecule.lone_pair_density,
                molecule.delocalized_density,
            ):
                electron_owners.extend([owner] * len(primitives))
                electron_primitives.extend(primitives)
        self._accumulate_ellipticals(
            signed_charge,
            electron_owners,
            electron_primitives,
            coefficient=-1.0,
        )
        integrated_charge = signed_charge.sum(dim=(-2, -1)) * self.pixel_area
        if channel == "field":
            output = signed_charge / (self.config.softsign_scale + signed_charge.abs())
        elif channel == "signed_charge":
            output = signed_charge
        else:
            output = self._electrostatic_potential(signed_charge)
        return ExpectedChargeTrainingBatchResult(
            channel=output[:, None],
            channel_name=channel,
            canonical_smiles=[molecule.canonical_smiles for molecule in compiled],
            molecule_keys=[molecule.molecule_key for molecule in compiled],
            formal_charge=torch.tensor(
                [molecule.formal_charge for molecule in compiled], device=self.device
            ),
            expected_electron_count=torch.tensor(
                [molecule.expected_electron_count for molecule in compiled],
                device=self.device,
            ),
            integrated_charge=integrated_charge,
        )

    def render_training_batch(
        self,
        smiles: list[str] | tuple[str, ...],
        channel: TrainingChannel = "electrostatic_potential",
    ) -> ExpectedChargeTrainingBatchResult:
        """Compile and render a memory-efficient single-channel molecular batch."""
        compiled = [self._compile_molecule(value) for value in smiles]
        return self.render_training_compiled_batch(compiled, channel)

    def render_batch(
        self, smiles: list[str] | tuple[str, ...]
    ) -> ExpectedChargeBatchResult:
        if not smiles:
            raise ValueError("render_batch requires at least one SMILES")
        compiled = [self._compile_molecule(value) for value in smiles]
        return self.render_compiled_batch(compiled)

    def render_compiled_batch(
        self, compiled: list[CompiledExpectedChargeMolecule]
    ) -> ExpectedChargeBatchResult:
        """Render all diagnostic channels from precompiled CPU molecule data."""
        if not compiled:
            raise ValueError("render_compiled_batch requires at least one molecule")
        batch_size = len(compiled)
        nuclear_owners, nuclear_primitives = self._flatten_primitives(
            compiled, "nuclear_density"
        )
        bond_owners, bond_primitives = self._flatten_primitives(compiled, "bond_density")
        lone_owners, lone_primitives = self._flatten_primitives(
            compiled, "lone_pair_density"
        )
        deloc_owners, deloc_primitives = self._flatten_primitives(
            compiled, "delocalized_density"
        )
        nuclear_density = self._rasterize_gaussians(
            batch_size, nuclear_owners, nuclear_primitives
        )
        bond_density = self._rasterize_ellipticals(
            batch_size, bond_owners, bond_primitives
        )
        lone_pair_density = self._rasterize_ellipticals(
            batch_size, lone_owners, lone_primitives
        )
        delocalized_density = self._rasterize_ellipticals(
            batch_size, deloc_owners, deloc_primitives
        )
        electron_density = bond_density + lone_pair_density + delocalized_density
        signed_charge = nuclear_density - electron_density
        field = signed_charge / (self.config.softsign_scale + signed_charge.abs())
        potential = self._electrostatic_potential(signed_charge)
        integrated_nuclear = nuclear_density.sum(dim=(-2, -1)) * self.pixel_area
        integrated_electrons = electron_density.sum(dim=(-2, -1)) * self.pixel_area
        integrated_charge = signed_charge.sum(dim=(-2, -1)) * self.pixel_area

        return ExpectedChargeBatchResult(
            field=field[:, None],
            signed_charge=signed_charge[:, None],
            nuclear_density=nuclear_density[:, None],
            electron_density=electron_density[:, None],
            bond_density=bond_density[:, None],
            lone_pair_density=lone_pair_density[:, None],
            delocalized_density=delocalized_density[:, None],
            electrostatic_potential=potential[:, None],
            canonical_smiles=[molecule.canonical_smiles for molecule in compiled],
            atom_count=torch.tensor(
                [len(molecule.atomic_numbers) for molecule in compiled], device=self.device
            ),
            hydrogen_count=torch.tensor(
                [int((molecule.atomic_numbers == 1).sum()) for molecule in compiled],
                device=self.device,
            ),
            formal_charge=torch.tensor(
                [molecule.formal_charge for molecule in compiled], device=self.device
            ),
            expected_electron_count=torch.tensor(
                [molecule.expected_electron_count for molecule in compiled],
                device=self.device,
            ),
            integrated_nuclear_charge=integrated_nuclear,
            integrated_electron_count=integrated_electrons,
            integrated_charge=integrated_charge,
            coordinates=[
                torch.as_tensor(molecule.coordinates, device=self.device) for molecule in compiled
            ],
            atomic_numbers=[
                torch.as_tensor(molecule.atomic_numbers, device=self.device)
                for molecule in compiled
            ],
            electronegativities=[
                torch.as_tensor(molecule.electronegativities, device=self.device)
                for molecule in compiled
            ],
            partial_charges=[
                torch.as_tensor(molecule.partial_charges, device=self.device)
                for molecule in compiled
            ],
        )

    def render(self, smiles: str) -> ExpectedChargeResult:
        return self.render_batch([smiles]).item(0)
