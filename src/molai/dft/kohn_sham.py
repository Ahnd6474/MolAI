"""GPU-accelerated two-dimensional valence Kohn-Sham teacher solver."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from molai.dft.layout import NuclearBatch
from molai.dft.solver import DFT2DConfig, OrbitalFreeDFT2D


@dataclass(frozen=True, slots=True)
class KohnSham2DConfig:
    resolution: int = 96
    extent: float = 1.0
    scf_iterations: int = 36
    orbital_steps: int = 20
    imaginary_time_step: float = 0.0025
    density_mixing: float = 0.35
    convergence_tolerance: float = 5e-4
    convergence_patience: int = 3
    external_softening: float = 0.055
    hartree_softening: float = 0.045
    initial_density_width: float = 0.10
    hartree_weight: float = 0.75
    exchange_weight: float = 0.65
    softsign_scale: float = 96.0
    orbital_probe_width: float = 0.075
    bond_ridge_width: float = 0.035
    bond_cutoff_ratio: float = 1.40
    seed: int = 17

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class KohnSham2DResult:
    field: Tensor
    signed_density: Tensor
    electron_density: Tensor
    core_density: Tensor
    deformation_density: Tensor
    electron_localization: Tensor
    bond_order_density: Tensor
    external_potential: Tensor
    orbitals: Tensor
    orbital_energies: Tensor
    occupancies: Tensor
    integrated_charge: Tensor
    iterations: int
    converged: bool
    final_density_change: float


class KohnSham2D:
    """Solve occupied 2D Kohn-Sham orbitals by split-step imaginary time."""

    def __init__(self, config: KohnSham2DConfig, device: torch.device | str) -> None:
        self.config = config
        self.device = torch.device(device)
        backend_config = DFT2DConfig(
            resolution=config.resolution,
            extent=config.extent,
            external_softening=config.external_softening,
            hartree_softening=config.hartree_softening,
            initial_density_width=config.initial_density_width,
            core_width_scale=1.0,
            hartree_weight=config.hartree_weight,
            exchange_weight=config.exchange_weight,
        )
        self.backend = OrbitalFreeDFT2D(backend_config, device)
        self.spacing = self.backend.spacing
        self.pixel_area = self.backend.pixel_area
        frequencies = (
            2.0 * math.pi * torch.fft.fftfreq(config.resolution, d=self.spacing, device=self.device)
        )
        ky, kx = torch.meshgrid(frequencies, frequencies, indexing="ij")
        self.kinetic_energy = 0.5 * (kx.square() + ky.square())
        self.kinetic_propagator = torch.exp(-config.imaginary_time_step * self.kinetic_energy)

    @staticmethod
    def _occupancies(electron_counts: Tensor) -> Tensor:
        orbitals = int(torch.ceil(electron_counts.max() / 2.0).item())
        indices = torch.arange(orbitals, device=electron_counts.device)[None]
        full = torch.floor(electron_counts / 2.0).long()[:, None]
        residual = (electron_counts - 2.0 * full.squeeze(1)).clamp(0.0, 2.0)
        occupancy = torch.where(indices < full, 2.0, 0.0)
        occupancy = torch.where(indices == full, residual[:, None], occupancy)
        return occupancy

    def _nuclear_charge_density(self, batch: NuclearBatch) -> Tensor:
        """Deposit point nuclei while exactly conserving their grid-integrated charge."""

        resolution = self.config.resolution
        grid_coordinates = (batch.coordinates + self.config.extent) / self.spacing
        x = grid_coordinates[..., 0].clamp(0.0, resolution - 1.0001)
        y = grid_coordinates[..., 1].clamp(0.0, resolution - 1.0001)
        x0 = x.floor().long()
        y0 = y.floor().long()
        x1 = (x0 + 1).clamp_max(resolution - 1)
        y1 = (y0 + 1).clamp_max(resolution - 1)
        wx = x - x0
        wy = y - y0

        density = torch.zeros(
            batch.coordinates.shape[0],
            resolution * resolution,
            device=self.device,
        )
        charge_density = batch.effective_charges * batch.atom_mask / self.pixel_area
        for ix, iy, weight in (
            (x0, y0, (1.0 - wx) * (1.0 - wy)),
            (x1, y0, wx * (1.0 - wy)),
            (x0, y1, (1.0 - wx) * wy),
            (x1, y1, wx * wy),
        ):
            density.scatter_add_(1, iy * resolution + ix, charge_density * weight)
        return density.reshape(-1, resolution, resolution)

    def _orthonormalize(self, orbitals: Tensor) -> Tensor:
        batch, count, height, width = orbitals.shape
        matrix = orbitals.reshape(batch, count, height * width).transpose(1, 2)
        matrix = matrix * math.sqrt(self.pixel_area)
        orthogonal, _ = torch.linalg.qr(matrix, mode="reduced")
        matrix = orthogonal / math.sqrt(self.pixel_area)
        return matrix.transpose(1, 2).reshape(batch, count, height, width)

    def _initialize_orbitals(self, initial_density: Tensor, count: int) -> Tensor:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config.seed)
        shape = (initial_density.shape[0], count, *initial_density.shape[-2:])
        orbitals = torch.randn(shape, device=self.device, generator=generator)
        orbitals[:, 0] = initial_density.sqrt()
        for _ in range(3):
            transformed = torch.fft.fft2(orbitals)
            orbitals = torch.fft.ifft2(transformed * self.kinetic_propagator[None, None]).real
        return self._orthonormalize(orbitals)

    def _propagate(self, orbitals: Tensor, potential: Tensor) -> Tensor:
        shifted = potential - potential.amin(dim=(-2, -1), keepdim=True)
        half_potential = torch.exp(
            (-0.5 * self.config.imaginary_time_step * shifted).clamp(-20.0, 0.0)
        )
        orbitals = orbitals * half_potential[:, None]
        orbitals = torch.fft.ifft2(
            torch.fft.fft2(orbitals) * self.kinetic_propagator[None, None]
        ).real
        orbitals = orbitals * half_potential[:, None]
        return self._orthonormalize(orbitals)

    def _orbital_energies(self, orbitals: Tensor, potential: Tensor) -> Tensor:
        kinetic_orbitals = torch.fft.ifft2(
            torch.fft.fft2(orbitals) * self.kinetic_energy[None, None]
        ).real
        hamiltonian_orbitals = kinetic_orbitals + potential[:, None] * orbitals
        return (orbitals * hamiltonian_orbitals).sum(dim=(-2, -1)) * self.pixel_area

    def _density(self, orbitals: Tensor, occupancies: Tensor) -> Tensor:
        return (orbitals.square() * occupancies[:, :, None, None]).sum(dim=1)

    def _effective_potential(self, density: Tensor, external: Tensor) -> Tensor:
        hartree = self.backend._hartree_potential(density)
        exchange_coefficient = 4.0 * math.sqrt(2.0) / (3.0 * math.sqrt(math.pi))
        exchange_potential = (
            -1.5
            * self.config.exchange_weight
            * exchange_coefficient
            * density.clamp_min(1e-10).sqrt()
        )
        return external + self.config.hartree_weight * hartree + exchange_potential

    def _electron_localization(
        self,
        orbitals: Tensor,
        occupancies: Tensor,
        density: Tensor,
    ) -> Tensor:
        dx_orbitals = torch.gradient(orbitals, spacing=self.spacing, dim=-1)[0]
        dy_orbitals = torch.gradient(orbitals, spacing=self.spacing, dim=-2)[0]
        kinetic_density = 0.5 * (
            occupancies[:, :, None, None] * (dx_orbitals.square() + dy_orbitals.square())
        ).sum(dim=1)
        dx_density = torch.gradient(density, spacing=self.spacing, dim=-1)[0]
        dy_density = torch.gradient(density, spacing=self.spacing, dim=-2)[0]
        weizsaecker_density = (
            (dx_density.square() + dy_density.square()) / density.clamp_min(1e-6) / 8.0
        )
        pauli_density = (kinetic_density - weizsaecker_density).clamp_min(0.0)
        uniform_density = (math.pi / 2.0) * density.square()
        return 1.0 / (1.0 + (pauli_density / uniform_density.clamp_min(1e-6)).square())

    def _bond_order_density(
        self,
        orbitals: Tensor,
        occupancies: Tensor,
        batch: NuclearBatch,
    ) -> Tensor:
        """Project the occupied density matrix onto atoms and rasterize its bonds."""

        displacement = self.backend.grid[None, None] - batch.coordinates[:, :, None, None, :]
        radius_squared = displacement.square().sum(dim=-1)
        probes = (
            torch.exp(-0.5 * radius_squared / self.config.orbital_probe_width**2)
            * batch.atom_mask[:, :, None, None]
        )
        probe_norm = (probes.square().sum(dim=(-2, -1)) * self.pixel_area).sqrt()
        probes = probes / probe_norm[:, :, None, None].clamp_min(1e-8)
        projections = torch.einsum("bkhw,bahw->bka", orbitals, probes) * self.pixel_area
        bond_orders = torch.einsum("bk,bka,bkc->bac", occupancies, projections, projections).abs()

        output = torch.zeros(
            orbitals.shape[0],
            self.config.resolution,
            self.config.resolution,
            device=orbitals.device,
        )
        grid = self.backend.grid
        for batch_index in range(orbitals.shape[0]):
            atom_count = int(batch.atom_mask[batch_index].sum())
            if atom_count < 2:
                continue
            coordinates = batch.coordinates[batch_index, :atom_count]
            distances = torch.cdist(coordinates, coordinates)
            nearest = distances.masked_fill(
                torch.eye(atom_count, device=distances.device, dtype=torch.bool),
                float("inf"),
            ).amin(dim=1)
            cutoff = self.config.bond_cutoff_ratio * nearest.median()
            pair_mask = torch.triu((distances <= cutoff) & (distances > 0), diagonal=1)
            pairs = torch.nonzero(pair_mask, as_tuple=False)
            if not len(pairs):
                continue
            weights = bond_orders[batch_index, pairs[:, 0], pairs[:, 1]]
            weights = weights / weights.amax().clamp_min(1e-8)
            for pair, weight in zip(pairs, weights, strict=True):
                start = coordinates[pair[0]]
                vector = coordinates[pair[1]] - start
                relative = grid - start
                progress = (relative * vector).sum(dim=-1) / vector.square().sum().clamp_min(1e-8)
                progress = progress.clamp(0.0, 1.0)
                closest = start + progress[..., None] * vector
                perpendicular_squared = (grid - closest).square().sum(dim=-1)
                ridge = torch.exp(-0.5 * perpendicular_squared / self.config.bond_ridge_width**2)
                output[batch_index] += weight * ridge
        return output / output.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    def solve(self, batch: NuclearBatch) -> KohnSham2DResult:
        external, _, atomic_density = self.backend._nuclear_fields(batch)
        core = self._nuclear_charge_density(batch)
        occupancies = self._occupancies(batch.electron_counts)
        orbitals = self._initialize_orbitals(atomic_density, occupancies.shape[1])
        density = atomic_density
        stable_steps = 0
        final_change = float("inf")
        converged = False
        iterations = self.config.scf_iterations

        with torch.no_grad():
            for scf_step in range(self.config.scf_iterations):
                potential = self._effective_potential(density, external)
                for _ in range(self.config.orbital_steps):
                    orbitals = self._propagate(orbitals, potential)
                energies = self._orbital_energies(orbitals, potential)
                order = energies.argsort(dim=1)
                orbitals = torch.gather(
                    orbitals,
                    1,
                    order[:, :, None, None].expand_as(orbitals),
                )
                energies = torch.gather(energies, 1, order)
                new_density = self._density(orbitals, occupancies)
                normalization = new_density.sum(dim=(-2, -1), keepdim=True) * self.pixel_area
                new_density = (
                    new_density
                    * batch.electron_counts[:, None, None]
                    / normalization.clamp_min(1e-8)
                )
                final_change = float(
                    (
                        (new_density - density).abs().sum(dim=(-2, -1))
                        * self.pixel_area
                        / batch.electron_counts
                    ).max()
                )
                density = (
                    1.0 - self.config.density_mixing
                ) * density + self.config.density_mixing * new_density
                if final_change < self.config.convergence_tolerance:
                    stable_steps += 1
                    if stable_steps >= self.config.convergence_patience:
                        converged = True
                        iterations = scf_step + 1
                        break
                else:
                    stable_steps = 0

            density = self._density(orbitals, occupancies)
            density = (
                density
                * batch.electron_counts[:, None, None]
                / (density.sum(dim=(-2, -1), keepdim=True) * self.pixel_area).clamp_min(1e-8)
            )
            potential = self._effective_potential(density, external)
            energies = self._orbital_energies(orbitals, potential)
            deformation = density - atomic_density
            localization = self._electron_localization(orbitals, occupancies, density)
            signed = core - density
            bond_order_density = self._bond_order_density(orbitals, occupancies, batch)
            # The unencoded signed density conserves charge. Softsign is only a bounded,
            # invertible representation for image-space learning.
            field = (signed / (self.config.softsign_scale + signed.abs()))[:, None]
            integrated_charge = signed.sum(dim=(-2, -1)) * self.pixel_area

        return KohnSham2DResult(
            field=field,
            signed_density=signed[:, None],
            electron_density=density[:, None],
            core_density=core[:, None],
            deformation_density=deformation[:, None],
            electron_localization=localization[:, None],
            bond_order_density=bond_order_density[:, None],
            external_potential=external[:, None],
            orbitals=orbitals,
            orbital_energies=energies,
            occupancies=occupancies,
            integrated_charge=integrated_charge,
            iterations=iterations,
            converged=converged,
            final_density_change=final_change,
        )
