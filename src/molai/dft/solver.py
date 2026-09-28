"""Batched GPU implementation of a 2D orbital-free pseudo-DFT solver.

This is a physics-inspired two-dimensional valence model, not an all-electron
three-dimensional quantum-chemistry calculation. Density positivity and electron
number are enforced exactly through a softmax parameterization. Hartree interactions
use a cached FFT convolution and all molecules in a batch are optimized together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from molai.dft.layout import NuclearBatch


@dataclass(frozen=True, slots=True)
class DFT2DConfig:
    resolution: int = 128
    extent: float = 1.0
    steps: int = 160
    learning_rate: float = 0.08
    convergence_tolerance: float = 5e-5
    convergence_patience: int = 6
    external_softening: float = 0.055
    hartree_softening: float = 0.045
    initial_density_width: float = 0.10
    core_width_scale: float = 0.70
    thomas_fermi_weight: float = 0.50
    weizsaecker_weight: float = 0.08
    hartree_weight: float = 0.75
    exchange_weight: float = 0.65
    field_tanh_scale: float = 0.035
    gradient_clip: float = 5.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class DFT2DResult:
    field: Tensor
    signed_density: Tensor
    electron_density: Tensor
    core_density: Tensor
    external_potential: Tensor
    total_energy: Tensor
    integrated_charge: Tensor
    iterations: int
    converged: bool
    final_relative_energy_change: float


class OrbitalFreeDFT2D:
    """Minimize a 2D Thomas-Fermi-Weizsaecker/Hartree/LDA functional."""

    def __init__(self, config: DFT2DConfig, device: torch.device | str) -> None:
        self.config = config
        self.device = torch.device(device)
        if config.resolution < 16:
            raise ValueError("resolution must be at least 16")
        axis = torch.linspace(
            -config.extent,
            config.extent,
            config.resolution,
            device=self.device,
            dtype=torch.float32,
        )
        self.spacing = float(axis[1] - axis[0])
        self.pixel_area = self.spacing**2
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        self.grid = torch.stack((grid_x, grid_y), dim=-1)
        self.hartree_kernel_fft = self._make_hartree_kernel_fft()

    def _make_hartree_kernel_fft(self) -> Tensor:
        size = self.config.resolution * 2
        indices = torch.arange(size, device=self.device, dtype=torch.float32)
        displacements = torch.where(indices < self.config.resolution, indices, indices - size)
        coordinates = displacements * self.spacing
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        kernel = torch.rsqrt(grid_x.square() + grid_y.square() + self.config.hartree_softening**2)
        return torch.fft.rfft2(kernel)

    def _nuclear_fields(self, batch: NuclearBatch) -> tuple[Tensor, Tensor, Tensor]:
        displacement = self.grid[None, None] - batch.coordinates[:, :, None, None, :]
        radius_squared = displacement.square().sum(dim=-1)
        mask = batch.atom_mask[:, :, None, None]
        charges = batch.effective_charges[:, :, None, None]
        widths = batch.softening_widths[:, :, None, None]

        external_width = torch.sqrt(widths.square() + self.config.external_softening**2)
        external = -(charges * torch.rsqrt(radius_squared + external_width.square()) * mask).sum(
            dim=1
        )

        core_width = widths * self.config.core_width_scale
        gaussians = torch.exp(-0.5 * radius_squared / core_width.square()) * mask
        normalization = gaussians.sum(dim=(-2, -1), keepdim=True) * self.pixel_area
        gaussians = gaussians / normalization.clamp_min(1e-12)
        core = (charges * gaussians).sum(dim=1)

        initial_width = widths + self.config.initial_density_width
        initial = torch.exp(-0.5 * radius_squared / initial_width.square()) * charges * mask
        initial = initial.sum(dim=1).clamp_min(1e-12)
        initial = initial / (initial.sum(dim=(-2, -1), keepdim=True) * self.pixel_area)
        initial = initial * batch.electron_counts[:, None, None]
        return external, core, initial

    def _hartree_potential(self, density: Tensor) -> Tensor:
        batch, height, width = density.shape
        padded = torch.zeros(batch, height * 2, width * 2, device=density.device)
        padded[:, :height, :width] = density
        potential = torch.fft.irfft2(
            torch.fft.rfft2(padded) * self.hartree_kernel_fft,
            s=padded.shape[-2:],
        )
        return potential[:, :height, :width] * self.pixel_area

    def _density_from_logits(self, logits: Tensor, electrons: Tensor) -> Tensor:
        probabilities = torch.softmax(logits.flatten(1), dim=-1).reshape_as(logits)
        return probabilities * electrons[:, None, None] / self.pixel_area

    def _energy(self, density: Tensor, external: Tensor) -> Tensor:
        area = self.pixel_area
        tf = (
            self.config.thomas_fermi_weight
            * torch.pi
            / 2.0
            * (density.square().sum(dim=(-2, -1)) * area)
        )

        dx = (density[:, :, 1:] - density[:, :, :-1]) / self.spacing
        dy = (density[:, 1:, :] - density[:, :-1, :]) / self.spacing
        nx = 0.5 * (density[:, :, 1:] + density[:, :, :-1])
        ny = 0.5 * (density[:, 1:, :] + density[:, :-1, :])
        weizsaecker = (
            self.config.weizsaecker_weight
            / 8.0
            * area
            * (
                (dx.square() / nx.clamp_min(1e-6)).sum(dim=(-2, -1))
                + (dy.square() / ny.clamp_min(1e-6)).sum(dim=(-2, -1))
            )
        )

        external_energy = (density * external).sum(dim=(-2, -1)) * area
        hartree_potential = self._hartree_potential(density)
        hartree = (
            0.5
            * self.config.hartree_weight
            * (density * hartree_potential).sum(dim=(-2, -1))
            * area
        )
        exchange_coefficient = 4.0 * (2.0**0.5) / (3.0 * torch.pi**0.5)
        exchange = (
            -self.config.exchange_weight
            * exchange_coefficient
            * density.clamp_min(0.0).pow(1.5).sum(dim=(-2, -1))
            * area
        )
        return tf + weizsaecker + external_energy + hartree + exchange

    def solve(self, batch: NuclearBatch) -> DFT2DResult:
        external, core, initial = self._nuclear_fields(batch)
        logits = torch.nn.Parameter(initial.log())
        optimizer = torch.optim.Adam([logits], lr=self.config.learning_rate)
        previous_energy: Tensor | None = None
        stable_steps = 0
        converged = False
        iterations = self.config.steps
        final_relative_change = float("inf")

        for step in range(self.config.steps):
            optimizer.zero_grad(set_to_none=True)
            density = self._density_from_logits(logits, batch.electron_counts)
            energy = self._energy(density, external)
            loss = (energy / batch.electron_counts).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("2D DFT energy became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([logits], self.config.gradient_clip)
            optimizer.step()

            detached = energy.detach()
            if previous_energy is not None:
                relative = (
                    (detached - previous_energy).abs() / previous_energy.abs().clamp_min(1.0)
                ).max()
                final_relative_change = float(relative)
                stable_steps = (
                    stable_steps + 1 if float(relative) < self.config.convergence_tolerance else 0
                )
                if stable_steps >= self.config.convergence_patience:
                    converged = True
                    iterations = step + 1
                    break
            previous_energy = detached

        with torch.no_grad():
            density = self._density_from_logits(logits, batch.electron_counts)
            total_energy = self._energy(density, external)
            signed = core - density
            field = torch.tanh(self.config.field_tanh_scale * signed)
            integrated_charge = signed.sum(dim=(-2, -1)) * self.pixel_area
        return DFT2DResult(
            field=field[:, None],
            signed_density=signed[:, None],
            electron_density=density[:, None],
            core_density=core[:, None],
            external_potential=external[:, None],
            total_energy=total_energy,
            integrated_charge=integrated_charge,
            iterations=iterations,
            converged=converged,
            final_relative_energy_change=final_relative_change,
        )
