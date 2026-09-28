"""SMILES and spectrum encoders sharing one conditioning interface."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SmilesConditionEncoder(nn.Module):
    """Encode tokenized SMILES without imposing image-space global attention."""

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        dim: int = 384,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.encoder = nn.GRU(
            dim,
            dim // 2,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, token_ids: Tensor) -> Tensor:
        mask = token_ids.ne(self.pad_token_id)
        encoded, _ = self.encoder(self.embedding(token_ids))
        weights = mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.output(pooled)


class SpectrumConditionEncoder(nn.Module):
    """Hierarchical set encoder for multiple spectra from one molecule."""

    def __init__(
        self,
        metadata_dim: int,
        dim: int = 384,
        fourier_bands: int = 12,
    ) -> None:
        super().__init__()
        self.fourier_bands = fourier_bands
        peak_input = fourier_bands * 4 + 2
        self.peak_mlp = nn.Sequential(
            nn.Linear(peak_input, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.spectrum_mlp = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.molecule_mlp = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def _fourier(self, values: Tensor) -> Tensor:
        frequencies = 2.0 ** torch.arange(
            self.fourier_bands, device=values.device, dtype=values.dtype
        )
        angles = 2.0 * math.pi * values.unsqueeze(-1) * frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    def forward(
        self,
        peaks: Tensor,
        peak_mask: Tensor,
        spectrum_mask: Tensor,
        metadata: Tensor,
    ) -> Tensor:
        """Encode `[B,S,P,2]` m/z-intensity peaks and `[B,S,M]` metadata."""

        if peaks.ndim != 4 or peaks.shape[-1] != 2:
            raise ValueError("peaks must have shape [B,S,P,2]")
        mz = peaks[..., 0] / 1_500.0
        intensity = peaks[..., 1].clamp_min(0.0)
        features = torch.cat(
            (
                self._fourier(mz),
                self._fourier(intensity),
                mz.unsqueeze(-1),
                intensity.unsqueeze(-1),
            ),
            dim=-1,
        )
        encoded = self.peak_mlp(features)
        peak_weights = peak_mask.unsqueeze(-1).to(encoded.dtype)
        peak_mean = (encoded * peak_weights).sum(dim=2) / peak_weights.sum(dim=2).clamp_min(1.0)
        masked = encoded.masked_fill(~peak_mask.unsqueeze(-1), -torch.inf)
        peak_max = masked.amax(dim=2)
        peak_max = torch.where(torch.isfinite(peak_max), peak_max, torch.zeros_like(peak_max))
        spectrum = self.spectrum_mlp(
            torch.cat((peak_mean, peak_max, self.metadata_mlp(metadata)), dim=-1)
        )
        spectrum_weights = spectrum_mask.unsqueeze(-1).to(spectrum.dtype)
        molecule_mean = (spectrum * spectrum_weights).sum(dim=1) / spectrum_weights.sum(
            dim=1
        ).clamp_min(1.0)
        spectrum_masked = spectrum.masked_fill(~spectrum_mask.unsqueeze(-1), -torch.inf)
        molecule_max = spectrum_masked.amax(dim=1)
        molecule_max = torch.where(
            torch.isfinite(molecule_max), molecule_max, torch.zeros_like(molecule_max)
        )
        return self.molecule_mlp(torch.cat((molecule_mean, molecule_max), dim=-1))


class AxialConditionPlane(nn.Module):
    """Lift a condition vector to a low-rank full-resolution 2D context plane."""

    def __init__(
        self,
        condition_dim: int,
        spatial_dim: int,
        max_resolution: int = 256,
        rank: int = 8,
    ) -> None:
        super().__init__()
        self.max_resolution = max_resolution
        self.rank = rank
        self.row_basis = nn.Parameter(torch.randn(max_resolution, rank) * 0.02)
        self.column_basis = nn.Parameter(torch.randn(max_resolution, rank) * 0.02)
        self.row_coefficients = nn.Linear(condition_dim, rank * spatial_dim)
        self.column_coefficients = nn.Linear(condition_dim, rank * spatial_dim)
        self.global_affine = nn.Linear(condition_dim, spatial_dim * 2)
        self.norm = nn.LayerNorm(spatial_dim)

    def forward(self, condition: Tensor, height: int, width: int) -> Tensor:
        if height > self.max_resolution or width > self.max_resolution:
            raise ValueError("spatial resolution exceeds condition projector capacity")
        batch = condition.shape[0]
        dim = self.global_affine.out_features // 2
        row_coefficients = self.row_coefficients(condition).reshape(batch, self.rank, dim)
        column_coefficients = self.column_coefficients(condition).reshape(batch, self.rank, dim)
        rows = torch.einsum("hr,brd->bhd", self.row_basis[:height], row_coefficients)
        columns = torch.einsum("wr,brd->bwd", self.column_basis[:width], column_coefficients)
        scale, shift = self.global_affine(condition).chunk(2, dim=-1)
        plane = rows[:, :, None, :] + columns[:, None, :, :]
        plane = plane * (1.0 + scale[:, None, None, :]) + shift[:, None, None, :]
        return self.norm(plane)
