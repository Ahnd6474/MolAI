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


class ContinuousPeakEmbedding(nn.Module):
    """Encode a sparse peak as ``B(m/z) + g(I) A(m/z) + C(neutral-loss)``."""

    def __init__(
        self,
        dim: int,
        fourier_bands: int = 12,
        max_mz: float = 1_500.0,
        intensity_scale: float = 9.0,
    ) -> None:
        super().__init__()
        self.fourier_bands = fourier_bands
        self.max_mz = max_mz
        self.intensity_scale = intensity_scale
        position_dim = fourier_bands * 2 + 1
        self.base = nn.Sequential(
            nn.Linear(position_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.intensity_direction = nn.Sequential(
            nn.Linear(position_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.neutral_loss = nn.Sequential(
            nn.Linear(position_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.output = nn.LayerNorm(dim)

    def _position_features(self, values: Tensor) -> Tensor:
        normalized = values.clamp(min=0.0, max=self.max_mz) / self.max_mz
        frequencies = 2.0 ** torch.arange(
            self.fourier_bands,
            device=values.device,
            dtype=values.dtype,
        )
        angles = 2.0 * math.pi * normalized.unsqueeze(-1) * frequencies
        return torch.cat(
            (normalized.unsqueeze(-1), angles.sin(), angles.cos()), dim=-1
        )

    def forward(self, mz: Tensor, intensity: Tensor, precursor_mz: Tensor) -> Tensor:
        mz_features = self._position_features(mz)
        loss = (precursor_mz.unsqueeze(-1) - mz).clamp_min(0.0)
        loss_features = self._position_features(loss)
        amplitude = torch.log1p(self.intensity_scale * intensity.clamp_min(0.0))
        amplitude = amplitude / math.log1p(self.intensity_scale)
        encoded = self.base(mz_features)
        encoded = encoded + amplitude.unsqueeze(-1) * self.intensity_direction(mz_features)
        encoded = encoded + self.neutral_loss(loss_features)
        return self.output(encoded)


class RelativeMassSelfAttention(nn.Module):
    """Self-attention with a learned per-head bias for absolute peak mass gaps."""

    def __init__(
        self,
        dim: int,
        heads: int,
        relative_bands: int = 16,
        relative_bins: int = 512,
        relative_mass_max: float = 256.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("attention dim must be divisible by heads")
        if relative_bands < 2 or relative_bins < 2 or relative_mass_max <= 0.0:
            raise ValueError("relative mass configuration must be positive")
        self.heads = heads
        self.head_dim = dim // heads
        self.relative_mass_max = relative_mass_max
        self.relative_bins = relative_bins
        self.qkv = nn.Linear(dim, dim * 3)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.relative_projection = nn.Linear(relative_bands, heads, bias=False)
        centers = torch.linspace(0.0, 1.0, relative_bands)
        positions = torch.linspace(0.0, 1.0, relative_bins)
        width = 1.0 / (relative_bands - 1)
        relative_basis = torch.exp(
            -0.5 * ((positions[:, None] - centers[None]) / width).square()
        )
        self.register_buffer("relative_basis", relative_basis, persistent=False)

    def _relative_bias(self, masses: Tensor) -> Tensor:
        difference = (masses.unsqueeze(-1) - masses.unsqueeze(-2)).abs()
        normalized = difference.clamp_max(self.relative_mass_max) / self.relative_mass_max
        buckets = (normalized * (self.relative_bins - 1)).round().long()
        bias_table = self.relative_projection(self.relative_basis)
        return bias_table[buckets].permute(0, 3, 1, 2)

    def forward(self, tokens: Tensor, masses: Tensor, mask: Tensor) -> Tensor:
        batch, length, dim = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        scores = scores + self._relative_bias(masses).float()
        scores = scores.masked_fill(~mask[:, None, None, :], -torch.inf)
        weights = self.dropout(scores.softmax(dim=-1)).to(value.dtype)
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch, length, dim)
        attended = attended.masked_fill(~mask.unsqueeze(-1), 0.0)
        return self.output(attended)


class RelativeMassTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        relative_bands: int,
        relative_mass_max: float,
        ffn_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden = int(dim * ffn_ratio)
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = RelativeMassSelfAttention(
            dim,
            heads,
            relative_bands=relative_bands,
            relative_mass_max=relative_mass_max,
            dropout=dropout,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor, masses: Tensor, mask: Tensor) -> Tensor:
        tokens = tokens + self.attention(self.attention_norm(tokens), masses, mask)
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens.masked_fill(~mask.unsqueeze(-1), 0.0)


class SpectrumConditionEncoder(nn.Module):
    """Hierarchical attention encoder for sparse peaks and replicate spectra."""

    def __init__(
        self,
        metadata_dim: int,
        dim: int = 384,
        fourier_bands: int = 12,
        heads: int = 8,
        peak_layers: int = 3,
        spectrum_layers: int = 2,
        relative_bands: int = 16,
        relative_mass_max: float = 256.0,
        ffn_ratio: float = 2.0,
        dropout: float = 0.1,
        max_mz: float = 1_500.0,
        max_peaks: int = 256,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if max_peaks < 1:
            raise ValueError("max_peaks must be positive")
        self.max_peaks = max_peaks
        self.max_mz = max_mz
        self.peak_embedding = ContinuousPeakEmbedding(
            dim,
            fourier_bands=fourier_bands,
            max_mz=max_mz,
        )
        self.metadata_embedding = nn.Sequential(
            nn.Linear(metadata_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.spectrum_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.peak_blocks = nn.ModuleList(
            [
                RelativeMassTransformerBlock(
                    dim,
                    heads,
                    relative_bands,
                    relative_mass_max,
                    ffn_ratio,
                    dropout,
                )
                for _ in range(peak_layers)
            ]
        )
        spectrum_layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim_feedforward=int(dim * ffn_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spectrum_encoder = nn.TransformerEncoder(
            spectrum_layer,
            num_layers=spectrum_layers,
            enable_nested_tensor=False,
        )
        self.molecule_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    @staticmethod
    def _validate_inputs(
        peaks: Tensor,
        peak_mask: Tensor,
        spectrum_mask: Tensor,
        metadata: Tensor,
        precursor_mz: Tensor | None,
    ) -> None:
        if peaks.ndim != 4 or peaks.shape[-1] != 2:
            raise ValueError("peaks must have shape [B,S,P,2]")
        if peak_mask.shape != peaks.shape[:-1]:
            raise ValueError("peak_mask must have shape [B,S,P]")
        if spectrum_mask.shape != peaks.shape[:2]:
            raise ValueError("spectrum_mask must have shape [B,S]")
        if metadata.ndim != 3 or metadata.shape[:2] != peaks.shape[:2]:
            raise ValueError("metadata must have shape [B,S,M]")
        if precursor_mz is not None and precursor_mz.shape != peaks.shape[:2]:
            raise ValueError("precursor_mz must have shape [B,S]")

    def forward(
        self,
        peaks: Tensor,
        peak_mask: Tensor,
        spectrum_mask: Tensor,
        metadata: Tensor,
        precursor_mz: Tensor | None = None,
    ) -> Tensor:
        """Encode peaks `[B,S,P,2]` and metadata `[B,S,M]` into `[B,D]`."""
        self._validate_inputs(peaks, peak_mask, spectrum_mask, metadata, precursor_mz)
        peak_mask = peak_mask.bool()
        spectrum_mask = spectrum_mask.bool()
        if peaks.shape[2] > self.max_peaks:
            ranking = torch.nan_to_num(peaks[..., 1], nan=-torch.inf)
            ranking = ranking.masked_fill(~peak_mask, -torch.inf)
            indices = ranking.topk(self.max_peaks, dim=2).indices
            peaks = peaks.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, 2))
            peak_mask = peak_mask.gather(2, indices)
        batch, spectra, peaks_per_spectrum, _ = peaks.shape
        mz = torch.nan_to_num(
            peaks[..., 0], nan=0.0, posinf=self.max_mz, neginf=0.0
        )
        intensity = torch.nan_to_num(peaks[..., 1], nan=0.0, posinf=1.0, neginf=0.0)
        if precursor_mz is None:
            masked_mz = mz.masked_fill(~peak_mask, 0.0)
            precursor_mz = masked_mz.amax(dim=-1)

        peak_tokens = self.peak_embedding(mz, intensity, precursor_mz)
        metadata_tokens = self.metadata_embedding(metadata)
        peak_tokens = peak_tokens + metadata_tokens.unsqueeze(2)
        spectrum_tokens = self.spectrum_token.expand(batch, spectra, -1)
        spectrum_tokens = spectrum_tokens + metadata_tokens

        tokens = torch.cat((spectrum_tokens.unsqueeze(2), peak_tokens), dim=2)
        token_mask = torch.cat(
            (
                torch.ones(
                    batch,
                    spectra,
                    1,
                    dtype=torch.bool,
                    device=peaks.device,
                ),
                peak_mask,
            ),
            dim=2,
        )
        masses = torch.cat((precursor_mz.unsqueeze(-1), mz), dim=2)
        flat_tokens = tokens.reshape(batch * spectra, peaks_per_spectrum + 1, -1)
        flat_masses = masses.reshape(batch * spectra, peaks_per_spectrum + 1)
        flat_mask = token_mask.reshape(batch * spectra, peaks_per_spectrum + 1)
        for block in self.peak_blocks:
            flat_tokens = block(flat_tokens, flat_masses, flat_mask)
        spectrum_embeddings = flat_tokens[:, 0].reshape(batch, spectra, -1)

        molecule_token = self.molecule_token.expand(batch, -1, -1)
        molecule_tokens = torch.cat((molecule_token, spectrum_embeddings), dim=1)
        molecule_mask = torch.cat(
            (
                torch.ones(batch, 1, dtype=torch.bool, device=peaks.device),
                spectrum_mask,
            ),
            dim=1,
        )
        molecule_tokens = self.spectrum_encoder(
            molecule_tokens,
            src_key_padding_mask=~molecule_mask,
        )
        return self.output(molecule_tokens[:, 0])


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
