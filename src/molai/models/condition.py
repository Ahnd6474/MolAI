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


class SegmentAttentionPool(nn.Module):
    """Multi-head learned-query attention over variable-length tensor segments."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("pooling dimension must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.norm = nn.LayerNorm(dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.query = nn.Parameter(torch.randn(heads, self.head_dim) * 0.02)
        self.output = nn.Linear(dim, dim)

    def forward(self, tokens: Tensor, segments: Tensor, segment_count: int) -> Tensor:
        if tokens.ndim != 2 or segments.shape != tokens.shape[:1]:
            raise ValueError("tokens must be [N,D] and segments must be [N]")
        normalized = self.norm(tokens)
        keys = self.key(normalized).reshape(-1, self.heads, self.head_dim)
        values = self.value(normalized).reshape(-1, self.heads, self.head_dim)
        scores = (keys * self.query[None]).sum(dim=-1) / math.sqrt(self.head_dim)
        maximum = scores.new_full((segment_count, self.heads), -torch.inf)
        maximum.scatter_reduce_(
            0,
            segments[:, None].expand(-1, self.heads),
            scores,
            reduce="amax",
            include_self=True,
        )
        weights = (scores - maximum[segments]).exp()
        denominator = scores.new_zeros(segment_count, self.heads)
        denominator.index_add_(0, segments, weights)
        weights = weights / denominator[segments].clamp_min(1e-12)
        pooled = values.new_zeros(segment_count, self.heads, self.head_dim)
        pooled.index_add_(0, segments, weights[..., None] * values)
        return self.output(pooled.reshape(segment_count, -1))


class SetFeedForwardBlock(nn.Module):
    def __init__(self, dim: int, ffn_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        return tokens + self.ffn(tokens)


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
        peak_chunk_batch: int = 256,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if max_peaks < 1:
            raise ValueError("max_peaks must be positive")
        if peak_chunk_batch < 1:
            raise ValueError("peak_chunk_batch must be positive")
        self.max_peaks = max_peaks
        self.peak_chunk_size = max_peaks
        self.peak_chunk_batch = peak_chunk_batch
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
        self.chunk_pool = SegmentAttentionPool(dim, heads)
        self.spectrum_blocks = nn.ModuleList(
            [SetFeedForwardBlock(dim, ffn_ratio, dropout) for _ in range(spectrum_layers)]
        )
        self.spectrum_pool = SegmentAttentionPool(dim, heads)
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
        if precursor_mz is None:
            masked_mz = peaks[..., 0].masked_fill(~peak_mask, 0.0)
            precursor_mz = masked_mz.amax(dim=-1)
        chunks: list[Tensor] = []
        chunk_masks: list[Tensor] = []
        chunk_to_spectrum: list[int] = []
        spectrum_to_molecule: list[int] = []
        metadata_parts: list[Tensor] = []
        precursor_parts: list[Tensor] = []
        spectrum_index = 0
        for batch_index in range(peaks.shape[0]):
            for local_spectrum in range(peaks.shape[1]):
                if not bool(spectrum_mask[batch_index, local_spectrum]):
                    continue
                values = peaks[batch_index, local_spectrum][
                    peak_mask[batch_index, local_spectrum]
                ]
                values = values[values[:, 0].argsort()]
                for start in range(0, len(values), self.peak_chunk_size):
                    chunk_values = values[start : start + self.peak_chunk_size]
                    chunk = peaks.new_zeros(self.peak_chunk_size, 2)
                    mask = torch.zeros(
                        self.peak_chunk_size, dtype=torch.bool, device=peaks.device
                    )
                    chunk[: len(chunk_values)] = chunk_values
                    mask[: len(chunk_values)] = True
                    chunks.append(chunk)
                    chunk_masks.append(mask)
                    chunk_to_spectrum.append(spectrum_index)
                metadata_parts.append(metadata[batch_index, local_spectrum])
                precursor_parts.append(precursor_mz[batch_index, local_spectrum])
                spectrum_to_molecule.append(batch_index)
                spectrum_index += 1
        return self.forward_ragged(
            torch.stack(chunks),
            torch.stack(chunk_masks),
            torch.tensor(chunk_to_spectrum, device=peaks.device),
            torch.tensor(spectrum_to_molecule, device=peaks.device),
            torch.stack(metadata_parts),
            torch.stack(precursor_parts),
            peaks.shape[0],
        )

    def forward_ragged(
        self,
        peak_chunks: Tensor,
        peak_mask: Tensor,
        chunk_to_spectrum: Tensor,
        spectrum_to_molecule: Tensor,
        metadata: Tensor,
        precursor_mz: Tensor,
        molecule_count: int,
    ) -> Tensor:
        """Encode all peaks and spectra from a CSR/chunked batch without truncation."""

        if peak_chunks.ndim != 3 or peak_chunks.shape[-1] != 2:
            raise ValueError("peak_chunks must have shape [K,P,2]")
        if peak_mask.shape != peak_chunks.shape[:2]:
            raise ValueError("peak_mask must have shape [K,P]")
        spectrum_count = metadata.shape[0]
        if chunk_to_spectrum.shape != peak_chunks.shape[:1]:
            raise ValueError("chunk_to_spectrum must have shape [K]")
        if spectrum_to_molecule.shape != (spectrum_count,):
            raise ValueError("spectrum_to_molecule must have shape [S]")
        if precursor_mz.shape != (spectrum_count,):
            raise ValueError("precursor_mz must have shape [S]")

        metadata_tokens = self.metadata_embedding(metadata)
        chunk_embedding_parts: list[Tensor] = []
        for start in range(0, len(peak_chunks), self.peak_chunk_batch):
            end = min(start + self.peak_chunk_batch, len(peak_chunks))
            chunk_slice = peak_chunks[start:end]
            mask_slice = peak_mask[start:end].bool()
            mapping_slice = chunk_to_spectrum[start:end]
            mz = torch.nan_to_num(
                chunk_slice[..., 0], nan=0.0, posinf=self.max_mz, neginf=0.0
            )
            intensity = torch.nan_to_num(
                chunk_slice[..., 1], nan=0.0, posinf=1.0, neginf=0.0
            )
            chunk_precursor = precursor_mz[mapping_slice]
            chunk_metadata = metadata_tokens[mapping_slice]
            peak_tokens = self.peak_embedding(mz, intensity, chunk_precursor)
            peak_tokens = peak_tokens + chunk_metadata[:, None]
            chunk_tokens = self.spectrum_token.expand(end - start, -1, -1)
            chunk_tokens = chunk_tokens + chunk_metadata[:, None]
            tokens = torch.cat((chunk_tokens, peak_tokens), dim=1)
            token_mask = torch.cat(
                (
                    torch.ones(
                        end - start, 1, dtype=torch.bool, device=peak_chunks.device
                    ),
                    mask_slice,
                ),
                dim=1,
            )
            masses = torch.cat((chunk_precursor[:, None], mz), dim=1)
            for block in self.peak_blocks:
                tokens = block(tokens, masses, token_mask)
            chunk_embedding_parts.append(tokens[:, 0])
        chunk_embeddings = torch.cat(chunk_embedding_parts)
        spectrum_embeddings = self.chunk_pool(
            chunk_embeddings, chunk_to_spectrum, spectrum_count
        )
        spectrum_embeddings = spectrum_embeddings + metadata_tokens
        for block in self.spectrum_blocks:
            spectrum_embeddings = block(spectrum_embeddings)
        molecule_embeddings = self.spectrum_pool(
            spectrum_embeddings, spectrum_to_molecule, molecule_count
        )
        molecule_embeddings = molecule_embeddings + self.molecule_token[:, 0]
        return self.output(molecule_embeddings)


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
