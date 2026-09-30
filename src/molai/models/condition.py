"""SMILES and spectrum encoders sharing one conditioning interface."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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


class PositionwiseAffinePeakEmbedding(nn.Module):
    """Give every discretized m/z position its own learnable affine vectors."""

    def __init__(
        self,
        dim: int,
        position_dim: int = 32,
        mz_bin_width: float = 0.01,
        mz_upper_bound: float = 5_000.0,
    ) -> None:
        super().__init__()
        if position_dim < 1 or mz_bin_width <= 0.0 or mz_upper_bound <= 0.0:
            raise ValueError("peak position configuration must be positive")
        self.mz_bin_width = mz_bin_width
        self.mz_upper_bound = mz_upper_bound
        self.position_count = math.ceil(mz_upper_bound / mz_bin_width) + 1
        self.slope = nn.Embedding(self.position_count, position_dim)
        self.intercept_table = nn.Embedding(self.position_count, position_dim)
        self.output = nn.Linear(position_dim, dim)
        nn.init.normal_(self.slope.weight, std=0.02)
        nn.init.normal_(self.intercept_table.weight, std=0.02)

    def position_indices(self, mz: Tensor) -> Tensor:
        return (mz / self.mz_bin_width).round().long().clamp(0, self.position_count - 1)

    def coefficients(self, mz: Tensor) -> tuple[Tensor, Tensor]:
        positions = self.position_indices(mz)
        slope = F.linear(self.slope(positions), self.output.weight)
        intercept = F.linear(
            self.intercept_table(positions), self.output.weight, self.output.bias
        )
        return slope, intercept

    def intercept(self, mz: Tensor) -> Tensor:
        return self.coefficients(mz)[1]

    def forward(self, mz: Tensor, intensity: Tensor) -> Tensor:
        slope, intercept = self.coefficients(mz)
        return slope * intensity.unsqueeze(-1) + intercept


class ExponentialDistanceConvBlock(nn.Module):
    """Indexwise convolution gated by compact exponential m/z distance kernels."""

    def __init__(
        self,
        dim: int,
        heads: int,
        kernel_size: int = 5,
        tau_min: float = 0.005,
        tau_max: float = 2.0,
        cutoff_multiplier: float = 8.0,
        ffn_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("convolution dim must be divisible by heads")
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd number")
        if not 0.0 < tau_min <= tau_max or cutoff_multiplier <= 0.0:
            raise ValueError("exponential distance scales must be positive")
        self.heads = heads
        self.head_dim = dim // heads
        self.kernel_size = kernel_size
        self.radius = kernel_size // 2
        self.cutoff_multiplier = cutoff_multiplier
        initial_tau = torch.logspace(
            math.log10(tau_min), math.log10(tau_max), heads
        )
        self.raw_tau = nn.Parameter(torch.log(torch.expm1(initial_tau)))
        self.offset_weight = nn.Parameter(torch.full((kernel_size, heads), 1.0 / kernel_size))
        self.norm = nn.LayerNorm(dim)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim)
        hidden = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def _distance_gate(self, distance: Tensor) -> Tensor:
        tau = F.softplus(self.raw_tau.float()).clamp_min(1e-5)
        cutoff = self.cutoff_multiplier * tau
        baseline = math.exp(-self.cutoff_multiplier)
        decay = (torch.exp(-distance[..., None].float() / tau) - baseline) / (
            1.0 - baseline
        )
        return decay.clamp_min(0.0) * (distance[..., None] < cutoff)

    def forward(
        self,
        tokens: Tensor,
        lower_mz: Tensor,
        upper_mz: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if tokens.ndim != 3 or lower_mz.shape != tokens.shape[:2]:
            raise ValueError("distance convolution expects [B,L,D] tokens and [B,L] masses")
        if upper_mz.shape != lower_mz.shape or mask.shape != lower_mz.shape:
            raise ValueError("mass intervals and mask must have shape [B,L]")
        batch, length, dim = tokens.shape
        normalized = self.norm(tokens)
        values = self.value(normalized).reshape(batch, length, self.heads, self.head_dim)
        padded_values = F.pad(values, (0, 0, 0, 0, self.radius, self.radius))
        neighbors = padded_values.unfold(1, self.kernel_size, 1)
        neighbors = neighbors.permute(0, 1, 4, 2, 3)

        padded_lower = F.pad(lower_mz, (self.radius, self.radius))
        padded_upper = F.pad(upper_mz, (self.radius, self.radius))
        neighbor_lower = padded_lower.unfold(1, self.kernel_size, 1)
        neighbor_upper = padded_upper.unfold(1, self.kernel_size, 1)
        left_gap = lower_mz[:, :, None] - neighbor_upper
        right_gap = neighbor_lower - upper_mz[:, :, None]
        distance = torch.maximum(left_gap, right_gap).clamp_min(0.0)

        padded_mask = F.pad(mask, (self.radius, self.radius), value=False)
        neighbor_mask = padded_mask.unfold(1, self.kernel_size, 1)
        gate = self._distance_gate(distance)
        gate = gate * neighbor_mask[..., None] * mask[:, :, None, None]
        gate = gate * self.offset_weight[None, None]
        mixed = (neighbors.float() * gate[..., None]).sum(dim=2)
        mixed = mixed.to(tokens.dtype).reshape(batch, length, dim)
        tokens = tokens + self.output(mixed)
        tokens = tokens + self.ffn(tokens)
        return tokens.masked_fill(~mask.unsqueeze(-1), 0.0)


def indexwise_max_pool(
    tokens: Tensor,
    lower_mz: Tensor,
    upper_mz: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Halve sorted sparse sequences without materializing absent m/z positions."""

    if tokens.shape[1] % 2:
        tokens = F.pad(tokens, (0, 0, 0, 1))
        lower_mz = F.pad(lower_mz, (0, 1), value=torch.inf)
        upper_mz = F.pad(upper_mz, (0, 1), value=-torch.inf)
        mask = F.pad(mask, (0, 1), value=False)
    batch, length, dim = tokens.shape
    pair_mask = mask.reshape(batch, length // 2, 2)
    pooled_mask = pair_mask.any(dim=-1)
    grouped = tokens.masked_fill(~mask.unsqueeze(-1), -torch.inf)
    grouped = grouped.reshape(batch, length // 2, 2, dim)
    pooled = grouped.amax(dim=2)
    pooled = torch.where(pooled_mask.unsqueeze(-1), pooled, 0.0)
    pooled_lower = lower_mz.reshape(batch, length // 2, 2).amin(dim=2)
    pooled_upper = upper_mz.reshape(batch, length // 2, 2).amax(dim=2)
    pooled_lower = torch.where(pooled_mask, pooled_lower, 0.0)
    pooled_upper = torch.where(pooled_mask, pooled_upper, 0.0)
    return pooled, pooled_lower, pooled_upper, pooled_mask


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
        pooled = scores.new_zeros(segment_count, self.heads, self.head_dim)
        pooled.index_add_(0, segments, weights[..., None] * values.float())
        pooled = pooled.to(values.dtype)
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
    """Distance-aware sparse CNN followed by compressed spectrum attention."""

    def __init__(
        self,
        metadata_dim: int,
        dim: int = 384,
        heads: int = 8,
        peak_conv_stages: int = 2,
        peak_conv_kernel_size: int = 5,
        peak_tau_min: float = 0.005,
        peak_tau_max: float = 2.0,
        peak_cutoff_multiplier: float = 8.0,
        spectrum_layers: int = 2,
        ffn_ratio: float = 2.0,
        dropout: float = 0.1,
        peak_position_dim: int = 32,
        mz_bin_width: float = 0.01,
        mz_upper_bound: float = 5_000.0,
        peak_chunk_size: int = 256,
        peak_chunk_batch: int = 256,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if peak_chunk_size < 1:
            raise ValueError("peak_chunk_size must be positive")
        if peak_chunk_batch < 1:
            raise ValueError("peak_chunk_batch must be positive")
        if peak_conv_stages not in {1, 2}:
            raise ValueError("peak_conv_stages must be one or two")
        self.peak_chunk_size = peak_chunk_size
        self.peak_chunk_batch = peak_chunk_batch
        self.mz_upper_bound = mz_upper_bound
        self.peak_embedding = PositionwiseAffinePeakEmbedding(
            dim,
            position_dim=peak_position_dim,
            mz_bin_width=mz_bin_width,
            mz_upper_bound=mz_upper_bound,
        )
        self.metadata_embedding = nn.Sequential(
            nn.Linear(metadata_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.precursor_embedding = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.peak_conv_blocks = nn.ModuleList(
            [
                ExponentialDistanceConvBlock(
                    dim,
                    heads,
                    kernel_size=peak_conv_kernel_size,
                    tau_min=peak_tau_min * (4**stage),
                    tau_max=peak_tau_max * (4**stage),
                    cutoff_multiplier=peak_cutoff_multiplier,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                )
                for stage in range(peak_conv_stages)
            ]
        )
        self.peak_attention = SegmentAttentionPool(dim, heads)
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

        precursor_mz = torch.nan_to_num(
            precursor_mz, nan=0.0, posinf=self.mz_upper_bound, neginf=0.0
        )
        metadata_tokens = self.metadata_embedding(metadata)
        precursor_feature = torch.log1p(precursor_mz.clamp_min(0.0))
        precursor_feature = precursor_feature / math.log1p(self.mz_upper_bound)
        metadata_tokens = metadata_tokens + self.precursor_embedding(
            precursor_feature.unsqueeze(-1)
        )
        compressed_parts: list[Tensor] = []
        compressed_segment_parts: list[Tensor] = []
        for start in range(0, len(peak_chunks), self.peak_chunk_batch):
            end = min(start + self.peak_chunk_batch, len(peak_chunks))
            chunk_slice = peak_chunks[start:end]
            mask_slice = peak_mask[start:end].bool()
            mapping_slice = chunk_to_spectrum[start:end]
            mz = torch.nan_to_num(
                chunk_slice[..., 0], nan=0.0, posinf=self.mz_upper_bound, neginf=0.0
            )
            intensity = torch.nan_to_num(
                chunk_slice[..., 1], nan=0.0, posinf=1.0, neginf=0.0
            )
            chunk_metadata = metadata_tokens[mapping_slice]
            tokens = self.peak_embedding(mz, intensity) + chunk_metadata[:, None]
            lower_mz = mz
            upper_mz = mz
            token_mask = mask_slice
            for block in self.peak_conv_blocks:
                tokens = block(tokens, lower_mz, upper_mz, token_mask)
                tokens, lower_mz, upper_mz, token_mask = indexwise_max_pool(
                    tokens, lower_mz, upper_mz, token_mask
                )
            valid_tokens = tokens[token_mask]
            token_segments = mapping_slice[:, None].expand(-1, tokens.shape[1])[
                token_mask
            ]
            compressed_parts.append(valid_tokens)
            compressed_segment_parts.append(token_segments)
        compressed_tokens = torch.cat(compressed_parts)
        compressed_segments = torch.cat(compressed_segment_parts)
        spectrum_embeddings = self.peak_attention(
            compressed_tokens, compressed_segments, spectrum_count
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
