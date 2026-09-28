"""Full-resolution axial and local attention primitives.

The implementation is adapted from the MIT-licensed full-resolution path in
Ahnd6474/Cloud-Matching. Spatial attention is always factorized into local,
row-axial, or column-axial operations; no quadratic global image attention is used.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def sinusoidal_2d_position(
    height: int,
    width: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return a deterministic `[H,W,D]` two-dimensional position encoding."""

    if dim % 4:
        raise ValueError("attention dimension must be divisible by four")
    quarter = dim // 4
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(quarter, device=device, dtype=torch.float32)
        / max(quarter - 1, 1)
    )
    rows = torch.arange(height, device=device, dtype=torch.float32)[:, None] * frequency
    columns = torch.arange(width, device=device, dtype=torch.float32)[:, None] * frequency
    row_encoding = torch.cat((rows.sin(), rows.cos()), dim=-1)
    column_encoding = torch.cat((columns.sin(), columns.cos()), dim=-1)
    position = torch.cat(
        (
            row_encoding[:, None, :].expand(-1, width, -1),
            column_encoding[None, :, :].expand(height, -1, -1),
        ),
        dim=-1,
    )
    return position.to(dtype=dtype)


def _partition_windows(
    tokens: Tensor,
    window_size: int,
    shift: int,
) -> tuple[Tensor, Tensor | None, tuple[int, int, int, int, int]]:
    batch, height, width, dim = tokens.shape
    if shift:
        tokens = torch.roll(tokens, shifts=(-shift, -shift), dims=(1, 2))
    padded_height = math.ceil(height / window_size) * window_size
    padded_width = math.ceil(width / window_size) * window_size
    pad_height = padded_height - height
    pad_width = padded_width - width
    if pad_height or pad_width:
        tokens = torch.nn.functional.pad(tokens, (0, 0, 0, pad_width, 0, pad_height))
    windows = (
        tokens.reshape(
            batch,
            padded_height // window_size,
            window_size,
            padded_width // window_size,
            window_size,
            dim,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, window_size * window_size, dim)
    )
    padding_mask = None
    if pad_height or pad_width:
        valid = torch.ones((batch, height, width), device=tokens.device, dtype=torch.bool)
        valid = torch.nn.functional.pad(valid, (0, pad_width, 0, pad_height), value=False)
        padding_mask = (
            valid.reshape(
                batch,
                padded_height // window_size,
                window_size,
                padded_width // window_size,
                window_size,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(-1, window_size * window_size)
            .logical_not()
        )
    return windows, padding_mask, (height, width, padded_height, padded_width, shift)


def _reverse_windows(
    windows: Tensor,
    window_size: int,
    metadata: tuple[int, int, int, int, int],
    batch: int,
) -> Tensor:
    height, width, padded_height, padded_width, shift = metadata
    dim = windows.shape[-1]
    tokens = (
        windows.reshape(
            batch,
            padded_height // window_size,
            padded_width // window_size,
            window_size,
            window_size,
            dim,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, padded_height, padded_width, dim)
    )
    tokens = tokens[:, :height, :width]
    if shift:
        tokens = torch.roll(tokens, shifts=(shift, shift), dims=(1, 2))
    return tokens


class FactorizedAttention2d(nn.Module):
    """Local-window or axial self/cross attention on a full-resolution grid."""

    def __init__(
        self,
        dim: int,
        heads: int,
        kind: str,
        window_size: int = 8,
        shifted: bool = False,
    ) -> None:
        super().__init__()
        if kind not in {"local", "row", "column"}:
            raise ValueError("kind must be local, row, or column")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.kind = kind
        self.window_size = window_size
        self.shift = window_size // 2 if shifted and window_size > 1 else 0
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, query: Tensor, context: Tensor | None = None) -> Tensor:
        self_attention = context is None
        context = query if context is None else context
        if query.ndim != 4 or query.shape != context.shape:
            raise ValueError("query and context must share [B,H,W,D] shape")
        batch, height, width, dim = query.shape
        if self.kind == "row":
            q = query.reshape(batch * height, width, dim)
            kv = context.reshape(batch * height, width, dim)
            result, _ = self.attention(q, kv, kv, need_weights=False)
            return result.reshape(batch, height, width, dim)
        if self.kind == "column":
            q = query.permute(0, 2, 1, 3).reshape(batch * width, height, dim)
            kv = context.permute(0, 2, 1, 3).reshape(batch * width, height, dim)
            result, _ = self.attention(q, kv, kv, need_weights=False)
            return result.reshape(batch, width, height, dim).permute(0, 2, 1, 3)

        q_windows, padding_mask, metadata = _partition_windows(query, self.window_size, self.shift)
        if self_attention:
            kv_windows = q_windows
            context_metadata = metadata
        else:
            kv_windows, _, context_metadata = _partition_windows(
                context, self.window_size, self.shift
            )
        if metadata != context_metadata:
            raise RuntimeError("query and context window layouts differ")
        result, _ = self.attention(
            q_windows,
            kv_windows,
            kv_windows,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return _reverse_windows(result, self.window_size, metadata, batch)


class AxialLocalCrossBlock(nn.Module):
    """Fuse two aligned grids using only local and axial cross attention."""

    def __init__(
        self,
        dim: int,
        heads: int,
        window_size: int,
        ffn_ratio: float = 2.0,
        shifted: bool = False,
        gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        hidden = round(dim * ffn_ratio)
        self.query_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.context_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.attentions = nn.ModuleList(
            [
                FactorizedAttention2d(dim, heads, "local", window_size, shifted),
                FactorizedAttention2d(dim, heads, "row"),
                FactorizedAttention2d(dim, heads, "column"),
            ]
        )
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.gates = nn.Parameter(torch.full((3,), gate_logit))
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        for index, (query_norm, context_norm, attention) in enumerate(
            zip(self.query_norms, self.context_norms, self.attentions, strict=True)
        ):
            update = attention(query_norm(query), context_norm(context))
            query = query + torch.sigmoid(self.gates[index]) * update
        return query + self.ff(query)


class AxialLocalMixerBlock(nn.Module):
    """One local, row-axial, or column-axial refinement block."""

    def __init__(
        self,
        dim: int,
        heads: int,
        kind: str,
        window_size: int,
        ffn_ratio: float = 2.0,
        shifted: bool = False,
    ) -> None:
        super().__init__()
        hidden = round(dim * ffn_ratio)
        self.norm = nn.LayerNorm(dim)
        self.attention = FactorizedAttention2d(
            dim, heads, kind, window_size=window_size, shifted=shifted
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.attention(self.norm(tokens))
        return tokens + self.ff(tokens)
