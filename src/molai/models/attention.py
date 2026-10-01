"""Full-resolution spatial attention primitives.

The implementation is adapted from the MIT-licensed full-resolution path in
Ahnd6474/Cloud-Matching. The primary path keeps a query at every pixel and
convolutionally compresses only the key/value context.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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


class MultiscaleCvTAttention2d(nn.Module):
    """Native-grid queries attending to convolutionally pooled multiscale K/V."""

    def __init__(
        self,
        dim: int,
        heads: int,
        kernel_sizes: list[int] | tuple[int, ...] = (3, 5, 7),
        output_sizes: list[int] | tuple[int, ...] = (8, 4, 2),
    ) -> None:
        super().__init__()
        kernels = tuple(int(value) for value in kernel_sizes)
        sizes = tuple(int(value) for value in output_sizes)
        if not kernels or len(kernels) != len(sizes):
            raise ValueError("CvT kernel/output lists must be non-empty and equally sized")
        if any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("CvT kernels must be positive odd integers")
        if any(size < 1 for size in sizes):
            raise ValueError("CvT output sizes must be positive")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")

        self.dim = dim
        self.kernel_sizes = kernels
        self.output_sizes = sizes
        self.context_convolutions = nn.ModuleList(
            [
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=dim,
                    bias=False,
                )
                for kernel in kernels
            ]
        )
        for kernel, convolution in zip(self.kernel_sizes, self.context_convolutions, strict=True):
            nn.init.constant_(convolution.weight, 1.0 / (kernel * kernel))
        self.scale_embeddings = nn.Parameter(torch.zeros(len(kernels), dim))
        nn.init.normal_(self.scale_embeddings, std=0.02)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)

    def pooled_token_count(self, height: int, width: int) -> int:
        """Return the K/V sequence length for an input spatial shape."""

        return sum(min(size, height) * min(size, width) for size in self.output_sizes)

    def pool_context(self, context: Tensor) -> Tensor:
        """Apply learned strided depthwise projections and concatenate scales."""

        if context.ndim != 4 or context.shape[-1] != self.dim:
            raise ValueError("context must have shape [B,H,W,D]")
        _, height, width, _ = context.shape
        channels_first = context.permute(0, 3, 1, 2)
        pooled_tokens: list[Tensor] = []
        for index, (convolution, output_size) in enumerate(
            zip(self.context_convolutions, self.output_sizes, strict=True)
        ):
            pooled_height = min(output_size, height)
            pooled_width = min(output_size, width)
            stride_height = max(1, height // pooled_height)
            stride_width = max(1, width // pooled_width)
            features = F.conv2d(
                channels_first,
                convolution.weight,
                bias=None,
                stride=(stride_height, stride_width),
                padding=convolution.padding,
                groups=self.dim,
            )
            pooled = F.adaptive_avg_pool2d(features, (pooled_height, pooled_width))
            tokens = pooled.flatten(2).transpose(1, 2)
            pooled_tokens.append(tokens + self.scale_embeddings[index])
        return torch.cat(pooled_tokens, dim=1)

    def forward(self, query: Tensor, context: Tensor | None = None) -> Tensor:
        context = query if context is None else context
        if query.ndim != 4 or query.shape[-1] != self.dim:
            raise ValueError("query must have shape [B,H,W,D]")
        if context.ndim != 4 or context.shape[0] != query.shape[0]:
            raise ValueError("context must have shape [B,H,W,D]")
        batch, height, width, dim = query.shape
        context_tokens = self.pool_context(context)
        result, _ = self.attention(
            query.reshape(batch, height * width, dim),
            context_tokens,
            context_tokens,
            need_weights=False,
        )
        return result.reshape(batch, height, width, dim)


class FullResolutionCvTEncoder(nn.Module):
    """3x3 full-resolution stem with local refinement and reduced-K/V attention."""

    def __init__(
        self,
        in_channels: int,
        dim: int,
        heads: int,
        kernel_sizes: list[int] | tuple[int, ...] = (3, 5, 7),
        output_sizes: list[int] | tuple[int, ...] = (8, 4, 2),
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        self.local_norm = nn.LayerNorm(dim)
        self.local_refinement = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        self.local_gate = nn.Parameter(torch.tensor(0.1))
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = MultiscaleCvTAttention2d(dim, heads, kernel_sizes, output_sizes)
        self.attention_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, image: Tensor) -> Tensor:
        tokens = self.input_projection(image).permute(0, 2, 3, 1)
        local = self.local_refinement(self.local_norm(tokens).permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )
        tokens = tokens + self.local_gate * local
        return tokens + self.attention_gate * self.attention(self.attention_norm(tokens))


class CvTCrossBlock(nn.Module):
    """Fuse a full-resolution query grid with multiscale pooled context."""

    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_ratio: float = 2.0,
        gate_init: float = 0.5,
        kernel_sizes: list[int] | tuple[int, ...] = (3, 5, 7),
        output_sizes: list[int] | tuple[int, ...] = (8, 4, 2),
    ) -> None:
        super().__init__()
        if not 0.0 < gate_init <= 1.0:
            raise ValueError("gate_init must lie in (0, 1]")
        hidden = max(dim, round(dim * ffn_ratio))
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = MultiscaleCvTAttention2d(dim, heads, kernel_sizes, output_sizes)
        self.gate_projection = nn.Linear(dim, 1)
        gate_logit = 12.0 if gate_init == 1.0 else math.log(gate_init / (1.0 - gate_init))
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.constant_(self.gate_projection.bias, gate_logit)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        update = self.attention(self.query_norm(query), self.context_norm(context))
        query = query + torch.sigmoid(self.gate_projection(update)) * update
        return query + self.ff(query)


class CvTMixerBlock(nn.Module):
    """Full-resolution self-attention with compressed multiscale K/V and FFN."""

    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_ratio: float = 2.0,
        kernel_sizes: list[int] | tuple[int, ...] = (3, 5, 7),
        output_sizes: list[int] | tuple[int, ...] = (8, 4, 2),
    ) -> None:
        super().__init__()
        hidden = max(dim, round(dim * ffn_ratio))
        self.norm = nn.LayerNorm(dim)
        self.attention = MultiscaleCvTAttention2d(dim, heads, kernel_sizes, output_sizes)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.attention(self.norm(tokens))
        return tokens + self.ff(tokens)


class RandomMemoryAttention(nn.Module):
    """Inject one per-sample Gaussian memory into all full-resolution queries."""

    def __init__(
        self,
        dim: int,
        heads: int,
        random_dim: int,
        temperature: float = 0.8,
        gate_init: float = 0.02,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if random_dim < 1 or temperature <= 0.0:
            raise ValueError("random_dim and temperature must be positive")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.random_dim = random_dim
        self.temperature = temperature
        self.query_norm = nn.LayerNorm(dim)
        self.query_projection = nn.Linear(dim, dim, bias=False)
        self.key_projection = nn.Linear(random_dim, dim, bias=False)
        self.value_projection = nn.Linear(random_dim, dim, bias=False)
        self.key_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.value_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.output_projection = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        query: Tensor,
        random_tokens: Tensor,
        spatial_amplitude: Tensor,
    ) -> Tensor:
        if query.ndim != 4 or random_tokens.ndim != 3:
            raise ValueError("query must be [B,H,W,D] and random_tokens [B,K,R]")
        batch, height, width, dim = query.shape
        if (
            dim != self.dim
            or random_tokens.shape[0] != batch
            or random_tokens.shape[-1] != self.random_dim
        ):
            raise ValueError("query/random memory dimensions do not match the module")
        if spatial_amplitude.shape != (batch, height, width):
            raise ValueError("spatial_amplitude must have shape [B,H,W]")

        projected_query = self.query_projection(self.query_norm(query)).reshape(
            batch, height * width, self.heads, self.head_dim
        )
        key = self.key_norm(self.key_projection(random_tokens)).reshape(
            batch, random_tokens.shape[1], self.heads, self.head_dim
        )
        value = self.value_norm(self.value_projection(random_tokens)).reshape(
            batch, random_tokens.shape[1], self.heads, self.head_dim
        )
        attended = F.scaled_dot_product_attention(
            projected_query.permute(0, 2, 1, 3) / self.temperature,
            key.permute(0, 2, 1, 3),
            value.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.permute(0, 2, 1, 3).reshape(batch, height, width, dim)
        update = self.output_projection(attended) * spatial_amplitude[..., None]
        return query + self.gate * update


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


class NoiseTokenCrossBlock(nn.Module):
    """Cross-attend every image token to a short, per-sample noise sequence.

    Query and noise-token batches are already flattened to ``B * M``.  One
    scaled-dot-product-attention call therefore processes every cloud sample
    while keeping samples completely independent from one another.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_ratio: float = 2.0,
        gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        hidden = round(dim * ffn_ratio)
        self.query_norm = nn.LayerNorm(dim)
        self.noise_norm = nn.LayerNorm(dim)
        self.query_projection = nn.Linear(dim, dim)
        self.key_value_projection = nn.Linear(dim, 2 * dim)
        self.output_projection = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.tensor(math.log(gate_init / (1.0 - gate_init))))
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        query: Tensor,
        noise_tokens: Tensor,
        spatial_amplitude: Tensor,
    ) -> Tensor:
        if query.ndim != 4 or noise_tokens.ndim != 3:
            raise ValueError("query must be [B,H,W,D] and noise_tokens must be [B,K,D]")
        batch, height, width, dim = query.shape
        if noise_tokens.shape[0] != batch or noise_tokens.shape[-1] != dim:
            raise ValueError("query and noise token batch/dimension must match")
        if spatial_amplitude.shape != (batch, height, width):
            raise ValueError("spatial_amplitude must have shape [B,H,W]")

        flat_query = self.query_norm(query).reshape(batch, height * width, dim)
        projected_query = self.query_projection(flat_query)
        projected_query = projected_query.reshape(
            batch, height * width, self.heads, self.head_dim
        ).transpose(1, 2)

        key_value = self.key_value_projection(self.noise_norm(noise_tokens))
        key_value = key_value.reshape(
            batch, noise_tokens.shape[1], 2, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        key, value = key_value.unbind(0)
        update = F.scaled_dot_product_attention(
            projected_query,
            key,
            value,
            dropout_p=0.0,
        )
        update = update.transpose(1, 2).reshape(batch, height, width, dim)
        update = self.output_projection(update)
        update = update * spatial_amplitude[..., None]
        query = query + torch.sigmoid(self.gate) * update
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
