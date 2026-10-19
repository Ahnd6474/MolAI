"""Non-autoregressive molecular-field to SMILES translation.

The decoder deliberately receives no previous SMILES tokens.  Learned position
queries first read spatial image tokens with cross-attention, then communicate
with every other output position through unmasked self-attention.  A leading
sample dimension is supported so independent random field branches can be
decoded with shared weights without averaging their images.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    """Choose the largest conventional GroupNorm count that divides channels."""

    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class MolecularFieldTokenEncoder(nn.Module):
    """Turn scalar molecular fields into spatial tokens for cross-attention."""

    def __init__(
        self,
        dim: int = 256,
        grid_size: int = 16,
        in_channels: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim < 16 or grid_size < 1 or in_channels < 1:
            raise ValueError("dim >= 16, grid_size >= 1 and in_channels >= 1 are required")
        first = max(16, dim // 4)
        second = max(32, dim // 2)
        self.grid_size = grid_size
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, first, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(_group_count(first), first),
            nn.GELU(),
            nn.Conv2d(first, second, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(second), second),
            nn.GELU(),
            nn.Conv2d(second, dim, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(dim), dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
        )
        self.image_positions = nn.Parameter(torch.empty(1, grid_size * grid_size, dim))
        self.output_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.image_positions, std=0.02)

    def forward(self, fields: Tensor) -> Tensor:
        if fields.ndim != 4:
            raise ValueError("fields must have shape [batch, channels, height, width]")
        features = self.features(fields)
        tokens = features.flatten(2).transpose(1, 2)
        return self.output_norm(self.dropout(tokens + self.image_positions))


class ParallelSmilesDecoderBlock(nn.Module):
    """Read the image first, then reconcile all SMILES positions bidirectionally."""

    def __init__(
        self,
        dim: int,
        heads: int,
        ff_multiplier: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        hidden = max(dim, int(dim * ff_multiplier))
        self.query_norm = nn.LayerNorm(dim)
        self.image_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, image_tokens: Tensor) -> Tensor:
        query = self.query_norm(queries)
        image = self.image_norm(image_tokens)
        update = self.cross_attention(
            query, image, image, need_weights=False
        )[0]
        queries = queries + self.residual_dropout(update)

        query = self.self_norm(queries)
        # There is intentionally no causal mask: every output position is predicted together.
        update = self.self_attention(query, query, query, need_weights=False)[0]
        queries = queries + self.residual_dropout(update)
        return queries + self.residual_dropout(self.ff(self.ff_norm(queries)))


@dataclass(frozen=True)
class ParallelSmilesOutput:
    """Parallel token and optional sequence-length predictions."""

    logits: Tensor
    length_logits: Tensor


class ParallelFieldToSmiles(nn.Module):
    """Decode every molecular-field branch into canonical SMILES in parallel.

    Inputs may be either ``[B,C,H,W]`` or ``[B,M,C,H,W]``.  The latter is
    flattened internally to one efficient ``B*M`` batch and restored on output;
    no averaging or information exchange occurs between random branches.
    """

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        dim: int = 256,
        heads: int = 8,
        layers: int = 4,
        image_grid_size: int = 16,
        in_channels: int = 1,
        ff_multiplier: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if vocab_size < 2 or max_length < 1 or layers < 1:
            raise ValueError("vocab_size >= 2, max_length >= 1 and layers >= 1 are required")
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.field_encoder = MolecularFieldTokenEncoder(
            dim=dim,
            grid_size=image_grid_size,
            in_channels=in_channels,
            dropout=dropout,
        )
        self.position_queries = nn.Parameter(torch.empty(1, max_length, dim))
        self.blocks = nn.ModuleList(
            ParallelSmilesDecoderBlock(dim, heads, ff_multiplier, dropout)
            for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(dim)
        self.token_head = nn.Linear(dim, vocab_size)
        # Classes 0..max_length make the head convenient for padded mini-batches.
        self.length_head = nn.Linear(dim, max_length + 1)
        nn.init.trunc_normal_(self.position_queries, std=0.02)

    def forward(self, fields: Tensor) -> ParallelSmilesOutput:
        if fields.ndim not in (4, 5):
            raise ValueError("fields must have shape [B,C,H,W] or [B,M,C,H,W]")
        leading_shape = fields.shape[:-3]
        flat_fields = fields.reshape(-1, *fields.shape[-3:])
        image_tokens = self.field_encoder(flat_fields)
        queries = self.position_queries.expand(len(flat_fields), -1, -1)
        for block in self.blocks:
            queries = block(queries, image_tokens)
        queries = self.output_norm(queries)
        logits = self.token_head(queries)
        length_logits = self.length_head(image_tokens.mean(dim=1))
        return ParallelSmilesOutput(
            logits=logits.reshape(*leading_shape, self.max_length, self.vocab_size),
            length_logits=length_logits.reshape(*leading_shape, self.max_length + 1),
        )

    @torch.no_grad()
    def predict_token_ids(
        self,
        fields: Tensor,
        *,
        eos_token_id: int,
        pad_token_id: int,
    ) -> Tensor:
        """Predict all slots once and mask positions following the first EOS."""

        token_ids = self(fields).logits.argmax(dim=-1)
        eos_seen = token_ids.eq(eos_token_id).cumsum(dim=-1).gt(0)
        after_first_eos = F.pad(eos_seen[..., :-1], (1, 0), value=False)
        return token_ids.masked_fill(after_first_eos, pad_token_id)


def parallel_smiles_loss(
    output: ParallelSmilesOutput,
    token_ids: Tensor,
    *,
    pad_token_id: int,
    strip_bos: bool = True,
    length_weight: float = 0.1,
) -> dict[str, Tensor]:
    """Average simultaneous token supervision over every independent branch.

    ``token_ids`` is molecule-level ``[B,T]`` data and is broadcast over a
    possible random-sample axis in the output.  By default it follows
    :class:`SmilesTokenizer` and removes its leading BOS before supervision.
    Padding after EOS is ignored rather than dominating shorter strings.
    """

    if token_ids.ndim != 2 or output.logits.ndim not in (3, 4):
        raise ValueError("token_ids must be [B,T] and logits [B,L,V] or [B,M,L,V]")
    if output.logits.shape[0] != token_ids.shape[0]:
        raise ValueError("token_ids and output batch sizes must match")
    targets = token_ids[:, 1:] if strip_bos else token_ids
    slots = output.logits.shape[-2]
    if targets.shape[1] > slots:
        raise ValueError("target SMILES exceeds the configured maximum output length")
    padded = token_ids.new_full((len(token_ids), slots), pad_token_id)
    padded[:, : targets.shape[1]] = targets

    if output.logits.ndim == 4:
        branches = output.logits.shape[1]
        padded = padded[:, None].expand(-1, branches, -1)
    token_loss = F.cross_entropy(
        output.logits.reshape(-1, output.logits.shape[-1]),
        padded.reshape(-1),
        ignore_index=pad_token_id,
    )

    lengths = targets.ne(pad_token_id).sum(dim=-1).clamp_max(slots)
    if output.length_logits.ndim == 3:
        lengths = lengths[:, None].expand(-1, output.length_logits.shape[1])
    length_loss = F.cross_entropy(
        output.length_logits.reshape(-1, slots + 1), lengths.reshape(-1)
    )
    return {
        "total": token_loss + float(length_weight) * length_loss,
        "token_cross_entropy": token_loss,
        "length_cross_entropy": length_loss,
    }
