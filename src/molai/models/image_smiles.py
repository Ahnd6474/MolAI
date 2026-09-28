"""Compact image-to-SMILES baseline used to evaluate field sufficiency."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from molai.models.smiles import SmilesDecoder


class MolecularFieldEncoder(nn.Module):
    """Encode a full-resolution scalar molecular field without patch tokenization."""

    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 192),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(192, output_dim), nn.LayerNorm(output_dim))

    def forward(self, field: Tensor) -> Tensor:
        return self.projection(self.features(field))


class FieldToSmiles(nn.Module):
    """CNN field encoder followed by an autoregressive SMILES decoder."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        decoder_hidden_dim: int = 256,
        decoder_layers: int = 2,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = MolecularFieldEncoder(embedding_dim)
        self.decoder = SmilesDecoder(
            vocab_size=vocab_size,
            molecular_dim=embedding_dim,
            hidden_dim=decoder_hidden_dim,
            layers=decoder_layers,
            pad_token_id=pad_token_id,
        )

    def forward(self, field: Tensor, input_ids: Tensor) -> Tensor:
        return self.decoder(self.encoder(field), input_ids)

    @torch.no_grad()
    def generate(
        self,
        field: Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int,
    ) -> Tensor:
        return self.decoder.greedy_decode(
            self.encoder(field), bos_token_id, eos_token_id, max_length
        )
