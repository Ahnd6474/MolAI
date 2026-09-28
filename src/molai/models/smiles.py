"""SMILES tokenization and conditional sequence decoding."""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch
from torch import Tensor, nn

TOKEN_PATTERN = re.compile(r"(\[[^\]]+\]|Br|Cl|Si|Se|Sn|As|@@?|%\d{2}|.)")


class SmilesTokenizer:
    """Small deterministic tokenizer with an explicit, serializable vocabulary."""

    SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>", "<mask>")

    def __init__(self, vocabulary: Iterable[str]) -> None:
        tokens = list(dict.fromkeys((*self.SPECIAL_TOKENS, *vocabulary)))
        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.id_to_token = tokens

    @classmethod
    def from_smiles(cls, smiles: Iterable[str]) -> SmilesTokenizer:
        vocabulary = sorted({token for value in smiles for token in TOKEN_PATTERN.findall(value)})
        return cls(vocabulary)

    def __len__(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    def encode(self, smiles: str, add_special_tokens: bool = True) -> list[int]:
        tokens = TOKEN_PATTERN.findall(smiles)
        ids = [self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in tokens]
        return [self.bos_id, *ids, self.eos_id] if add_special_tokens else ids

    def decode(self, token_ids: Iterable[int]) -> str:
        output: list[str] = []
        for token_id in token_ids:
            token = self.id_to_token[int(token_id)]
            if token == "<eos>":
                break
            if token not in self.SPECIAL_TOKENS:
                output.append(token)
        return "".join(output)


class SmilesDecoder(nn.Module):
    """Autoregressive SMILES decoder conditioned on a pooled molecular field."""

    def __init__(
        self,
        vocab_size: int,
        molecular_dim: int,
        hidden_dim: int = 384,
        layers: int = 4,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.layers = layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.initial_state = nn.Linear(molecular_dim, layers * hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, molecular_embedding: Tensor, input_ids: Tensor) -> Tensor:
        initial = self.initial_state(molecular_embedding).reshape(
            molecular_embedding.shape[0], self.layers, -1
        )
        initial = initial.permute(1, 0, 2).contiguous()
        decoded, _ = self.gru(self.embedding(input_ids), initial)
        return self.output(decoded)

    @torch.no_grad()
    def greedy_decode(
        self,
        molecular_embedding: Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int = 256,
    ) -> Tensor:
        batch = molecular_embedding.shape[0]
        state = self.initial_state(molecular_embedding).reshape(batch, self.layers, -1)
        state = state.permute(1, 0, 2).contiguous()
        token = torch.full(
            (batch, 1), bos_token_id, device=molecular_embedding.device, dtype=torch.long
        )
        generated: list[Tensor] = []
        finished = torch.zeros(batch, device=token.device, dtype=torch.bool)
        for _ in range(max_length):
            decoded, state = self.gru(self.embedding(token), state)
            token = self.output(decoded[:, -1]).argmax(dim=-1, keepdim=True)
            generated.append(token)
            finished |= token.squeeze(-1).eq(eos_token_id)
            if bool(finished.all()):
                break
        return torch.cat(generated, dim=1)
