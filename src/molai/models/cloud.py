"""Full-resolution molecular Cloud Matching model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from molai.models.attention import (
    AxialLocalCrossBlock,
    AxialLocalMixerBlock,
    NoiseTokenCrossBlock,
    sinusoidal_2d_position,
)
from molai.models.condition import AxialConditionPlane
from molai.models.smiles import SmilesDecoder


@dataclass(slots=True)
class MolecularCloudOutput:
    """Generated empirical field cloud and optional SMILES logits."""

    fields: Tensor
    spatial_noise_energy: Tensor
    molecular_embeddings: Tensor
    smiles_logits: Tensor | None = None


class MolecularFieldCloud(nn.Module):
    """Generate full-resolution molecular fields with axial/local attention only."""

    def __init__(
        self,
        field_channels: int = 1,
        condition_dim: int = 384,
        dim: int = 64,
        heads: int = 4,
        condition_cross_depth: int = 2,
        noise_cross_depth: int = 2,
        noise_token_count: int = 64,
        refine_depth: int = 8,
        window_size: int = 8,
        ffn_ratio: float = 2.0,
        max_resolution: int = 256,
        max_residual: float = 2.0,
        noise_energy_min: float = 1e-4,
        noise_energy_init: float = 0.1,
        noise_amplitude_max: float = 8.0,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if dim % heads or dim % 4:
            raise ValueError("dim must be divisible by heads and by four")
        if noise_token_count < 1:
            raise ValueError("noise_token_count must be positive")
        self.field_channels = field_channels
        self.dim = dim
        self.max_residual = max_residual
        self.noise_energy_min = noise_energy_min
        self.noise_amplitude_max = noise_amplitude_max
        self.noise_token_count = noise_token_count
        self.gradient_checkpointing = gradient_checkpointing

        self.field_embed = nn.Linear(field_channels, dim)
        self.condition_plane = AxialConditionPlane(
            condition_dim, dim, max_resolution=max_resolution
        )
        self.condition_blocks = nn.ModuleList(
            [
                AxialLocalCrossBlock(
                    dim,
                    heads,
                    window_size,
                    ffn_ratio,
                    shifted=bool(index % 2),
                    gate_init=0.5,
                )
                for index in range(condition_cross_depth)
            ]
        )
        self.noise_blocks = nn.ModuleList(
            [
                NoiseTokenCrossBlock(
                    dim,
                    heads,
                    ffn_ratio,
                    gate_init=0.1,
                )
                for _ in range(noise_cross_depth)
            ]
        )
        pattern = ("local", "row", "local", "column")
        self.refine_blocks = nn.ModuleList(
            [
                AxialLocalMixerBlock(
                    dim,
                    heads,
                    pattern[index % len(pattern)],
                    window_size,
                    ffn_ratio,
                    shifted=index % len(pattern) == 2,
                )
                for index in range(refine_depth)
            ]
        )
        self.energy_norm = nn.LayerNorm(dim)
        self.energy_head = nn.Linear(dim, 1)
        initial_amplitude = math.sqrt(max(noise_energy_init - noise_energy_min, 1e-8))
        nn.init.zeros_(self.energy_head.weight)
        nn.init.constant_(self.energy_head.bias, math.log(math.expm1(initial_amplitude)))
        self.noise_projection = nn.Linear(field_channels, dim, bias=False)
        self.noise_token_basis = nn.Parameter(torch.empty(noise_token_count, dim))
        nn.init.normal_(self.noise_token_basis, std=0.02)
        self.output_norm = nn.LayerNorm(dim)
        self.output_head = nn.Linear(dim, field_channels)
        self.readout = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, condition_dim))
        nn.init.normal_(self.output_head.weight, std=1e-3)
        nn.init.zeros_(self.output_head.bias)

    def _run(self, module: nn.Module, *args: Tensor) -> Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def encode_condition(self, current: Tensor, condition: Tensor) -> Tensor:
        if current.ndim != 4 or current.shape[1] != self.field_channels:
            raise ValueError("current must have shape [B,C,H,W]")
        _, _, height, width = current.shape
        tokens = self.field_embed(current.permute(0, 2, 3, 1))
        position = sinusoidal_2d_position(
            height, width, self.dim, tokens.device, tokens.dtype
        )[None]
        tokens = tokens + position
        context = self.condition_plane(condition, height, width) + position
        for block in self.condition_blocks:
            tokens = self._run(block, tokens, context)
        return tokens

    def spatial_noise_energy(self, encoded: Tensor) -> Tensor:
        raw = self.energy_head(self.energy_norm(encoded))
        amplitude = torch.nn.functional.softplus(raw.float()).clamp_max(self.noise_amplitude_max)
        return (self.noise_energy_min + amplitude.square()).to(encoded.dtype).squeeze(-1)

    def _prepare_noise(
        self,
        encoded: Tensor,
        samples: int,
        noise: Tensor | None,
    ) -> Tensor:
        batch = encoded.shape[0]
        if noise is None:
            random_noise = torch.randn(
                batch,
                samples,
                self.noise_token_count,
                self.dim,
                device=encoded.device,
                dtype=encoded.dtype,
            )
        else:
            if noise.ndim != 4 or noise.shape[:3] != (
                batch,
                samples,
                self.noise_token_count,
            ):
                raise ValueError(
                    "noise must have shape [B,M,K,D] or [B,M,K,C], "
                    "where K is noise_token_count"
                )
            random_noise = noise.to(device=encoded.device, dtype=encoded.dtype)
            if random_noise.shape[-1] == self.field_channels:
                random_noise = self.noise_projection(random_noise)
            elif random_noise.shape[-1] != self.dim:
                raise ValueError("noise channels must match field_channels or model dim")
        return random_noise + self.noise_token_basis.to(dtype=encoded.dtype)[None, None]

    def forward(
        self,
        current: Tensor,
        condition: Tensor,
        samples: int = 4,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        encoded = self.encode_condition(current, condition)
        energy = self.spatial_noise_energy(encoded)
        noise_tokens = self._prepare_noise(encoded, samples, noise)

        batch, height, width, _ = encoded.shape
        tokens = encoded[:, None].expand(-1, samples, -1, -1, -1)
        tokens = tokens.reshape(batch * samples, height, width, self.dim)
        noise_context = noise_tokens.reshape(
            batch * samples, self.noise_token_count, self.dim
        )
        spatial_amplitude = (energy / self.dim).sqrt()
        spatial_amplitude = spatial_amplitude[:, None].expand(-1, samples, -1, -1)
        spatial_amplitude = spatial_amplitude.reshape(batch * samples, height, width)
        for block in self.noise_blocks:
            tokens = self._run(block, tokens, noise_context, spatial_amplitude)
        for block in self.refine_blocks:
            tokens = self._run(block, tokens)

        residual = self.max_residual * torch.tanh(self.output_head(self.output_norm(tokens)))
        residual = residual.permute(0, 3, 1, 2)
        expanded_current = current[:, None].expand(-1, samples, -1, -1, -1)
        fields = expanded_current.reshape(batch * samples, *current.shape[1:]) + residual
        molecular_embeddings = self.readout(tokens.mean(dim=(1, 2)))
        return (
            fields.reshape(batch, samples, *current.shape[1:]),
            energy,
            molecular_embeddings.reshape(batch, samples, -1),
        )


class MolecularCloudModel(nn.Module):
    """Cloud generator with an optional autoregressive SMILES head."""

    def __init__(
        self,
        smiles_vocab_size: int,
        smiles_pad_token_id: int,
        condition_dim: int = 384,
        **cloud_kwargs: object,
    ) -> None:
        super().__init__()
        self.cloud = MolecularFieldCloud(condition_dim=condition_dim, **cloud_kwargs)
        self.smiles_decoder = SmilesDecoder(
            smiles_vocab_size,
            molecular_dim=condition_dim,
            hidden_dim=condition_dim,
            pad_token_id=smiles_pad_token_id,
        )

    def forward(
        self,
        current: Tensor,
        condition: Tensor,
        samples: int = 4,
        noise: Tensor | None = None,
        smiles_input_ids: Tensor | None = None,
    ) -> MolecularCloudOutput:
        fields, energy, embeddings = self.cloud(current, condition, samples, noise)
        logits = None
        if smiles_input_ids is not None:
            if smiles_input_ids.ndim == 2:
                smiles_input_ids = smiles_input_ids[:, None].expand(-1, samples, -1)
            flat_ids = smiles_input_ids.reshape(current.shape[0] * samples, -1)
            logits = self.smiles_decoder(embeddings.flatten(0, 1), flat_ids)
            logits = logits.reshape(current.shape[0], samples, *logits.shape[1:])
        return MolecularCloudOutput(fields, energy, embeddings, logits)
