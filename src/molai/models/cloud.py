"""Full-resolution molecular Cloud Matching model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from molai.models.attention import (
    CvTCrossBlock,
    CvTMixerBlock,
    FullResolutionCvTEncoder,
    RandomMemoryAttention,
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
    """Generate molecular fields with full-resolution Q and pooled multiscale K/V."""

    def __init__(
        self,
        field_channels: int = 1,
        condition_dim: int = 384,
        dim: int = 64,
        heads: int = 4,
        condition_cross_depth: int = 2,
        noise_cross_depth: int = 1,
        noise_token_count: int = 64,
        noise_token_dim: int | None = None,
        noise_temperature: float = 0.8,
        noise_gate_init: float = 0.02,
        refine_depth: int = 6,
        window_size: int = 8,
        ffn_ratio: float = 2.0,
        cvt_kernel_sizes: list[int] | tuple[int, ...] = (3, 5, 7),
        cvt_output_sizes: list[int] | tuple[int, ...] = (8, 4, 2),
        condition_gate_init: float = 0.5,
        max_resolution: int = 256,
        noise_energy_min: float = 1e-4,
        noise_energy_init: float = 0.1,
        noise_amplitude_max: float = 8.0,
        zero_mean_output: bool = False,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if dim % heads or dim % 4:
            raise ValueError("dim must be divisible by heads and by four")
        if noise_token_count < 1:
            raise ValueError("noise_token_count must be positive")
        if noise_token_dim is not None and noise_token_dim < 1:
            raise ValueError("noise_token_dim must be positive")
        if noise_cross_depth != 1:
            raise ValueError("the CvT cloud uses exactly one random-memory attention layer")
        if refine_depth < 1 or condition_cross_depth < 1:
            raise ValueError("condition and refinement depths must be positive")
        self.field_channels = field_channels
        self.dim = dim
        self.noise_token_dim = dim if noise_token_dim is None else noise_token_dim
        self.noise_energy_min = noise_energy_min
        self.noise_amplitude_max = noise_amplitude_max
        self.noise_token_count = noise_token_count
        self.zero_mean_output = zero_mean_output
        self.gradient_checkpointing = gradient_checkpointing

        self.field_encoder = FullResolutionCvTEncoder(
            field_channels,
            dim,
            heads,
            cvt_kernel_sizes,
            cvt_output_sizes,
        )
        self.condition_plane = AxialConditionPlane(
            condition_dim, dim, max_resolution=max_resolution
        )
        self.condition_blocks = nn.ModuleList(
            [
                CvTCrossBlock(
                    dim,
                    heads,
                    ffn_ratio,
                    gate_init=condition_gate_init,
                    kernel_sizes=cvt_kernel_sizes,
                    output_sizes=cvt_output_sizes,
                )
                for _ in range(condition_cross_depth)
            ]
        )
        self.noise_attention = RandomMemoryAttention(
            dim,
            heads,
            self.noise_token_dim,
            temperature=noise_temperature,
            gate_init=noise_gate_init,
        )
        self.refine_blocks = nn.ModuleList(
            [
                CvTMixerBlock(
                    dim,
                    heads,
                    ffn_ratio,
                    kernel_sizes=cvt_kernel_sizes,
                    output_sizes=cvt_output_sizes,
                )
                for _ in range(refine_depth)
            ]
        )
        self.energy_norm = nn.LayerNorm(dim)
        self.energy_head = nn.Linear(dim, 1)
        initial_amplitude = math.sqrt(max(noise_energy_init - noise_energy_min, 1e-8))
        nn.init.zeros_(self.energy_head.weight)
        nn.init.constant_(self.energy_head.bias, math.log(math.expm1(initial_amplitude)))
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
        tokens = self.field_encoder(current)
        position = sinusoidal_2d_position(height, width, self.dim, tokens.device, tokens.dtype)[
            None
        ]
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
                self.noise_token_dim,
                device=encoded.device,
                dtype=encoded.dtype,
            )
        else:
            if noise.ndim != 4 or noise.shape[:3] != (
                batch,
                samples,
                self.noise_token_count,
            ):
                raise ValueError("noise must have shape [B,M,K,R], where K is noise_token_count")
            random_noise = noise.to(device=encoded.device, dtype=encoded.dtype)
            if random_noise.shape[-1] != self.noise_token_dim:
                raise ValueError("noise channels must match noise_token_dim")
        return random_noise

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
            batch * samples, self.noise_token_count, self.noise_token_dim
        )
        spatial_amplitude = (energy / self.dim).sqrt()
        spatial_amplitude = spatial_amplitude[:, None].expand(-1, samples, -1, -1)
        spatial_amplitude = spatial_amplitude.reshape(batch * samples, height, width)
        tokens = self._run(self.noise_attention, tokens, noise_context, spatial_amplitude)
        for block in self.refine_blocks:
            tokens = self._run(block, tokens)

        residual = self.output_head(self.output_norm(tokens))
        residual = residual.permute(0, 3, 1, 2)
        expanded_current = current[:, None].expand(-1, samples, -1, -1, -1)
        fields = expanded_current.reshape(batch * samples, *current.shape[1:]) + residual
        if self.zero_mean_output:
            spatial_mean = fields.float().mean(dim=(-2, -1), keepdim=True)
            fields = fields - spatial_mean.to(fields.dtype)
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
