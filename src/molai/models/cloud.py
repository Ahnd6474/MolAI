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
    MultiscaleCvTAttention2d,
    RandomMemoryAttention,
    TokenCrossBlock2d,
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


@dataclass(slots=True)
class HiddenRolloutOutput:
    """Decoded fields and diagnostics from a weight-tied hidden rollout."""

    fields: Tensor
    anchor_reconstruction: Tensor
    final_hidden: Tensor
    gate_means: Tensor
    update_rms: Tensor
    spatial_noise_energy: Tensor


@dataclass(slots=True)
class AbsoluteCloudOutput:
    """Absolute one-step field cloud and encoder-anchor diagnostics."""

    fields: Tensor
    anchor_reconstruction: Tensor | None
    gate_means: Tensor
    update_rms: Tensor
    spatial_noise_energy: Tensor


@dataclass(slots=True)
class AbsoluteHiddenRolloutOutput:
    """Final field cloud from a single-path hidden rollout with late branching."""

    fields: Tensor
    final_hidden: Tensor
    intermediate_gate_means: Tensor
    intermediate_update_rms: Tensor
    final_gate_means: Tensor
    final_update_rms: Tensor
    spatial_noise_energy: Tensor


@dataclass(slots=True)
class AbsoluteTrajectoryRolloutOutput:
    """Decoded fields and diagnostics for persistent stochastic trajectories."""

    fields: Tensor
    final_hidden: Tensor
    hidden_consistency_mse: Tensor | None
    gate_means: Tensor
    update_rms: Tensor
    spatial_noise_energy: Tensor


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
        cvt_grid_sizes: list[int] | tuple[int, ...] = (8, 4, 2),
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
            cvt_grid_sizes,
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
                    grid_sizes=cvt_grid_sizes,
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
                    grid_sizes=cvt_grid_sizes,
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


class EncoderAnchoredHiddenUpdate(nn.Module):
    """Let an encoder state query a model state, then GLU-gate its residual update."""

    def __init__(
        self,
        dim: int,
        heads: int,
        max_level: int = 64,
        gate_init: float = 0.02,
        kernel_sizes: list[int] | tuple[int, ...] = (3, 5, 7),
        grid_sizes: list[int] | tuple[int, ...] = (32, 16, 8),
    ) -> None:
        super().__init__()
        if max_level < 1:
            raise ValueError("max_level must be positive")
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must be between zero and one")
        self.max_level = max_level
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = MultiscaleCvTAttention2d(
            dim, heads, kernel_sizes=kernel_sizes, grid_sizes=grid_sizes
        )
        self.candidate_projection = nn.Linear(dim, dim, bias=False)
        self.candidate_norm = nn.RMSNorm(dim)
        self.gate_projection = nn.Linear(dim, dim, bias=False)
        self.level_embedding = nn.Embedding(max_level + 1, dim)
        self.level_gate = nn.Linear(dim, dim, bias=False)
        self.level_amplitude = nn.Embedding(max_level + 1, 1)
        self.gate_bias = nn.Parameter(
            torch.full((dim,), math.log(gate_init / (1.0 - gate_init)))
        )

        nn.init.eye_(self.candidate_projection.weight)
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.zeros_(self.level_embedding.weight)
        nn.init.zeros_(self.level_gate.weight)
        nn.init.constant_(self.level_amplitude.weight, math.log(math.expm1(1.0)))

    def forward(
        self, encoder_hidden: Tensor, model_hidden: Tensor, levels: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        if model_hidden.shape != encoder_hidden.shape or model_hidden.ndim != 4:
            raise ValueError(
                "model_hidden and encoder_hidden must have the same [B,H,W,D] shape"
            )
        if levels is not None:
            if levels.shape != (model_hidden.shape[0],):
                raise ValueError("levels must have shape [B]")
            if not torch.compiler.is_compiling() and torch.any(
                (levels < 0) | (levels > self.max_level)
            ):
                raise ValueError(f"levels must be in [0, {self.max_level}]")

        cross_update = self.attention(
            self.query_norm(encoder_hidden), self.context_norm(model_hidden)
        )
        candidate = self.candidate_norm(self.candidate_projection(cross_update))
        gate_logits = self.gate_projection(cross_update) + self.gate_bias
        if levels is not None:
            level = self.level_embedding(levels.long())[:, None, None]
            gate_logits = gate_logits + self.level_gate(level)
        gate = torch.sigmoid(gate_logits)
        if levels is None:
            update = gate * candidate
        else:
            amplitude = torch.nn.functional.softplus(
                self.level_amplitude(levels.long()).float()
            ).to(candidate.dtype)
            update = amplitude[:, None, None] * gate * candidate
        return encoder_hidden + update, gate, update


class AbsoluteFieldHead(nn.Module):
    """Decode an absolute field while preserving raw encoder amplitudes.

    The direct projection intentionally bypasses normalization.  This makes it
    possible to train ``D(E(x)) = x``; the normalized nonlinear branch then adds
    the spatial correction needed for encoder-anchored rollout states.
    """

    def __init__(self, dim: int, field_channels: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = max(dim // 2, field_channels) if hidden_dim is None else hidden_dim
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.direct_projection = nn.Conv2d(dim, field_channels, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.input_projection = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1)
        self.spatial_mixer = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.output_projection = nn.Conv2d(hidden_dim, field_channels, kernel_size=1)
        nn.init.normal_(self.direct_projection.weight, std=1e-3)
        nn.init.zeros_(self.direct_projection.bias)
        nn.init.normal_(self.output_projection.weight, std=1e-3)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.ndim != 4:
            raise ValueError("hidden must have shape [B,H,W,D]")
        channels_first = hidden.permute(0, 3, 1, 2)
        direct = self.direct_projection(channels_first)
        normalized = self.norm(hidden).permute(0, 3, 1, 2)
        value = torch.nn.functional.gelu(self.input_projection(normalized))
        value = torch.nn.functional.gelu(value + self.spatial_mixer(value))
        return direct + self.output_projection(value)


class AbsoluteMolecularFieldCloud(nn.Module):
    """Generate absolute next-step images through an encoder-anchored backbone."""

    def __init__(
        self,
        *,
        field_channels: int = 1,
        condition_dim: int = 384,
        dim: int = 64,
        heads: int = 4,
        condition_cross_depth: int = 2,
        condition_gate_init: float = 0.5,
        noise_token_count: int = 64,
        noise_token_dim: int = 64,
        noise_temperature: float = 0.8,
        noise_gate_init: float = 0.02,
        refine_depth: int = 6,
        ffn_ratio: float = 2.0,
        cvt_kernel_sizes: list[int] | tuple[int, ...] = (3, 5, 7),
        cvt_grid_sizes: list[int] | tuple[int, ...] = (32, 16, 8),
        max_level: int = 64,
        anchor_gate_init: float = 0.02,
        decoder_dim: int = 64,
        noise_energy_min: float = 1e-4,
        noise_energy_init: float = 0.1,
        noise_amplitude_max: float = 8.0,
        zero_mean_output: bool = True,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.field_channels = field_channels
        self.dim = dim
        self.noise_token_count = noise_token_count
        self.noise_token_dim = noise_token_dim
        self.noise_energy_min = noise_energy_min
        self.noise_amplitude_max = noise_amplitude_max
        self.zero_mean_output = zero_mean_output
        self.gradient_checkpointing = gradient_checkpointing
        self.max_level = max_level

        self.field_encoder = FullResolutionCvTEncoder(
            field_channels, dim, heads, cvt_kernel_sizes, cvt_grid_sizes
        )
        self.level_embedding = nn.Embedding(max_level + 1, dim)
        self.condition_blocks = nn.ModuleList(
            [
                TokenCrossBlock2d(
                    dim,
                    condition_dim,
                    heads,
                    ffn_ratio,
                    gate_init=condition_gate_init,
                )
                for _ in range(condition_cross_depth)
            ]
        )
        self.noise_attention = RandomMemoryAttention(
            dim,
            heads,
            noise_token_dim,
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
                    grid_sizes=cvt_grid_sizes,
                )
                for _ in range(refine_depth)
            ]
        )
        self.energy_norm = nn.LayerNorm(dim)
        self.energy_head = nn.Linear(dim, 1)
        initial_amplitude = math.sqrt(max(noise_energy_init - noise_energy_min, 1e-8))
        nn.init.zeros_(self.energy_head.weight)
        nn.init.constant_(self.energy_head.bias, math.log(math.expm1(initial_amplitude)))
        self.state_update = EncoderAnchoredHiddenUpdate(
            dim,
            heads,
            max_level=max_level,
            gate_init=anchor_gate_init,
            kernel_sizes=cvt_kernel_sizes,
            grid_sizes=cvt_grid_sizes,
        )
        self.output_head = AbsoluteFieldHead(dim, field_channels, hidden_dim=decoder_dim)

    def _run(self, module: nn.Module, *args: Tensor) -> Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def encode_anchor(self, field: Tensor) -> Tensor:
        if field.ndim != 4 or field.shape[1] != self.field_channels:
            raise ValueError("field must have shape [B,C,H,W]")
        _, _, height, width = field.shape
        hidden = self.field_encoder(field)
        position = sinusoidal_2d_position(
            height, width, self.dim, hidden.device, hidden.dtype
        )[None]
        return hidden + position

    def decode_absolute(self, hidden: Tensor) -> Tensor:
        field = self.output_head(hidden)
        if self.zero_mean_output:
            field = field - field.float().mean(dim=(-2, -1), keepdim=True).to(field.dtype)
        return field

    def spatial_noise_energy(self, hidden: Tensor) -> Tensor:
        raw = self.energy_head(self.energy_norm(hidden))
        amplitude = torch.nn.functional.softplus(raw.float()).clamp_max(
            self.noise_amplitude_max
        )
        return (self.noise_energy_min + amplitude.square()).to(hidden.dtype).squeeze(-1)

    def forward(
        self,
        current: Tensor,
        condition_tokens: Tensor,
        levels: Tensor,
        *,
        samples: int = 4,
        noise: Tensor | None = None,
        condition_mask: Tensor | None = None,
        return_anchor_reconstruction: bool = False,
    ) -> AbsoluteCloudOutput:
        if condition_tokens.ndim != 3 or condition_tokens.shape[0] != current.shape[0]:
            raise ValueError("condition_tokens must have shape [B,K,C]")
        if levels.shape != (current.shape[0],):
            raise ValueError("levels must have shape [B]")
        if samples < 1:
            raise ValueError("samples must be positive")
        if torch.any((levels < 0) | (levels > self.max_level)):
            raise ValueError(f"levels must lie in [0, {self.max_level}]")

        batch, _, height, width = current.shape
        anchor = self.encode_anchor(current)
        anchor_reconstruction = (
            self.decode_absolute(anchor) if return_anchor_reconstruction else None
        )
        proposed = anchor + self.level_embedding(levels.long())[:, None, None]
        for block in self.condition_blocks:
            proposed = self._run(block, proposed, condition_tokens, condition_mask)
        energy = self.spatial_noise_energy(proposed)

        expected_noise = (
            batch,
            samples,
            self.noise_token_count,
            self.noise_token_dim,
        )
        if noise is None:
            noise = torch.randn(*expected_noise, device=current.device, dtype=current.dtype)
        elif noise.shape != expected_noise:
            raise ValueError("noise must have shape [B,M,K,R]")
        else:
            noise = noise.to(device=current.device, dtype=current.dtype)

        proposed = proposed[:, None].expand(-1, samples, -1, -1, -1)
        proposed = proposed.reshape(batch * samples, height, width, self.dim)
        expanded_anchor = anchor[:, None].expand(-1, samples, -1, -1, -1)
        expanded_anchor = expanded_anchor.reshape(batch * samples, height, width, self.dim)
        expanded_levels = levels[:, None].expand(-1, samples).reshape(-1)
        expanded_energy = energy[:, None].expand(-1, samples, -1, -1)
        expanded_energy = expanded_energy.reshape(batch * samples, height, width)
        noise_context = noise.reshape(
            batch * samples, self.noise_token_count, self.noise_token_dim
        )
        spatial_amplitude = (expanded_energy / self.dim).sqrt()
        proposed = self._run(
            self.noise_attention, proposed, noise_context, spatial_amplitude
        )
        for block in self.refine_blocks:
            proposed = self._run(block, proposed)
        hidden, gate, update = self.state_update(
            expanded_anchor, proposed, expanded_levels
        )
        fields = self.decode_absolute(hidden)
        return AbsoluteCloudOutput(
            fields=fields.reshape(batch, samples, *fields.shape[1:]),
            anchor_reconstruction=anchor_reconstruction,
            gate_means=gate.float().mean(dim=(1, 2, 3)).reshape(batch, samples),
            update_rms=update.float()
            .square()
            .mean(dim=(1, 2, 3))
            .sqrt()
            .reshape(batch, samples),
            spatial_noise_energy=energy,
        )


class AbsoluteHiddenRolloutCloud(nn.Module):
    """Run an absolute Cloud as cumulative shared residual steps.

    The input field is encoded once into the evolving ``enc`` state.  At every
    step the backbone produces ``h = Model(enc)``; the current encoder state
    queries that model state and receives a GLU-gated residual update.  Only the
    final step expands into the empirical output cloud used by the loss.
    """

    def __init__(
        self,
        cloud: AbsoluteMolecularFieldCloud,
        *,
        intermediate_refine_depth: int | None = None,
        use_level_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.cloud = cloud
        available = len(cloud.refine_blocks)
        if intermediate_refine_depth is None:
            intermediate_refine_depth = available
        if not 0 <= intermediate_refine_depth <= available:
            raise ValueError(
                "intermediate_refine_depth must lie between zero and refine_depth"
            )
        self.intermediate_refine_depth = intermediate_refine_depth
        self.use_level_conditioning = use_level_conditioning
        if not use_level_conditioning:
            for module in (
                cloud.level_embedding,
                cloud.state_update.level_embedding,
                cloud.state_update.level_gate,
                cloud.state_update.level_amplitude,
            ):
                module.requires_grad_(False)

    def _normalize_levels(
        self, levels: Tensor, batch: int, device: torch.device
    ) -> Tensor:
        levels = levels.to(device=device, dtype=torch.long)
        if levels.ndim == 1:
            levels = levels[None].expand(batch, -1)
        if levels.ndim != 2 or levels.shape[0] != batch:
            raise ValueError("levels must have shape [T] or [B,T]")
        if levels.shape[1] < 1:
            raise ValueError("at least one rollout level is required")
        if not torch.compiler.is_compiling() and torch.any(
            (levels < 0) | (levels > self.cloud.max_level)
        ):
            raise ValueError(f"levels must lie in [0, {self.cloud.max_level}]")
        return levels

    def _condition_and_refine(
        self,
        hidden: Tensor,
        condition_tokens: Tensor,
        condition_mask: Tensor | None,
        levels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        proposed = hidden
        if self.use_level_conditioning:
            proposed = proposed + self.cloud.level_embedding(levels)[:, None, None]
        for block in self.cloud.condition_blocks:
            proposed = self.cloud._run(
                block, proposed, condition_tokens, condition_mask
            )
        energy = self.cloud.spatial_noise_energy(proposed)
        return proposed, energy

    def forward(
        self,
        initial: Tensor,
        condition_tokens: Tensor,
        levels: Tensor,
        *,
        final_samples: int = 5,
        intermediate_noise: Tensor | None = None,
        final_noise: Tensor | None = None,
        condition_mask: Tensor | None = None,
    ) -> AbsoluteHiddenRolloutOutput:
        if initial.ndim != 4 or initial.shape[1] != self.cloud.field_channels:
            raise ValueError("initial must have shape [B,C,H,W]")
        if condition_tokens.ndim != 3 or condition_tokens.shape[0] != initial.shape[0]:
            raise ValueError("condition_tokens must have shape [B,K,C]")
        if final_samples < 2:
            raise ValueError("final_samples must be at least two")
        if condition_mask is not None and condition_mask.shape != condition_tokens.shape[:2]:
            raise ValueError("condition_mask must have shape [B,K]")

        batch, _, height, width = initial.shape
        levels = self._normalize_levels(levels, batch, initial.device)
        steps = levels.shape[1]
        intermediate_steps = steps - 1
        intermediate_shape = (
            batch,
            intermediate_steps,
            self.cloud.noise_token_count,
            self.cloud.noise_token_dim,
        )
        if intermediate_noise is None:
            intermediate_noise = torch.randn(
                *intermediate_shape, device=initial.device, dtype=initial.dtype
            )
        elif intermediate_noise.shape != intermediate_shape:
            raise ValueError("intermediate_noise must have shape [B,T-1,K,R]")
        else:
            intermediate_noise = intermediate_noise.to(
                device=initial.device, dtype=initial.dtype
            )

        final_shape = (
            batch,
            final_samples,
            self.cloud.noise_token_count,
            self.cloud.noise_token_dim,
        )
        if final_noise is None:
            final_noise = torch.randn(
                *final_shape, device=initial.device, dtype=initial.dtype
            )
        elif final_noise.shape != final_shape:
            raise ValueError("final_noise must have shape [B,M,K,R]")
        else:
            final_noise = final_noise.to(device=initial.device, dtype=initial.dtype)

        hidden = self.cloud.encode_anchor(initial)
        intermediate_gates = []
        intermediate_updates = []
        energies = []
        for step in range(intermediate_steps):
            step_levels = levels[:, step]
            proposed, energy = self._condition_and_refine(
                hidden, condition_tokens, condition_mask, step_levels
            )
            spatial_amplitude = (energy / self.cloud.dim).sqrt()
            proposed = self.cloud._run(
                self.cloud.noise_attention,
                proposed,
                intermediate_noise[:, step],
                spatial_amplitude,
            )
            for block in self.cloud.refine_blocks[: self.intermediate_refine_depth]:
                proposed = self.cloud._run(block, proposed)
            hidden, gate, update = self.cloud.state_update(
                hidden,
                proposed,
                step_levels if self.use_level_conditioning else None,
            )
            intermediate_gates.append(gate.float().mean(dim=(1, 2, 3)))
            intermediate_updates.append(
                update.float().square().mean(dim=(1, 2, 3)).sqrt()
            )
            energies.append(energy)

        final_levels = levels[:, -1]
        proposed, final_energy = self._condition_and_refine(
            hidden, condition_tokens, condition_mask, final_levels
        )
        energies.append(final_energy)
        proposed = proposed[:, None].expand(-1, final_samples, -1, -1, -1)
        proposed = proposed.reshape(batch * final_samples, height, width, self.cloud.dim)
        expanded_hidden = hidden[:, None].expand(-1, final_samples, -1, -1, -1)
        expanded_hidden = expanded_hidden.reshape(
            batch * final_samples, height, width, self.cloud.dim
        )
        expanded_levels = final_levels[:, None].expand(-1, final_samples).reshape(-1)
        expanded_energy = final_energy[:, None].expand(-1, final_samples, -1, -1)
        expanded_energy = expanded_energy.reshape(batch * final_samples, height, width)
        proposed = self.cloud._run(
            self.cloud.noise_attention,
            proposed,
            final_noise.reshape(
                batch * final_samples,
                self.cloud.noise_token_count,
                self.cloud.noise_token_dim,
            ),
            (expanded_energy / self.cloud.dim).sqrt(),
        )
        for block in self.cloud.refine_blocks:
            proposed = self.cloud._run(block, proposed)
        final_hidden, final_gate, final_update = self.cloud.state_update(
            expanded_hidden,
            proposed,
            expanded_levels if self.use_level_conditioning else None,
        )
        fields = self.cloud.decode_absolute(final_hidden)

        empty = initial.new_empty((batch, 0), dtype=torch.float32)
        return AbsoluteHiddenRolloutOutput(
            fields=fields.reshape(batch, final_samples, *fields.shape[1:]),
            final_hidden=final_hidden.reshape(
                batch, final_samples, height, width, self.cloud.dim
            ),
            intermediate_gate_means=(
                torch.stack(intermediate_gates, dim=1)
                if intermediate_gates
                else empty
            ),
            intermediate_update_rms=(
                torch.stack(intermediate_updates, dim=1)
                if intermediate_updates
                else empty
            ),
            final_gate_means=final_gate.float()
            .mean(dim=(1, 2, 3))
            .reshape(batch, final_samples),
            final_update_rms=final_update.float()
            .square()
            .mean(dim=(1, 2, 3))
            .sqrt()
            .reshape(batch, final_samples),
            spatial_noise_energy=torch.stack(energies, dim=1),
        )


class AbsoluteTrajectoryRolloutCloud(AbsoluteHiddenRolloutCloud):
    """Keep independent random paths alive through every hidden rollout step.

    Unlike :class:`AbsoluteHiddenRolloutCloud`, this module branches before the
    first transition and decodes every step.  This makes each transition
    directly supervisable without an exponential branch expansion.
    """

    def forward(
        self,
        initial: Tensor,
        condition_tokens: Tensor,
        levels: Tensor,
        *,
        samples: int = 4,
        noise: Tensor | None = None,
        condition_mask: Tensor | None = None,
        reencode_every: int | None = None,
        compute_hidden_consistency: bool = False,
    ) -> AbsoluteTrajectoryRolloutOutput:
        if initial.ndim != 4 or initial.shape[1] != self.cloud.field_channels:
            raise ValueError("initial must have shape [B,C,H,W]")
        if condition_tokens.ndim != 3 or condition_tokens.shape[0] != initial.shape[0]:
            raise ValueError("condition_tokens must have shape [B,K,C]")
        if samples < 2:
            raise ValueError("samples must be at least two")
        if reencode_every is not None and reencode_every < 1:
            raise ValueError("reencode_every must be positive when provided")
        if condition_mask is not None and condition_mask.shape != condition_tokens.shape[:2]:
            raise ValueError("condition_mask must have shape [B,K]")

        batch, _, height, width = initial.shape
        levels = self._normalize_levels(levels, batch, initial.device)
        steps = levels.shape[1]
        expected_noise = (
            batch,
            steps,
            samples,
            self.cloud.noise_token_count,
            self.cloud.noise_token_dim,
        )
        if noise is None:
            noise = torch.randn(
                *expected_noise, device=initial.device, dtype=initial.dtype
            )
        elif noise.shape != expected_noise:
            raise ValueError("noise must have shape [B,T,M,K,R]")
        else:
            noise = noise.to(device=initial.device, dtype=initial.dtype)

        hidden = self.cloud.encode_anchor(initial)
        hidden = hidden[:, None].expand(-1, samples, -1, -1, -1)
        expanded_condition = torch.repeat_interleave(
            condition_tokens, samples, dim=0
        )
        expanded_mask = None
        if condition_mask is not None:
            expanded_mask = torch.repeat_interleave(
                condition_mask, samples, dim=0
            )

        decoded_steps = []
        gate_steps = []
        update_steps = []
        energy_steps = []
        hidden_consistency_steps = []
        for step in range(steps):
            flat_hidden = hidden.reshape(batch * samples, height, width, self.cloud.dim)
            step_levels = levels[:, step][:, None].expand(-1, samples).reshape(-1)
            proposed, energy = self._condition_and_refine(
                flat_hidden, expanded_condition, expanded_mask, step_levels
            )
            proposed = self.cloud._run(
                self.cloud.noise_attention,
                proposed,
                noise[:, step].reshape(
                    batch * samples,
                    self.cloud.noise_token_count,
                    self.cloud.noise_token_dim,
                ),
                (energy / self.cloud.dim).sqrt(),
            )
            for block in self.cloud.refine_blocks[: self.intermediate_refine_depth]:
                proposed = self.cloud._run(block, proposed)
            flat_hidden, gate, update = self.cloud.state_update(
                flat_hidden,
                proposed,
                step_levels if self.use_level_conditioning else None,
            )
            decoded = self.cloud.decode_absolute(flat_hidden)
            if compute_hidden_consistency:
                # The re-encoded image is the manifold reference, not a second
                # trainable route through which the consistency loss can be
                # reduced.  The residual state alone is pulled toward E(D(h)).
                with torch.no_grad():
                    reencoded_hidden = self.cloud.encode_anchor(decoded.detach())
                hidden_consistency_steps.append(
                    (flat_hidden.float() - reencoded_hidden.float())
                    .square()
                    .mean(dim=(1, 2, 3))
                    .reshape(batch, samples)
                )
            hidden = flat_hidden.reshape(
                batch, samples, height, width, self.cloud.dim
            )
            decoded_steps.append(
                decoded.reshape(batch, samples, *decoded.shape[1:])
            )
            gate_steps.append(
                gate.float().mean(dim=(1, 2, 3)).reshape(batch, samples)
            )
            update_steps.append(
                update.float()
                .square()
                .mean(dim=(1, 2, 3))
                .sqrt()
                .reshape(batch, samples)
            )
            energy_steps.append(
                energy.reshape(batch, samples, height, width)
            )
            if (
                reencode_every is not None
                and (step + 1) % reencode_every == 0
                and step + 1 < steps
            ):
                flat_hidden = self.cloud.encode_anchor(decoded)
                hidden = flat_hidden.reshape(
                    batch, samples, height, width, self.cloud.dim
                )

        return AbsoluteTrajectoryRolloutOutput(
            fields=torch.stack(decoded_steps, dim=1),
            final_hidden=hidden,
            hidden_consistency_mse=(
                torch.stack(hidden_consistency_steps, dim=1)
                if hidden_consistency_steps
                else None
            ),
            gate_means=torch.stack(gate_steps, dim=1),
            update_rms=torch.stack(update_steps, dim=1),
            spatial_noise_energy=torch.stack(energy_steps, dim=1),
        )


class HiddenRolloutCloud(nn.Module):
    """Reuse a Cloud step as a weight-tied hidden residual network.

    The wrapped ``MolecularFieldCloud`` remains checkpoint-compatible.  Load the
    pretrained Cloud first, then construct this module for hidden-rollout fine-tuning.
    The input field is encoded once; decoded images never feed the next step.
    """

    def __init__(
        self,
        cloud: MolecularFieldCloud,
        *,
        max_level: int = 64,
        gate_init: float = 0.02,
        decoder_dim: int | None = None,
        kv_grid_sizes: list[int] | tuple[int, ...] = (32, 16, 8),
    ) -> None:
        super().__init__()
        self.cloud = cloud
        self.max_level = max_level
        condition_attention = cloud.condition_blocks[0].attention
        self.state_update = EncoderAnchoredHiddenUpdate(
            cloud.dim,
            condition_attention.attention.num_heads,
            max_level,
            gate_init,
            kernel_sizes=condition_attention.kernel_sizes,
            grid_sizes=kv_grid_sizes,
        )
        self.output_head = AbsoluteFieldHead(
            cloud.dim, cloud.field_channels, hidden_dim=decoder_dim
        )

    def encode_anchor(self, field: Tensor) -> Tensor:
        """Encode the fixed input-image anchor, including its spatial position."""

        if field.ndim != 4 or field.shape[1] != self.cloud.field_channels:
            raise ValueError("field must have shape [B,C,H,W]")
        _, _, height, width = field.shape
        hidden = self.cloud.field_encoder(field)
        position = sinusoidal_2d_position(
            height, width, self.cloud.dim, hidden.device, hidden.dtype
        )[None]
        return hidden + position

    def decode_absolute(self, hidden: Tensor) -> Tensor:
        """Decode a hidden grid without adding the input image as a residual."""

        field = self.output_head(hidden)
        if self.cloud.zero_mean_output:
            mean = field.float().mean(dim=(-2, -1), keepdim=True)
            field = field - mean.to(field.dtype)
        return field

    def _normalize_levels(
        self, levels: Tensor, batch: int, steps: int, device: torch.device
    ) -> Tensor:
        levels = levels.to(device=device, dtype=torch.long)
        if levels.ndim == 1 and levels.shape[0] == steps:
            levels = levels[None].expand(batch, -1)
        if levels.shape != (batch, steps):
            raise ValueError("levels must have shape [T] or [B,T]")
        if torch.any((levels < 0) | (levels > self.max_level)):
            raise ValueError(f"levels must be in [0, {self.max_level}]")
        return levels

    def _normalize_noise(
        self,
        noise: Tensor | None,
        batch: int,
        samples: int,
        steps: int,
        reference: Tensor,
    ) -> Tensor:
        expected = (
            batch,
            samples,
            steps,
            self.cloud.noise_token_count,
            self.cloud.noise_token_dim,
        )
        if noise is None:
            return torch.randn(*expected, device=reference.device, dtype=reference.dtype)
        if noise.shape != expected:
            raise ValueError("noise must have shape [B,M,T,K,R]")
        return noise.to(device=reference.device, dtype=reference.dtype)

    def forward(
        self,
        initial: Tensor,
        condition: Tensor,
        levels: Tensor,
        *,
        samples: int = 4,
        noise: Tensor | None = None,
    ) -> HiddenRolloutOutput:
        if initial.ndim != 4 or initial.shape[1] != self.cloud.field_channels:
            raise ValueError("initial must have shape [B,C,H,W]")
        if condition.ndim != 2 or condition.shape[0] != initial.shape[0]:
            raise ValueError("condition must have shape [B,D]")
        if samples < 1:
            raise ValueError("samples must be positive")
        if levels.ndim not in (1, 2):
            raise ValueError("levels must have shape [T] or [B,T]")

        batch, _, height, width = initial.shape
        steps = levels.shape[-1] if levels.ndim > 1 else levels.shape[0]
        if steps < 1:
            raise ValueError("at least one rollout level is required")
        levels = self._normalize_levels(levels, batch, steps, initial.device)

        encoder_hidden = self.encode_anchor(initial)
        anchor_reconstruction = self.decode_absolute(encoder_hidden)
        position = sinusoidal_2d_position(
            height, width, self.cloud.dim, encoder_hidden.device, encoder_hidden.dtype
        )[None]
        hidden = encoder_hidden
        context = self.cloud.condition_plane(condition, height, width) + position
        hidden = hidden[:, None].expand(-1, samples, -1, -1, -1)
        hidden = hidden.reshape(batch * samples, height, width, self.cloud.dim)
        encoder_hidden = encoder_hidden[:, None].expand(-1, samples, -1, -1, -1)
        encoder_hidden = encoder_hidden.reshape(
            batch * samples, height, width, self.cloud.dim
        )
        context = context[:, None].expand(-1, samples, -1, -1, -1)
        context = context.reshape(batch * samples, height, width, self.cloud.dim)
        noise = self._normalize_noise(noise, batch, samples, steps, hidden)

        fields = []
        energies = []
        gate_means = []
        update_rms = []
        for step in range(steps):
            proposed = hidden
            for block in self.cloud.condition_blocks:
                proposed = self.cloud._run(block, proposed, context)
            energy = self.cloud.spatial_noise_energy(proposed)
            spatial_amplitude = (energy / self.cloud.dim).sqrt()
            noise_step = noise[:, :, step].reshape(
                batch * samples,
                self.cloud.noise_token_count,
                self.cloud.noise_token_dim,
            )
            proposed = self.cloud._run(
                self.cloud.noise_attention, proposed, noise_step, spatial_amplitude
            )
            for block in self.cloud.refine_blocks:
                proposed = self.cloud._run(block, proposed)

            step_levels = levels[:, step, None].expand(-1, samples).reshape(-1)
            hidden, gate, update = self.state_update(
                encoder_hidden, proposed, step_levels
            )
            field = self.decode_absolute(hidden)

            fields.append(field.reshape(batch, samples, *field.shape[1:]))
            energies.append(energy.reshape(batch, samples, height, width))
            gate_means.append(gate.float().mean(dim=(1, 2, 3)).reshape(batch, samples))
            update_rms.append(
                update.float()
                .square()
                .mean(dim=(1, 2, 3))
                .sqrt()
                .reshape(batch, samples)
            )

        return HiddenRolloutOutput(
            fields=torch.stack(fields, dim=2),
            anchor_reconstruction=anchor_reconstruction,
            final_hidden=hidden.reshape(batch, samples, height, width, self.cloud.dim),
            gate_means=torch.stack(gate_means, dim=2),
            update_rms=torch.stack(update_rms, dim=2),
            spatial_noise_energy=torch.stack(energies, dim=2),
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
