"""Checkpoint-compatible residual-image rollouts from the pre-absolute Cloud."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from molai.models.attention import TokenCrossBlock2d, sinusoidal_2d_position
from molai.models.cloud import MolecularFieldCloud
from molai.models.condition import SpectrumConditionEncoder, SpectrumTokenEncoder


class ScalarGatedRandomMemoryAttention(nn.Module):
    """Preserve the exact scalar gate and parameter names of the relative runs.

    Do not silently replace this by the newer GLU attention: the pretrained
    output projection and signed scalar gate have different semantics.
    """

    def __init__(self, dim: int, heads: int, random_dim: int, temperature: float, gate_init: float):
        super().__init__()
        if dim % heads or temperature <= 0:
            raise ValueError("dim must be divisible by heads and temperature positive")
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

    def forward(self, query: Tensor, random_tokens: Tensor, spatial_amplitude: Tensor) -> Tensor:
        batch, height, width, dim = query.shape
        if dim != self.dim or random_tokens.shape[0] != batch or random_tokens.shape[-1] != self.random_dim:
            raise ValueError("query/random memory dimensions do not match")
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
            key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3),
            dropout_p=0.0, is_causal=False,
        )
        attended = attended.permute(0, 2, 1, 3).reshape(batch, height, width, dim)
        update = self.output_projection(attended) * spatial_amplitude[..., None]
        return query + self.gate * update


class UnnormalizedGaussianRandomMemoryAttention(nn.Module):
    """Attend to conditionally sampled Gaussian tokens without erasing moments.

    The Gaussian controller owns the token mean and standard deviation.  K/V
    normalization here would make a scalar change in standard deviation nearly
    invisible, so only the image query keeps the historical pre-normalization.
    Parameter names intentionally match ``ScalarGatedRandomMemoryAttention`` so
    its learned projections and scalar residual gate can be transferred exactly.
    """

    def __init__(self, dim: int, heads: int, random_dim: int, temperature: float, gate_init: float):
        super().__init__()
        if dim % heads or temperature <= 0:
            raise ValueError("dim must be divisible by heads and temperature positive")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.random_dim = random_dim
        self.temperature = temperature
        self.query_norm = nn.LayerNorm(dim)
        self.query_projection = nn.Linear(dim, dim, bias=False)
        self.key_projection = nn.Linear(random_dim, dim, bias=False)
        self.value_projection = nn.Linear(random_dim, dim, bias=False)
        self.output_projection = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, query: Tensor, random_tokens: Tensor, spatial_amplitude: Tensor) -> Tensor:
        batch, height, width, dim = query.shape
        if dim != self.dim or random_tokens.shape[0] != batch or random_tokens.shape[-1] != self.random_dim:
            raise ValueError("query/random memory dimensions do not match")
        if spatial_amplitude.shape != (batch, height, width):
            raise ValueError("spatial_amplitude must have shape [B,H,W]")
        projected_query = self.query_projection(self.query_norm(query)).reshape(
            batch, height * width, self.heads, self.head_dim
        )
        key = self.key_projection(random_tokens).reshape(
            batch, random_tokens.shape[1], self.heads, self.head_dim
        )
        value = self.value_projection(random_tokens).reshape(
            batch, random_tokens.shape[1], self.heads, self.head_dim
        )
        attended = F.scaled_dot_product_attention(
            projected_query.permute(0, 2, 1, 3) / self.temperature,
            key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3),
            dropout_p=0.0, is_causal=False,
        )
        attended = attended.permute(0, 2, 1, 3).reshape(batch, height, width, dim)
        update = self.output_projection(attended) * spatial_amplitude[..., None]
        return query + self.gate * update


class ConditionalGaussianTokenController(nn.Module):
    """Infer one diagonal Gaussian from MS-Q / 32x32 image-K,V alignment."""

    def __init__(
        self,
        image_dim: int,
        ms_dim: int,
        token_dim: int,
        heads: int,
        *,
        image_grid_size: int = 32,
        ffn_ratio: float = 2.0,
        cross_gate_init: float = 0.1,
        sigma_min: float = 0.05,
        sigma_max: float = 2.0,
        sigma_init: float = 1.0,
    ) -> None:
        super().__init__()
        if image_dim % heads:
            raise ValueError("image/controller dimension must be divisible by heads")
        if image_grid_size < 1:
            raise ValueError("image_grid_size must be positive")
        if not 0 <= sigma_min < sigma_init < sigma_max:
            raise ValueError("require 0 <= sigma_min < sigma_init < sigma_max")
        self.image_grid_size = image_grid_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.ms_projection = nn.Linear(ms_dim, image_dim)
        self.ms_norm = nn.LayerNorm(image_dim)
        self.image_norm = nn.LayerNorm(image_dim)
        self.cross_attention = nn.MultiheadAttention(image_dim, heads, batch_first=True)
        self.cross_gate = nn.Parameter(torch.tensor(float(cross_gate_init)))
        hidden = max(image_dim, round(image_dim * ffn_ratio))
        self.ffn_norm = nn.LayerNorm(image_dim)
        self.ffn = nn.Sequential(
            nn.Linear(image_dim, hidden), nn.GELU(), nn.Linear(hidden, image_dim),
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, image_dim) * 0.02)
        self.pool_norm = nn.LayerNorm(image_dim)
        self.pool_attention = nn.MultiheadAttention(image_dim, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(image_dim)
        self.mean_head = nn.Linear(image_dim, token_dim)
        self.scale_head = nn.Linear(image_dim, token_dim)
        nn.init.normal_(self.mean_head.weight, std=1e-3)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        fraction = (sigma_init - sigma_min) / (sigma_max - sigma_min)
        nn.init.constant_(self.scale_head.bias, math.log(fraction / (1.0 - fraction)))

    def forward(
        self, ms_tokens: Tensor, ms_mask: Tensor, image_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if ms_tokens.ndim != 3 or ms_mask.shape != ms_tokens.shape[:2]:
            raise ValueError("MS tokens and mask must have shapes [B,L,D] and [B,L]")
        if image_hidden.ndim != 4 or image_hidden.shape[0] % ms_tokens.shape[0]:
            raise ValueError("image paths must be grouped by their MS owner")
        paths_per_spectrum = image_hidden.shape[0] // ms_tokens.shape[0]
        if paths_per_spectrum > 1:
            # A cached MS bank is shared by all recurrent image paths.  The
            # controller still predicts path-specific moments because its K/V
            # image differs after the first rollout step.
            ms_tokens = ms_tokens.repeat_interleave(paths_per_spectrum, dim=0)
            ms_mask = ms_mask.repeat_interleave(paths_per_spectrum, dim=0)
        # Production images are 128x128.  The min keeps small unit-test models
        # meaningful without silently upsampling their spatial representation.
        side = min(self.image_grid_size, image_hidden.shape[1], image_hidden.shape[2])
        image = F.adaptive_avg_pool2d(
            image_hidden.permute(0, 3, 1, 2), (side, side)
        ).flatten(2).transpose(1, 2)
        ms = self.ms_projection(ms_tokens)
        normalized_image = self.image_norm(image)
        update, _ = self.cross_attention(
            self.ms_norm(ms), normalized_image, normalized_image,
            need_weights=False,
        )
        aligned = (ms + self.cross_gate * update).masked_fill(~ms_mask[..., None], 0.0)
        aligned = (aligned + self.ffn(self.ffn_norm(aligned))).masked_fill(
            ~ms_mask[..., None], 0.0
        )
        query = self.pool_query.expand(len(ms_tokens), -1, -1)
        pooled, _ = self.pool_attention(
            query, self.pool_norm(aligned), self.pool_norm(aligned),
            key_padding_mask=~ms_mask.bool(), need_weights=False,
        )
        pooled = self.output_norm(pooled[:, 0])
        mean = self.mean_head(pooled)
        fraction = torch.sigmoid(self.scale_head(pooled).float())
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * fraction
        return mean, sigma.to(mean.dtype)


def build_relative_models(
    checkpoint: dict, metadata_dim: int, *, gradient_checkpointing: bool = False,
    cloud_class: type[MolecularFieldCloud] = MolecularFieldCloud,
) -> tuple[MolecularFieldCloud, SpectrumConditionEncoder]:
    """Strictly restore the old model without changing dimensions or K/V grids."""
    config = checkpoint["config"]
    options = dict(config["model"])
    options["dim"] = options.pop("fullres_dim")
    if "cvt_output_sizes" in options:
        options["cvt_grid_sizes"] = options.pop("cvt_output_sizes")
    options["gradient_checkpointing"] = gradient_checkpointing
    cloud = cloud_class(**options)
    if "noise_attention.gate" in checkpoint["cloud"]:
        cloud.noise_attention = ScalarGatedRandomMemoryAttention(
            cloud.dim, int(options["heads"]), cloud.noise_token_dim,
            float(options["noise_temperature"]), float(options["noise_gate_init"]),
        )
    encoder = SpectrumConditionEncoder(
        metadata_dim=metadata_dim, dim=int(options["condition_dim"]),
        peak_chunk_size=int(config["training"].get("peak_chunk_size", 256)),
        **config["spectrum_encoder"],
    )
    cloud.load_state_dict(checkpoint["cloud"], strict=True)
    encoder.load_state_dict(checkpoint["condition_encoder"], strict=True)
    # This historical readout is not used by image-only objectives.
    cloud.readout.requires_grad_(False)
    return cloud, encoder


class TokenConditionRelativeCloud(MolecularFieldCloud):
    """Retain the pretrained image residual; query real MS tokens, not a plane."""

    cache_condition_kv = False
    packed_condition_kv = False
    uncheckpointed_refine_blocks = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_token_controller: ConditionalGaussianTokenController | None = None

    def _run(self, module, *args):
        skip = self.uncheckpointed_refine_blocks
        if skip and any(module is block for block in self.refine_blocks[-skip:]):
            return module(*args)
        return super()._run(module, *args)

    def prepare_condition(self, tokens: Tensor, mask: Tensor):
        projected = tuple(block.project_context(tokens) for block in self.condition_blocks)
        if (self.packed_condition_kv and tokens.is_cuda
                and torch.backends.cuda.is_flash_attention_available()):
            counts = mask.sum(dim=1, dtype=torch.int32)
            cu_seq = torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))
            return tuple((key.transpose(1, 2)[mask], value.transpose(1, 2)[mask], cu_seq, tokens.shape[1])
                         for key, value in projected)
        return projected

    def encode_image(self, current: Tensor) -> Tensor:
        if current.ndim != 4 or current.shape[1] != self.field_channels:
            raise ValueError("current must have shape [B,C,H,W]")
        hidden = self.field_encoder(current)
        height, width = current.shape[-2:]
        return hidden + sinusoidal_2d_position(
            height, width, self.dim, hidden.device, hidden.dtype,
        )[None]

    def apply_condition(self, hidden: Tensor, tokens: Tensor, mask: Tensor, prepared) -> Tensor:
        for index, block in enumerate(self.condition_blocks):
            if prepared is None:
                hidden = self._run(block, hidden, tokens, mask)
            else:
                context = prepared[index]
                if len(context) == 4:
                    hidden = self._run(block.forward_varlen, hidden, *context)
                else:
                    key, value = context
                    hidden = self._run(block.forward_projected, hidden, key, value, mask)
        return hidden

    def encode_condition(self, current: Tensor, condition) -> Tensor:
        tokens, mask, prepared = condition
        return self.apply_condition(self.encode_image(current), tokens, mask, prepared)

    def forward(self, current, condition, samples=4, noise=None, condition_mask=None,
                prepared_condition=None):
        if self.random_token_controller is None:
            return super().forward(
                current, (condition, condition_mask, prepared_condition), samples, noise,
            )
        if condition_mask is None or condition_mask.shape != condition.shape[:2]:
            raise ValueError("conditional Gaussian tokens require a matching MS mask")
        image_hidden = self.encode_image(current)
        encoded = self.apply_condition(image_hidden, condition, condition_mask, prepared_condition)
        energy = self.spatial_noise_energy(encoded)
        epsilon = self._prepare_noise(encoded, samples, noise)
        mean, sigma = self._run(
            self.random_token_controller, condition, condition_mask, image_hidden,
        )
        gaussian_tokens = mean[:, None, None, :] + sigma[:, None, None, :] * epsilon

        batch, height, width, _ = encoded.shape
        tokens = encoded[:, None].expand(-1, samples, -1, -1, -1)
        tokens = tokens.reshape(batch * samples, height, width, self.dim)
        random_context = gaussian_tokens.reshape(
            batch * samples, self.noise_token_count, self.noise_token_dim
        )
        spatial_amplitude = (energy / self.dim).sqrt()
        spatial_amplitude = spatial_amplitude[:, None].expand(-1, samples, -1, -1)
        spatial_amplitude = spatial_amplitude.reshape(batch * samples, height, width)
        tokens = self._run(self.noise_attention, tokens, random_context, spatial_amplitude)
        for block in self.refine_blocks:
            tokens = self._run(block, tokens)

        residual = self.output_head(self.output_norm(tokens)).permute(0, 3, 1, 2)
        expanded_current = current[:, None].expand(-1, samples, -1, -1, -1)
        fields = expanded_current.reshape(batch * samples, *current.shape[1:]) + residual
        if self.zero_mean_output:
            fields = fields - fields.float().mean(dim=(-2, -1), keepdim=True).to(fields.dtype)
        molecular_embeddings = self.readout(tokens.mean(dim=(1, 2)))
        return (
            fields.reshape(batch, samples, *current.shape[1:]),
            energy,
            molecular_embeddings.reshape(batch, samples, -1),
        )


def configure_conditional_gaussian_tokens(
    cloud: TokenConditionRelativeCloud, options: dict,
) -> dict:
    """Install or validate the opt-in MS/image-conditioned Gaussian memory."""
    enabled = bool(options.get("conditional_gaussian_tokens", False))
    if not enabled:
        if cloud.random_token_controller is not None:
            raise ValueError("a Gaussian-token checkpoint cannot disable its controller")
        return {"conditional_gaussian_tokens": False}
    if cloud.random_token_controller is not None:
        return {
            "conditional_gaussian_tokens": True,
            "gaussian_controller": "restored",
            "gaussian_image_grid_size": cloud.random_token_controller.image_grid_size,
        }
    old_attention = cloud.noise_attention
    if not isinstance(old_attention, ScalarGatedRandomMemoryAttention):
        raise TypeError(
            "conditional Gaussian conversion requires the historical scalar-gated random attention"
        )
    attention = UnnormalizedGaussianRandomMemoryAttention(
        cloud.dim,
        old_attention.heads,
        cloud.noise_token_dim,
        old_attention.temperature,
        float(old_attention.gate.detach()),
    )
    # LayerNorm without affine parameters contributed no checkpoint tensors, so
    # every learned projection and the historical scalar gate transfer strictly.
    attention.load_state_dict(old_attention.state_dict(), strict=True)
    cloud.noise_attention = attention
    cloud.random_token_controller = ConditionalGaussianTokenController(
        image_dim=cloud.dim,
        ms_dim=int(options.get("gaussian_ms_dim", 64)),
        token_dim=cloud.noise_token_dim,
        heads=int(options.get("gaussian_controller_heads", old_attention.heads)),
        image_grid_size=int(options.get("gaussian_image_grid_size", 32)),
        ffn_ratio=float(options.get("gaussian_controller_ffn_ratio", 2.0)),
        cross_gate_init=float(options.get("gaussian_cross_gate_init", 0.1)),
        sigma_min=float(options.get("gaussian_sigma_min", 0.05)),
        sigma_max=float(options.get("gaussian_sigma_max", 2.0)),
        sigma_init=float(options.get("gaussian_sigma_init", 1.0)),
    )
    return {
        "conditional_gaussian_tokens": True,
        "gaussian_controller": "new-ms-query-image-kv-attention-pool",
        "gaussian_image_grid_size": cloud.random_token_controller.image_grid_size,
        "gaussian_zero": "epsilon-zero-all-64-tokens-equal-predicted-mean",
        "gaussian_random": "predicted-mean-plus-sigma-times-standard-normal",
        "random_attention_kv_norm": "removed",
        "transferred_random_attention_tensors": len(attention.state_dict()),
    }


def build_token_relative_models(checkpoint: dict, metadata_dim: int, *, gradient_checkpointing=True):
    """Audit every reused weight; replace only the legacy MS conditioning path."""
    cloud, old_encoder = build_relative_models(
        checkpoint, metadata_dim, gradient_checkpointing=gradient_checkpointing,
        cloud_class=TokenConditionRelativeCloud,
    )
    # The inheritance changes only conditioning. Image/refine/noise/output modules
    # remain the exact loaded module objects, including the signed random gate.
    del cloud.condition_plane
    options = checkpoint["config"]["model"]
    cloud.condition_blocks = nn.ModuleList([
        TokenCrossBlock2d(cloud.dim, 64, int(options["heads"]),
                          float(options["ffn_ratio"]), float(options["condition_gate_init"]))
        for _ in range(int(options["condition_cross_depth"]))
    ])
    spectrum = dict(checkpoint["config"]["spectrum_encoder"])
    spectrum.update(heads=4, peak_conv_stages=2, spectrum_layers=2)
    encoder = SpectrumTokenEncoder(
        metadata_dim=metadata_dim, dim=64,
        peak_chunk_size=int(checkpoint["config"]["training"].get("peak_chunk_size", 256)),
        **spectrum,
    )
    old_state, new_state = old_encoder.state_dict(), encoder.state_dict()
    reused = {key: value for key, value in old_state.items()
              if key in new_state and value.shape == new_state[key].shape}
    encoder.load_state_dict({**new_state, **reused}, strict=True)
    audit = {
        "mode": "64d-ms-tokens-ordinary-cross-attention",
        "reused_ms_keys": sorted(reused),
        "new_ms_keys": sorted(set(new_state) - set(reused)),
        "replaced_cloud_prefixes": ["condition_plane", "condition_blocks"],
        "unchanged_image_noise_refine_output": True,
    }
    return cloud, encoder, audit


def build_fresh_token_relative_models(config: dict, metadata_dim: int, *, gradient_checkpointing=True):
    """Initialize the relative/token topology without loading any trained weights."""
    options = dict(config["model"])
    options["dim"] = options.pop("fullres_dim")
    if "cvt_output_sizes" in options:
        options["cvt_grid_sizes"] = options.pop("cvt_output_sizes")
    options.update(condition_dim=64, gradient_checkpointing=gradient_checkpointing)
    cloud = TokenConditionRelativeCloud(**options)
    # Preserve the historical relative architecture, not the newer GLU variant.
    cloud.noise_attention = ScalarGatedRandomMemoryAttention(
        cloud.dim, int(options["heads"]), cloud.noise_token_dim,
        float(options["noise_temperature"]), float(options["noise_gate_init"]),
    )
    del cloud.condition_plane
    cloud.condition_blocks = nn.ModuleList([
        TokenCrossBlock2d(cloud.dim, 64, int(options["heads"]),
                          float(options["ffn_ratio"]), float(options["condition_gate_init"]))
        for _ in range(int(options["condition_cross_depth"]))
    ])
    gaussian_audit = configure_conditional_gaussian_tokens(
        cloud, config.get("relative_image_rollout", {}),
    )
    cloud.readout.requires_grad_(False)
    spectrum = dict(config["spectrum_encoder"])
    spectrum.update(heads=4, peak_conv_stages=2, spectrum_layers=2)
    encoder = SpectrumTokenEncoder(
        metadata_dim=metadata_dim, dim=64,
        peak_chunk_size=int(config["training"].get("peak_chunk_size", 256)), **spectrum,
    )
    return cloud, encoder, {
        "mode": "scratch-64d-ms-tokens-ordinary-cross-attention",
        "initialization": "random", "loaded_parameter_tensors": 0,
        "random_attention": "conditional-gaussian" if gaussian_audit["conditional_gaussian_tokens"]
        else "historical-scalar-gate",
        **gaussian_audit,
    }


def build_ms_reset_relative_models(checkpoint: dict, config: dict, metadata_dim: int,
                                   *, gradient_checkpointing=True):
    """Strictly preserve the entire token Cloud; initialize only the MS encoder.

    The caller supplies a copy of the checkpoint config with only spectrum input
    settings replaced/bound to the new dataset. Never partially load a backbone
    or reuse compatible MS tensors: resetting means the whole encoder is new.
    """
    if "condition_plane.weight" in checkpoint["cloud"] or any(
        key.startswith("condition_plane.") for key in checkpoint["cloud"]
    ):
        raise ValueError("MS-only reset requires a token-conditioned checkpoint")
    if config["model"] != checkpoint["config"]["model"]:
        raise ValueError("MS-only reset cannot change the Cloud architecture")
    cloud, encoder, _ = build_fresh_token_relative_models(
        config, metadata_dim, gradient_checkpointing=gradient_checkpointing,
    )
    cloud.load_state_dict(checkpoint["cloud"], strict=True)
    restored = cloud.state_dict()
    if not all(torch.equal(restored[key], value) for key, value in checkpoint["cloud"].items()):
        raise RuntimeError("Cloud weight transfer changed a tensor")
    return cloud, encoder, {
        "mode": "pretrained-cloud-reset-ms-encoder",
        "strict_cloud_load": True,
        "all_cloud_tensors_identical": True,
        "loaded_cloud_tensors": len(restored),
        "loaded_ms_tensors": 0,
        "new_ms_keys": sorted(encoder.state_dict()),
        "replaced_cloud_prefixes": [],
        "optimizer_state": "fresh-all-parameters",
    }


def restore_token_relative_models(checkpoint: dict, metadata_dim: int, *, gradient_checkpointing=True):
    """Restore a trained token-conditioned Cloud and MS encoder without conversion."""
    if any(key.startswith("condition_plane.") for key in checkpoint["cloud"]):
        raise ValueError("strict token restoration requires a token-conditioned checkpoint")
    cloud, encoder, _ = build_fresh_token_relative_models(
        checkpoint["config"], metadata_dim, gradient_checkpointing=gradient_checkpointing,
    )
    cloud.load_state_dict(checkpoint["cloud"], strict=True)
    encoder.load_state_dict(checkpoint["condition_encoder"], strict=True)
    for module, saved in ((cloud, checkpoint["cloud"]), (encoder, checkpoint["condition_encoder"])):
        if not all(torch.equal(value, saved[key]) for key, value in module.state_dict().items()):
            raise RuntimeError("strict token restoration changed a tensor")
    return cloud, encoder, {"mode": "strict-pretrained-token-cloud-and-ms",
                            "strict_cloud_load": True, "strict_ms_load": True,
                            "all_loaded_tensors_identical": True,
                            "loaded_cloud_tensors": len(checkpoint["cloud"]),
                            "loaded_ms_tensors": len(checkpoint["condition_encoder"])}


def configure_relative_execution(cloud, options):
    """Execution-only knobs: no changes to parameters, paths, grids, or objective."""
    cloud.cache_condition_kv = bool(options.get("cache_ms_kv", False))
    cloud.packed_condition_kv = bool(options.get("packed_ms_kv", False))
    cloud.uncheckpointed_refine_blocks = int(options.get("uncheckpointed_refine_blocks", 0))
    if not 0 <= cloud.uncheckpointed_refine_blocks <= len(cloud.refine_blocks):
        raise ValueError("uncheckpointed refine count exceeds the model depth")
    compile_options = {"emulate_precision_casts": bool(options.get("preserve_bf16_rounding", True))}

    def compile_call(function):
        return torch.compile(function, fullgraph=True, dynamic=True, options=compile_options)

    if bool(options.get("compile_blocks", False)):
        cloud.field_encoder.forward = compile_call(cloud.field_encoder.forward)
        optional = (() if cloud.random_token_controller is None else (cloud.random_token_controller,))
        for module in (*cloud.condition_blocks, *optional, cloud.noise_attention, *cloud.refine_blocks):
            module.forward = compile_call(module.forward)
        if cloud.cache_condition_kv:
            for module in cloud.condition_blocks:
                module.forward_projected = compile_call(module.forward_projected)
                module.forward_varlen = compile_call(module.forward_varlen)
    elif bool(options.get("compile_pointwise", False)):
        for block in (*cloud.condition_blocks, *cloud.refine_blocks):
            block.ff.forward = compile_call(block.ff.forward)
    return compile_options


class RelativeImageRolloutCloud(nn.Module):
    """Maintain zero + independent random image paths with full recurrent BP.

    Every step re-encodes its own predicted image. Branch once, not M**T times.
    The MS encoder is evaluated outside this module, once per molecule/batch.
    """

    def __init__(self, cloud: MolecularFieldCloud):
        super().__init__()
        self.cloud = cloud

    def forward(
        self, initial: Tensor, condition: Tensor, *, steps: int = 4,
        random_samples: int = 4, noise: Tensor | None = None, condition_mask: Tensor | None = None,
    ) -> Tensor:
        if steps < 1 or random_samples < 2:
            raise ValueError("steps must be positive and U-stat needs at least two random paths")
        token_conditioning = isinstance(self.cloud, TokenConditionRelativeCloud)
        if initial.ndim != 4 or condition.ndim != (3 if token_conditioning else 2) or condition.shape[0] != initial.shape[0]:
            raise ValueError("condition must match the cloud's vector/token interface")
        if token_conditioning and (condition_mask is None or condition_mask.shape != condition.shape[:2]):
            raise ValueError("MS tokens require a matching [B,K] validity mask")
        batch = len(initial)
        paths = random_samples + 1
        shape = (batch, steps, paths, self.cloud.noise_token_count, self.cloud.noise_token_dim)
        if noise is None:
            noise = torch.randn(shape, device=initial.device, dtype=initial.dtype)
        elif noise.shape != shape:
            raise ValueError("noise must have shape [B,T,1+M,K,R]")
        # A genuinely deterministic zero-token trajectory at every step.
        noise = torch.cat((torch.zeros_like(noise[:, :, :1]), noise[:, :, 1:]), dim=2)
        cached_condition = token_conditioning and self.cloud.cache_condition_kv
        expanded_condition = condition if cached_condition else condition.repeat_interleave(paths, dim=0)
        first_kwargs = {"condition_mask": condition_mask} if token_conditioning else {}
        later_kwargs = {"condition_mask": (condition_mask if cached_condition else condition_mask.repeat_interleave(paths, dim=0))} if token_conditioning else {}
        if cached_condition:
            prepared = self.cloud.prepare_condition(condition, condition_mask)
            first_kwargs["prepared_condition"] = prepared
            later_kwargs["prepared_condition"] = prepared
        frames = []
        current = initial
        for step in range(steps):
            if step == 0:
                fields, _, _ = self.cloud(current, condition, samples=paths, noise=noise[:, step], **first_kwargs)
            else:
                fields, _, _ = self.cloud(
                    current.flatten(0, 1), expanded_condition, samples=1,
                    noise=noise[:, step].reshape(batch * paths, 1, *noise.shape[-2:]),
                    **later_kwargs,
                )
                fields = fields[:, 0].reshape(batch, paths, *initial.shape[1:])
            current = fields  # Deliberately no detach: later losses train earlier corrections.
            frames.append(fields)
        return torch.stack(frames, dim=1)
