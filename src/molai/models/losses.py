"""Cloud Matching distribution losses for molecular fields."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _atrous_blur(images: Tensor, dilation: int) -> Tensor:
    channels = images.shape[1]
    kernel_1d = images.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    kernel = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel.expand(channels, 1, -1, -1)
    padding = 2 * dilation
    return F.conv2d(images, kernel, padding=padding, dilation=dilation, groups=channels)


def full_band_distance(first: Tensor, second: Tensor, levels: int = 3) -> Tensor:
    """Stride-free multi-band Charbonnier distance for paired image batches."""

    if first.shape != second.shape or first.ndim != 4:
        raise ValueError("distance inputs must share [B,C,H,W] shape")
    first_bands = _full_band_features(first, levels)
    second_bands = _full_band_features(second, levels)
    distances = [
        torch.sqrt((left - right).square() + 1e-6).mean(dim=(1, 2, 3))
        for left, right in zip(first_bands, second_bands, strict=True)
    ]
    return torch.stack(distances).mean(dim=0)


def _full_band_features(images: Tensor, levels: int) -> tuple[Tensor, ...]:
    """Compute a reusable undecimated band pyramid for an image batch."""

    residual = images
    bands = []
    for level in range(levels):
        low = _atrous_blur(residual, 2**level)
        bands.append(residual - low)
        residual = low
    return (*bands, residual)


def _cloud_band_features(cloud: Tensor, levels: int) -> tuple[Tensor, ...]:
    batch, samples = cloud.shape[:2]
    bands = _full_band_features(cloud.flatten(0, 1), levels)
    return tuple(band.reshape(batch, samples, *band.shape[1:]) for band in bands)


def _pairwise_band_distance(
    first: tuple[Tensor, ...], second: tuple[Tensor, ...]
) -> Tensor:
    distances = []
    for left_band, right_band in zip(first, second, strict=True):
        difference = left_band[:, :, None] - right_band[:, None]
        distances.append(
            torch.sqrt(difference.square() + 1e-6).mean(dim=(3, 4, 5))
        )
    return torch.stack(distances).mean(dim=0)


def _pairwise_cloud_distance(first: Tensor, second: Tensor, levels: int) -> Tensor:
    return _pairwise_band_distance(
        _cloud_band_features(first, levels),
        _cloud_band_features(second, levels),
    )


def _off_diagonal_mean(distances: Tensor) -> Tensor:
    """Average square pairwise-distance matrices without their diagonal."""

    if distances.ndim != 3 or distances.shape[1] != distances.shape[2]:
        raise ValueError("self-distance tensor must have shape [B,M,M]")
    samples = distances.shape[1]
    if samples < 2:
        raise ValueError("unbiased energy distance requires at least two samples")
    diagonal = distances.diagonal(dim1=1, dim2=2).sum(dim=1)
    return ((distances.sum(dim=(1, 2)) - diagonal) / (samples * (samples - 1))).mean()


class FullBandEnergyDistance(nn.Module):
    """Energy distance between empirical clouds of field corrections.

    The default U-statistic excludes self-pairs from the within-cloud terms. This
    removes the finite-sample diversity shrinkage of the V-statistic, which is
    especially large for the small sample clouds used during training.

    Distances average scalar Charbonnier differences at each band/pixel. Thus
    matching this score is not a certificate of the full joint spatial law, nor
    of a conditional law when samples from different parent states are pooled.
    """

    def __init__(
        self,
        levels: int = 3,
        include_target_constant: bool = True,
        unbiased: bool = True,
    ) -> None:
        super().__init__()
        self.levels = levels
        self.include_target_constant = include_target_constant
        self.unbiased = unbiased

    def forward(self, predicted: Tensor, target: Tensor, current: Tensor) -> Tensor:
        if predicted.ndim != 5 or target.ndim != 5:
            raise ValueError("clouds must have shape [B,M,C,H,W]")
        predicted_correction = predicted - current[:, None]
        target_correction = target - current[:, None]
        predicted_bands = _cloud_band_features(predicted_correction, self.levels)
        target_bands = _cloud_band_features(target_correction, self.levels)
        cross = _pairwise_band_distance(predicted_bands, target_bands).mean()
        predicted_distances = _pairwise_band_distance(predicted_bands, predicted_bands)
        within_predicted = (
            _off_diagonal_mean(predicted_distances)
            if self.unbiased
            else predicted_distances.mean()
        )
        loss = 2.0 * cross - within_predicted
        if self.include_target_constant:
            target_distances = _pairwise_band_distance(target_bands, target_bands)
            within_target = (
                _off_diagonal_mean(target_distances)
                if self.unbiased
                else target_distances.mean()
            )
            loss = loss - within_target
        return loss


def vp_cloud_geometry_losses(
    fields: Tensor,
    target_mean: Tensor,
    posterior_noise_rms: Tensor,
    random_tokens: Tensor,
    *,
    token_scaled_radius: bool = True,
) -> dict[str, Tensor]:
    """Geometry terms of Navigation loss for per-input VP stochastic innovations.

    ``fields`` and per-path means are [B,T,1+M,C,H,W]; a shared mean of shape
    [B,T,C,H,W] is also accepted for single-input callers. Posterior noise RMS
    is [B,T]. Tokens are the *actual current-step* Random inputs [B,T,M,K,R].
    Each innovation is output minus its own input-conditioned posterior mean,
    equivalently (output-input) minus the analytic posterior drift. Past token
    histories are not used to supervise a new step's stochastic direction.

    Radius is measured about the analytic target mean, never a movable predicted
    center. Each current token RMS has E[r**2]=1 for standard Gaussian tokens.
    This is a geometry regularizer, not a full distribution-matching objective.
    No ensemble centering or antithetic sampling changes the independent Random
    tokens. This does not certify conditional output diversity: pooled parents
    can contribute apparent innovation diversity without use of fresh tokens.
    Clean/zero-variance targets have zero radius and no angle loss. At exactly
    zero offsets both radius and angle gradients can vanish despite positive loss.
    """
    if fields.ndim != 6 or target_mean.shape not in (
        fields.shape, fields.shape[:2] + fields.shape[3:],
    ):
        raise ValueError("fields must be [B,T,1+M,C,H,W]; target_mean must match or share [B,T,C,H,W]")
    batch, steps, paths = fields.shape[:3]
    samples = paths - 1
    if samples < 2 or posterior_noise_rms.shape != (batch, steps):
        raise ValueError("geometry requires M>=2 and posterior_noise_rms [B,T]")
    if random_tokens.ndim != 5 or random_tokens.shape[:3] != (batch, steps, samples):
        raise ValueError("random_tokens must have shape [B,T,M,K,R], excluding Zero")

    # FP32 reductions/Gram matrices remain FP32 even under the trainer's autocast.
    with torch.autocast(device_type=fields.device.type, enabled=False):
        mean = target_mean.detach().float()
        if mean.ndim == 5:
            mean = mean[:, :, None].expand_as(fields)
        sigma = posterior_noise_rms.detach().float()
        offsets = (fields[:, :, 1:].float() - mean[:, :, 1:]).flatten(3)
        zero_errors = (fields[:, :, 0].float() - mean[:, :, 0]).square()
        step_zero = zero_errors.mean(dim=(0, 2, 3, 4))

        tokens = random_tokens.detach().float().flatten(3)
        token_gram = torch.einsum("btid,btjd->btij", tokens, tokens)
        token_square = token_gram.diagonal(dim1=-2, dim2=-1).clamp_min(0)
        token_rms = (token_square / tokens.shape[-1]).sqrt()
        target_radius = sigma[:, :, None].expand(-1, -1, samples)
        if token_scaled_radius:
            target_radius = target_radius * token_rms

        # An exact zero with finite derivative at clean/collapsed predictions.
        epsilon = 1e-6
        offset_rms = (offsets.square().mean(dim=-1) + epsilon**2).sqrt() - epsilon
        step_radius = (offset_rms - target_radius).square().mean(dim=(0, 2))

        output_direction = F.normalize(offsets, dim=-1, eps=epsilon)
        output_cosine = torch.einsum("btid,btjd->btij", output_direction, output_direction)
        token_norm = token_square.sqrt().clamp_min(epsilon)
        token_cosine = token_gram / (token_norm[..., :, None] * token_norm[..., None, :])
        off_diagonal = ~torch.eye(samples, device=fields.device, dtype=torch.bool)
        active = (sigma > 0)[:, :, None, None] & off_diagonal[None, None]
        nonzero_token = token_square > epsilon**2
        active = active & nonzero_token[..., :, None] & nonzero_token[..., None, :]
        angle_error = (output_cosine - token_cosine.clamp(-1, 1)).square()
        step_direction = (angle_error * active).sum(dim=(0, 2, 3)) / active.sum(
            dim=(0, 2, 3)
        ).clamp_min(1)

        return {
            "zero_mean_mse": step_zero.mean(),
            "random_radius": step_radius.mean(),
            "random_direction": step_direction.mean(),
            "step_zero_mean_mse": step_zero,
            "step_random_radius": step_radius,
            "step_random_direction": step_direction,
            "random_offset_rms": offset_rms.detach().mean(),
            "target_offset_rms": target_radius.detach().mean(),
            "direction_active_fraction": (sigma > 0).float().mean(),
        }
