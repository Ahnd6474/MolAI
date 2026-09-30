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


class FullBandEnergyDistance(nn.Module):
    """Energy distance between empirical clouds of field corrections."""

    def __init__(self, levels: int = 3, include_target_constant: bool = True) -> None:
        super().__init__()
        self.levels = levels
        self.include_target_constant = include_target_constant

    def forward(self, predicted: Tensor, target: Tensor, current: Tensor) -> Tensor:
        if predicted.ndim != 5 or target.ndim != 5:
            raise ValueError("clouds must have shape [B,M,C,H,W]")
        predicted_correction = predicted - current[:, None]
        target_correction = target - current[:, None]
        predicted_bands = _cloud_band_features(predicted_correction, self.levels)
        target_bands = _cloud_band_features(target_correction, self.levels)
        cross = _pairwise_band_distance(predicted_bands, target_bands).mean()
        within_predicted = _pairwise_band_distance(predicted_bands, predicted_bands).mean()
        loss = 2.0 * cross - within_predicted
        if self.include_target_constant:
            within_target = _pairwise_band_distance(target_bands, target_bands).mean()
            loss = loss - within_target
        return loss
