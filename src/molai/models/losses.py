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
    error = first - second
    distances = []
    for level in range(levels):
        low = _atrous_blur(error, 2**level)
        high = error - low
        distances.append(torch.sqrt(high.square() + 1e-6).mean(dim=(1, 2, 3)))
        error = low
    distances.append(torch.sqrt(error.square() + 1e-6).mean(dim=(1, 2, 3)))
    return torch.stack(distances).mean(dim=0)


def _pairwise_cloud_distance(first: Tensor, second: Tensor, levels: int) -> Tensor:
    batch, first_samples = first.shape[:2]
    second_samples = second.shape[1]
    left = first[:, :, None].expand(-1, -1, second_samples, -1, -1, -1)
    right = second[:, None].expand(-1, first_samples, -1, -1, -1, -1)
    flat_distance = full_band_distance(
        left.reshape(-1, *first.shape[2:]),
        right.reshape(-1, *second.shape[2:]),
        levels,
    )
    return flat_distance.reshape(batch, first_samples, second_samples)


class FullBandEnergyDistance(nn.Module):
    """Energy distance between empirical clouds of field corrections."""

    def __init__(self, levels: int = 3) -> None:
        super().__init__()
        self.levels = levels

    def forward(self, predicted: Tensor, target: Tensor, current: Tensor) -> Tensor:
        if predicted.ndim != 5 or target.ndim != 5:
            raise ValueError("clouds must have shape [B,M,C,H,W]")
        predicted_correction = predicted - current[:, None]
        target_correction = target - current[:, None]
        cross = _pairwise_cloud_distance(
            predicted_correction, target_correction, self.levels
        ).mean()
        within_predicted = _pairwise_cloud_distance(
            predicted_correction, predicted_correction, self.levels
        ).mean()
        within_target = _pairwise_cloud_distance(
            target_correction, target_correction, self.levels
        ).mean()
        return 2.0 * cross - within_predicted - within_target
