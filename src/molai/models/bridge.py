"""Analytic VP transition targets for Cloud Matching pretraining."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class BridgeBatch:
    current: Tensor
    target_cloud: Tensor
    current_levels: Tensor
    answer_levels: Tensor


class VPSchedule:
    def __init__(
        self,
        steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device | str = "cpu",
    ) -> None:
        beta = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alpha_bar = torch.cumprod(1.0 - beta, dim=0)

    def sample_training_batch(
        self,
        clean: Tensor,
        samples: int,
        answer_jump: int = 10,
        clean_answer_probability: float = 0.5,
    ) -> BridgeBatch:
        """Sample current fields and exact arbitrary-skip posterior clouds."""

        batch = clean.shape[0]
        device = clean.device
        current_levels = torch.randint(1, len(self.alpha_bar), (batch,), device=device)
        goal_levels = torch.floor(torch.rand(batch, device=device) * current_levels.float()).long()
        answer_levels = (goal_levels - answer_jump).clamp_min(0)
        force_clean = torch.rand(batch, device=device) < clean_answer_probability
        answer_levels = torch.where(force_clean, torch.zeros_like(answer_levels), answer_levels)

        alpha_s = self.alpha_bar[current_levels].view(batch, 1, 1, 1)
        current_noise = torch.randn_like(clean)
        current = alpha_s.sqrt() * clean + (1.0 - alpha_s).sqrt() * current_noise

        alpha_a = self.alpha_bar[answer_levels].clone()
        alpha_a = torch.where(answer_levels.eq(0), torch.ones_like(alpha_a), alpha_a)
        alpha_s_flat = self.alpha_bar[current_levels]
        alpha_s_given_a = alpha_s_flat / alpha_a
        denominator = (1.0 - alpha_s_flat).clamp_min(1e-8)
        clean_coefficient = alpha_a.sqrt() * (1.0 - alpha_s_given_a) / denominator
        current_coefficient = alpha_s_given_a.sqrt() * (1.0 - alpha_a) / denominator
        variance = ((1.0 - alpha_a) * (1.0 - alpha_s_given_a) / denominator).clamp_min(0.0)

        shape = (batch, 1, 1, 1, 1)
        mean = (
            clean_coefficient.view(shape) * clean[:, None]
            + current_coefficient.view(shape) * current[:, None]
        )
        target_noise = torch.randn(
            batch,
            samples,
            *clean.shape[1:],
            device=device,
            dtype=clean.dtype,
        )
        target_cloud = mean + variance.sqrt().view(shape) * target_noise
        target_cloud = torch.where(
            answer_levels.view(batch, 1, 1, 1, 1).eq(0),
            clean[:, None],
            target_cloud,
        )
        return BridgeBatch(current, target_cloud, current_levels, answer_levels)


class GeometricVESchedule:
    """Variance-exploding bridge with exponentially spaced noise magnitudes."""

    def __init__(
        self,
        levels: int = 64,
        sigma_min: float = 0.01,
        sigma_max: float = 0.6,
        device: torch.device | str = "cpu",
    ) -> None:
        if levels < 2:
            raise ValueError("levels must be at least two")
        if not 0.0 < sigma_min < sigma_max:
            raise ValueError("sigma bounds must satisfy 0 < sigma_min < sigma_max")
        noisy_sigmas = torch.exp(
            torch.linspace(math.log(sigma_min), math.log(sigma_max), levels, device=device)
        )
        self.sigmas = torch.cat((torch.zeros(1, device=device), noisy_sigmas))

    def sample_training_batch(
        self,
        clean: Tensor,
        samples: int,
        current_levels: Tensor | None = None,
        answer_jump: int = 8,
        clean_answer_probability: float = 0.5,
    ) -> BridgeBatch:
        """Sample a noisy current image and exact lower-noise posterior cloud."""

        batch = clean.shape[0]
        device = clean.device
        if current_levels is None:
            current_levels = torch.randint(1, len(self.sigmas), (batch,), device=device)
        else:
            current_levels = current_levels.to(device=device, dtype=torch.long)
            if current_levels.shape != (batch,):
                raise ValueError("current_levels must have shape [B]")
            if bool(((current_levels < 1) | (current_levels >= len(self.sigmas))).any()):
                raise ValueError("current_levels are outside the configured noise schedule")

        answer_levels = (current_levels - answer_jump).clamp_min(0)
        force_clean = torch.rand(batch, device=device) < clean_answer_probability
        answer_levels = torch.where(force_clean, torch.zeros_like(answer_levels), answer_levels)

        sigma_s = self.sigmas[current_levels]
        sigma_a = self.sigmas[answer_levels]
        current = clean + sigma_s.view(batch, 1, 1, 1) * torch.randn_like(clean)

        sigma_s_squared = sigma_s.square().clamp_min(1e-12)
        ratio = sigma_a.square() / sigma_s_squared
        shape = (batch, 1, 1, 1, 1)
        mean = (1.0 - ratio).view(shape) * clean[:, None] + ratio.view(shape) * current[:, None]
        variance = (
            sigma_a.square() * (sigma_s_squared - sigma_a.square()) / sigma_s_squared
        ).clamp_min(0.0)
        target_cloud = mean + variance.sqrt().view(shape) * torch.randn(
            batch,
            samples,
            *clean.shape[1:],
            device=device,
            dtype=clean.dtype,
        )
        target_cloud = torch.where(
            answer_levels.view(batch, 1, 1, 1, 1).eq(0),
            clean[:, None],
            target_cloud,
        )
        return BridgeBatch(current, target_cloud, current_levels, answer_levels)
