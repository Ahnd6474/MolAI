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
        self.noise_scale = 1.0

    def sample_training_batch(
        self,
        clean: Tensor,
        samples: int,
        current_levels: Tensor | None = None,
        answer_jump: int | Tensor = 10,
        clean_answer_probability: float = 0.0,
    ) -> BridgeBatch:
        """Sample current fields and exact arbitrary-skip posterior clouds."""

        batch = clean.shape[0]
        device = clean.device
        if current_levels is None:
            current_levels = torch.randint(1, len(self.alpha_bar), (batch,), device=device)
        else:
            current_levels = current_levels.to(device=device, dtype=torch.long)
            if current_levels.shape != (batch,):
                raise ValueError("current_levels must have shape [B]")
            if bool(((current_levels < 0) | (current_levels >= len(self.alpha_bar))).any()):
                raise ValueError("current_levels are outside the configured noise schedule")
        jumps = torch.as_tensor(answer_jump, device=device, dtype=torch.long)
        if jumps.ndim == 0:
            jumps = jumps.expand(batch)
        if jumps.shape != (batch,) or bool((jumps < 0).any()):
            raise ValueError("answer_jump must be non-negative and scalar or shape [B]")
        answer_levels = (current_levels - jumps).clamp_min(0)
        force_clean = torch.rand(batch, device=device) < clean_answer_probability
        answer_levels = torch.where(force_clean, torch.zeros_like(answer_levels), answer_levels)

        alpha_s = self.alpha_bar[current_levels].view(batch, 1, 1, 1)
        current_noise = torch.randn_like(clean)
        current = alpha_s.sqrt() * clean + (
            self.noise_scale * (1.0 - alpha_s).sqrt() * current_noise
        )

        target_cloud = self.sample_target_cloud(
            clean,
            current,
            current_levels,
            answer_levels,
            samples,
        )
        return BridgeBatch(current, target_cloud, current_levels, answer_levels)

    def sample_target_cloud(
        self,
        clean: Tensor,
        current: Tensor,
        current_levels: Tensor,
        answer_levels: Tensor,
        samples: int,
    ) -> Tensor:
        """Sample the exact posterior target from a supplied current state."""

        batch = clean.shape[0]
        if current.shape != clean.shape:
            raise ValueError("current and clean must share shape [B,C,H,W]")
        current_levels = current_levels.to(device=clean.device, dtype=torch.long)
        answer_levels = answer_levels.to(device=clean.device, dtype=torch.long)
        if current_levels.shape != (batch,) or answer_levels.shape != (batch,):
            raise ValueError("transition levels must have shape [B]")
        if bool((answer_levels > current_levels).any()):
            raise ValueError("answer levels cannot be noisier than current levels")

        mean, variance = self.target_mean_and_variance(
            clean, current, current_levels, answer_levels
        )
        target_noise = torch.randn(
            batch,
            samples,
            *clean.shape[1:],
            device=clean.device,
            dtype=clean.dtype,
        )
        target_cloud = mean[:, None] + self.noise_scale * variance.sqrt().view(
            batch, 1, 1, 1, 1
        ) * target_noise
        return torch.where(
            answer_levels.view(batch, 1, 1, 1, 1).eq(0),
            clean[:, None],
            target_cloud,
        )

    def target_mean_and_variance(
        self,
        clean: Tensor,
        current: Tensor,
        current_levels: Tensor,
        answer_levels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return the exact arbitrary-skip posterior mean and scalar variance."""

        batch = clean.shape[0]
        alpha_a = self.alpha_bar[answer_levels].clone()
        alpha_a = torch.where(answer_levels.eq(0), torch.ones_like(alpha_a), alpha_a)
        alpha_s_flat = self.alpha_bar[current_levels]
        alpha_s_given_a = alpha_s_flat / alpha_a
        denominator = (1.0 - alpha_s_flat).clamp_min(1e-8)
        clean_coefficient = alpha_a.sqrt() * (1.0 - alpha_s_given_a) / denominator
        current_coefficient = alpha_s_given_a.sqrt() * (1.0 - alpha_a) / denominator
        variance = ((1.0 - alpha_a) * (1.0 - alpha_s_given_a) / denominator).clamp_min(0.0)

        shape = (batch, 1, 1, 1)
        mean = (
            clean_coefficient.view(shape) * clean
            + current_coefficient.view(shape) * current
        )
        mean = torch.where(answer_levels.view(batch, 1, 1, 1).eq(0), clean, mean)
        return mean, variance


class CosineVPSchedule(VPSchedule):
    """Cosine variance-preserving schedule with a pure-noise terminal level."""

    def __init__(
        self,
        levels: int = 64,
        offset: float = 0.008,
        noise_scale: float = 1.0,
        zero_mean_noise: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        if levels < 2:
            raise ValueError("levels must be at least two")
        if not 0.0 <= offset < 1.0:
            raise ValueError("cosine offset must lie in [0, 1)")
        if noise_scale <= 0.0:
            raise ValueError("noise_scale must be positive")
        level = torch.arange(levels + 1, device=device, dtype=torch.float32)
        angle = ((level / levels + offset) / (1.0 + offset)) * (math.pi / 2.0)
        alpha_bar = angle.cos().square()
        alpha_bar = alpha_bar / alpha_bar[0]
        alpha_bar[0] = 1.0
        alpha_bar[-1] = 0.0
        self.alpha_bar = alpha_bar
        self.noise_scale = float(noise_scale)
        self.zero_mean_noise = bool(zero_mean_noise)

    def _center_noise(self, noise: Tensor) -> Tensor:
        if not self.zero_mean_noise:
            return noise
        return noise - noise.mean(dim=(-2, -1), keepdim=True)

    def sample_training_batch(
        self,
        clean: Tensor,
        samples: int,
        current_levels: Tensor | None = None,
        answer_jump: int | Tensor = 8,
        clean_answer_probability: float = 0.0,
    ) -> BridgeBatch:
        batch = clean.shape[0]
        device = clean.device
        if current_levels is None:
            current_levels = torch.randint(1, len(self.alpha_bar), (batch,), device=device)
        else:
            current_levels = current_levels.to(device=device, dtype=torch.long)
            if current_levels.shape != (batch,):
                raise ValueError("current_levels must have shape [B]")
            if bool(((current_levels < 0) | (current_levels >= len(self.alpha_bar))).any()):
                raise ValueError("current_levels are outside the configured noise schedule")
        jumps = torch.as_tensor(answer_jump, device=device, dtype=torch.long)
        if jumps.ndim == 0:
            jumps = jumps.expand(batch)
        if jumps.shape != (batch,) or bool((jumps < 0).any()):
            raise ValueError("answer_jump must be non-negative and scalar or shape [B]")
        answer_levels = (current_levels - jumps).clamp_min(0)
        force_clean = torch.rand(batch, device=device) < clean_answer_probability
        answer_levels = torch.where(force_clean, torch.zeros_like(answer_levels), answer_levels)

        alpha_s = self.alpha_bar[current_levels].view(batch, 1, 1, 1)
        current_noise = self._center_noise(torch.randn_like(clean))
        current = alpha_s.sqrt() * clean + (
            self.noise_scale * (1.0 - alpha_s).sqrt() * current_noise
        )
        target_cloud = self.sample_target_cloud(
            clean,
            current,
            current_levels,
            answer_levels,
            samples,
        )
        return BridgeBatch(current, target_cloud, current_levels, answer_levels)

    def sample_target_cloud(
        self,
        clean: Tensor,
        current: Tensor,
        current_levels: Tensor,
        answer_levels: Tensor,
        samples: int,
    ) -> Tensor:
        batch = clean.shape[0]
        if current.shape != clean.shape:
            raise ValueError("current and clean must share shape [B,C,H,W]")
        current_levels = current_levels.to(device=clean.device, dtype=torch.long)
        answer_levels = answer_levels.to(device=clean.device, dtype=torch.long)
        if current_levels.shape != (batch,) or answer_levels.shape != (batch,):
            raise ValueError("transition levels must have shape [B]")
        if bool((answer_levels > current_levels).any()):
            raise ValueError("answer levels cannot be noisier than current levels")

        mean, variance = self.target_mean_and_variance(
            clean, current, current_levels, answer_levels
        )
        target_noise = self._center_noise(
            torch.randn(
                batch,
                samples,
                *clean.shape[1:],
                device=clean.device,
                dtype=clean.dtype,
            )
        )
        target_cloud = mean[:, None] + self.noise_scale * variance.sqrt().view(
            batch, 1, 1, 1, 1
        ) * target_noise
        return torch.where(
            answer_levels.view(batch, 1, 1, 1, 1).eq(0),
            clean[:, None],
            target_cloud,
        )


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
        clean_answer_probability: float = 0.0,
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
