"""Train a fresh or restored relative Cloud with differentiable image feedback.

--prepare-only validates forward/backward without optimizer updates or training
checkpoints. The source checkpoint controls the image backbone; a separate YAML controls
the new fine-tuning phase. Zero + four random paths persist without tree branching.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from relative_condition_alignment import run_alignment
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from train_absolute_cloud import RaggedSpectrumTokenRunner, _postpool_peak_token_counts
from train_cloud import (
    RaggedSpectrumConditionRunner,
    _condition_from_batch,
    _distributed_mean,
    _make_schedule,
    _transition_jumps,
    _update_latest,
)

from molai.data import ShardShuffleSampler, SpectrumFieldDataset, collate_spectrum_field_batch
from molai.models.bridge import CosineVPSchedule
from molai.models.condition import bind_spectrum_metadata
from molai.models.losses import FullBandEnergyDistance, vp_cloud_geometry_losses
from molai.models.parallel_smiles import (
    ParallelFieldToSmiles,
    ParallelSmilesOutput,
    parallel_smiles_loss,
)
from molai.models.relative_rollout import (
    RelativeImageRolloutCloud,
    build_fresh_token_relative_models,
    build_ms_reset_relative_models,
    build_relative_models,
    build_token_relative_models,
    configure_conditional_gaussian_tokens,
    configure_relative_image_grids,
    configure_relative_level_conditioning,
    expand_relative_refine_depth,
    configure_relative_execution,
    restore_token_relative_models,
)
from molai.models.smiles import SmilesTokenizer


def condition_arguments(condition):
    if isinstance(condition, tuple):
        return {"condition": condition[0], "condition_mask": condition[1]}
    return {"condition": condition}


def level_arguments(cloud, start, answer_levels=None, *, steps=None):
    if getattr(cloud, "level_conditioner", None) is None:
        return {}
    before = (torch.cat((start[:, None], answer_levels[:, :-1]), dim=1)
              if answer_levels is not None else
              (start[:, None] - torch.arange(steps, device=start.device)).clamp_min(0))
    return {"current_levels": before}


def optimizer_parameter_names(cloud, encoder, text_model):
    groups = [[f"cloud.{name}" for name, p in cloud.named_parameters() if p.requires_grad]
              + [f"condition_encoder.{name}" for name, p in encoder.named_parameters() if p.requires_grad]]
    if text_model is not None:
        groups.append([f"parallel_smiles.{name}" for name, p in text_model.named_parameters() if p.requires_grad])
    return groups


def restore_optimizer_adding_level_parameters(optimizer, saved, names, source_names):
    """Retain named Adam states; allow only new time/CvT layers and zero-padded FiLM rows."""
    state = optimizer.state_dict()
    if len(saved["param_groups"]) != len(state["param_groups"]):
        raise ValueError("optimizer group count changed")
    remapped = {"state": {}, "param_groups": []}
    added, expanded = [], []
    for old_group, group, group_names, old_names in zip(
        saved["param_groups"], state["param_groups"], names, source_names, strict=True,
    ):
        if len(old_names) != len(old_group["params"]) or len(group_names) != len(group["params"]):
            raise ValueError("optimizer parameter-name inventory does not match saved groups")
        lookup = dict(zip(group_names, group["params"], strict=True))
        if set(old_names) - set(lookup):
            raise ValueError("optimizer migration would drop a trained parameter")
        new_names = set(group_names) - set(old_names)
        old_blocks = [int(name.split(".")[2]) for name in old_names if name.startswith("cloud.refine_blocks.")]
        old_depth = max(old_blocks, default=-1) + 1
        def allowed(name):
            return (name.startswith("cloud.level_conditioner.") or
                    (name.startswith("cloud.refine_blocks.") and int(name.split(".")[2]) >= old_depth))
        if any(not allowed(name) for name in new_names):
            raise ValueError("only new level-conditioning or appended refine parameters may lack optimizer state")
        added.extend(sorted(new_names))
        current_parameters = dict(zip(group_names, optimizer.param_groups[len(remapped["param_groups"])]["params"], strict=True))
        for name, old_id in zip(old_names, old_group["params"], strict=True):
            if old_id in saved["state"]:
                old_state = saved["state"][old_id]
                parameter = current_parameters[name]
                moment = old_state.get("exp_avg")
                if moment is not None and moment.shape != parameter.shape:
                    if (name not in ("cloud.level_conditioner.mlp.2.weight", "cloud.level_conditioner.mlp.2.bias")
                            or moment.ndim != parameter.ndim or moment.shape[1:] != parameter.shape[1:]
                            or moment.shape[0] >= parameter.shape[0]):
                        raise ValueError("only appended level-modulation rows may expand saved Adam moments")
                    old_state = dict(old_state)
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        if key in old_state:
                            padded = old_state[key].new_zeros(parameter.shape)
                            padded[:moment.shape[0]].copy_(old_state[key])
                            old_state[key] = padded
                    expanded.append({"name": name, "previous_shape": list(moment.shape),
                                     "new_shape": list(parameter.shape), "new_moments": "zero"})
                remapped["state"][lookup[name]] = old_state
        remapped["param_groups"].append({**old_group, "params": group["params"]})
    optimizer.load_state_dict(remapped)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            moment = optimizer.state.get(parameter, {}).get("exp_avg")
            if moment is not None and moment.shape != parameter.shape:
                raise ValueError("restored Adam moment shape does not match its parameter")
    return {"restored_parameter_states": len(remapped["state"]), "fresh_parameters": added,
            "expanded_parameter_states": expanded}


def apply_transition_schedule(config, options):
    """Explicit phase override: never silently inherit an obsolete jump schedule."""
    previous = copy.deepcopy(config["cloud_matching"].get("transition_jumps"))
    requested = options.get("transition_jumps")
    if requested is not None:
        if not isinstance(requested, list) or not requested:
            raise ValueError("transition_jumps override must be a non-empty list")
        candidate = copy.deepcopy(config)
        candidate["cloud_matching"]["transition_jumps"] = copy.deepcopy(requested)
        maximum = int(candidate["cloud_matching"]["noise_schedule"]["levels"])
        _transition_jumps(torch.arange(maximum + 1), candidate)
        config["cloud_matching"]["transition_jumps"] = copy.deepcopy(requested)
    return {"previous": previous,
            "effective": copy.deepcopy(config["cloud_matching"].get("transition_jumps")),
            "overridden": requested is not None}


def make_lr_scheduler(optimizer, options, updates, *, global_step=0):
    """Absolute-step schedule, optionally re-anchored without resetting Adam moments."""
    peak = float(options["learning_rate"])
    minimum = float(options["min_learning_rate"]) / peak
    warmup = int(options["warmup_steps"])
    anchor = int(options.get("lr_schedule_start_step", 0))
    start_lr = options.get("warmup_start_learning_rate")
    if not 0 <= anchor < updates or not 0 < minimum <= 1 or warmup < 0:
        raise ValueError("invalid LR schedule horizon, minimum or warmup")
    if start_lr is not None and not 0 < float(start_lr) <= peak:
        raise ValueError("warmup start LR must be positive and no greater than peak LR")
    if anchor and start_lr is None:
        raise ValueError("re-anchored schedule requires warmup_start_learning_rate")
    if not anchor <= global_step <= updates:
        raise ValueError("global step falls outside the LR schedule")

    def factor(step):
        elapsed = max(0, step - anchor)
        if elapsed < warmup:
            if start_lr is None:
                return (elapsed + 1) / max(1, warmup)
            initial = float(start_lr) / peak
            return initial + (1 - initial) * elapsed / max(1, warmup)
        progress = min(1., (elapsed - warmup) / max(1, updates - anchor - warmup))
        return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * progress))

    # load_state_dict preserves moments and parameter order. Only LR is changed.
    for group in optimizer.param_groups:
        group["initial_lr"] = float(group.get("schedule_peak_lr", peak))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor, last_epoch=global_step - 1)


def add_parallel_smiles_supervision(
    losses: dict[str, Tensor],
    text_model,
    random_fields: Tensor,
    token_ids: Tensor,
    options: dict,
) -> tuple[dict[str, Tensor], ParallelSmilesOutput]:
    """Decode every Random branch independently and add its token objective."""

    output = text_model(random_fields)
    text = parallel_smiles_loss(
        output,
        token_ids,
        pad_token_id=int(options["smiles_pad_token_id"]),
        length_weight=float(options.get("smiles_length_loss_weight", 0.1)),
    )
    image_total = losses["total"]
    combined = dict(losses)
    combined.update(
        image_total=image_total,
        smiles_total=text["total"],
        smiles_token_cross_entropy=text["token_cross_entropy"],
        smiles_length_cross_entropy=text["length_cross_entropy"],
        total=image_total + float(options["smiles_loss_weight"]) * text["total"],
    )
    return combined, output


@torch.no_grad()
def parallel_smiles_metrics(
    output: ParallelSmilesOutput,
    token_ids: Tensor,
    pad_token_id: int,
) -> dict[str, Tensor]:
    """Token, per-branch exact, best-branch exact and length accuracy."""

    targets = token_ids[:, 1:]
    slots = output.logits.shape[-2]
    padded = token_ids.new_full((len(token_ids), slots), pad_token_id)
    padded[:, : targets.shape[1]] = targets
    if output.logits.ndim == 4:
        padded = padded[:, None].expand(-1, output.logits.shape[1], -1)
    predicted = output.logits.argmax(dim=-1)
    valid = padded.ne(pad_token_id)
    correct = predicted.eq(padded)
    token_accuracy = (correct & valid).sum() / valid.sum().clamp_min(1)
    exact = (correct | ~valid).all(dim=-1).float()
    if exact.ndim == 2:
        branch_exact = exact.mean()
        best_exact = exact.amax(dim=1).mean()
    else:
        branch_exact = best_exact = exact.mean()
    lengths = targets.ne(pad_token_id).sum(dim=-1).clamp_max(slots)
    if output.length_logits.ndim == 3:
        lengths = lengths[:, None].expand(-1, output.length_logits.shape[1])
    length_accuracy = output.length_logits.argmax(dim=-1).eq(lengths).float().mean()
    return {
        "smiles_token_accuracy": token_accuracy,
        "smiles_exact_match": branch_exact,
        "smiles_best_of_4_exact_match": best_exact,
        "smiles_length_accuracy": length_accuracy,
    }


@torch.no_grad()
def rollout_initial_and_levels(schedule, clean: Tensor, start: Tensor, steps: int, config):
    """Sample one input and plan levels without generating per-step target clouds."""
    initial = schedule.sample_training_batch(clean, 1, current_levels=start, answer_jump=0).current
    current, levels = start, []
    for _ in range(steps):
        current = (current - _transition_jumps(current, config)).clamp_min(0)
        levels.append(current)
    return initial, torch.stack(levels, dim=1)


@torch.no_grad()
def step_posterior_targets(schedule, clean, initial, fields, start, levels, *, sample_innovations=True):
    """Detached VP transition teachers conditioned on each path's actual input.

    Return optional innovation samples [B,T,M,C,H,W], per-path means
    [B,T,1+M,C,H,W], and centered-noise RMS [B,T]. Navigation disables reference
    sampling entirely; the older U-stat objective can still request samples.
    Only teacher construction is detached; the model's image feedback is not.
    """
    if fields.ndim != 6 or fields.shape[0] != len(clean) or fields.shape[3:] != clean.shape[1:]:
        raise ValueError("fields must have shape [B,T,1+M,C,H,W] matching clean")
    batch, steps, paths = fields.shape[:3]
    if levels.shape != (batch, steps) or start.shape != (batch,) or initial.shape != clean.shape:
        raise ValueError("initial/start/levels must match [B,C,H,W]/[B]/[B,T]")
    with torch.autocast(device_type=fields.device.type, enabled=False):
        inputs = torch.cat((initial.float()[:, None, None].expand(-1, 1, paths, -1, -1, -1),
                            fields.detach().float()[:, :-1]), dim=1)
        before = torch.cat((start[:, None], levels[:, :-1]), dim=1)
        truth = clean.float()[:, None, None].expand_as(inputs)
        before = before[:, :, None].expand(batch, steps, paths).reshape(-1)
        after = levels[:, :, None].expand(batch, steps, paths).reshape(-1)
        mean, variance = schedule.target_mean_and_variance(
            truth.reshape(-1, *clean.shape[1:]), inputs.flatten(0, 2), before, after,
        )
        mean = mean.reshape_as(inputs)
        # The variance depends only on levels, not the branch's current image.
        variance = variance.reshape(batch, steps, paths)[:, :, 0]
        innovations = None
        if sample_innovations:
            innovations = torch.randn(batch, steps, paths - 1, *clean.shape[1:],
                                      device=fields.device, dtype=torch.float32)
            if schedule.zero_mean_noise:
                innovations -= innovations.mean(dim=(-2, -1), keepdim=True)
            innovations *= schedule.noise_scale * variance.sqrt()[:, :, None, None, None, None]
        pixels = clean.shape[-2] * clean.shape[-1]
        centering = 1.0 - 1.0 / pixels if schedule.zero_mean_noise else 1.0
        noise_rms = schedule.noise_scale * (variance * centering).sqrt()
    return innovations, mean, noise_rms


@torch.no_grad()
def current_mean_global_vp_targets(schedule, clean, start, levels, samples):
    """Zero targets this level's signal; Random targets the next global marginal.

    Teachers never depend on predictions. Spatially centered Gaussian reference
    images are independent across branches and steps. No posterior is used.
    """
    before = torch.cat((start[:, None], levels[:, :-1]), dim=1)
    alpha_before, alpha_after = schedule.alpha_bar[before], schedule.alpha_bar[levels]
    truth = clean.detach().float()[:, None]
    zero_mean = alpha_before.sqrt()[..., None, None, None] * truth
    random_mean = alpha_after.sqrt()[..., None, None, None] * truth
    means = torch.cat((zero_mean[:, :, None],
                       random_mean[:, :, None].expand(-1, -1, samples, -1, -1, -1)), dim=2)
    epsilon = torch.randn(len(clean), levels.shape[1], samples, *clean.shape[1:],
                          device=clean.device, dtype=torch.float32)
    if schedule.zero_mean_noise:
        epsilon -= epsilon.mean(dim=(-2, -1), keepdim=True)
    sigma = schedule.noise_scale * (1.0 - alpha_after).clamp_min(0).sqrt()
    targets = random_mean[:, :, None] + sigma[..., None, None, None, None] * epsilon
    pixels = clean.shape[-2] * clean.shape[-1]
    correction = 1.0 - 1.0 / pixels if schedule.zero_mean_noise else 1.0
    return targets, means, sigma * math.sqrt(correction)


def current_mean_global_vp_losses(fields, targets, means, radius, distance, options):
    """Exactly two terms: current-signal Zero MSE + next-global Random U-stat."""
    with torch.autocast(device_type=fields.device.type, enabled=False):
        values = fields.float()
        step_zero = (values[:, :, 0] - means[:, :, 0]).square().mean(dim=(0, 2, 3, 4))
        centered = values[:, :, 1:] - means[:, :, 1:]
        rms = (centered.square().mean(dim=(3, 4, 5)) + 1e-12).sqrt() - 1e-6
        origin = torch.zeros_like(fields[:, 0, 0], dtype=torch.float32)
        step_ustat = torch.stack([distance(values[:, step, 1:], targets[:, step], origin)
                                 for step in range(fields.shape[1])])
        diversity = values[:, :, 1:].detach().var(dim=2, unbiased=False).mean()
        samples = fields.shape[2] - 1
        target_diversity = radius.float().square().mean() * (samples - 1) / samples
        total = (float(options["zero_current_mean_mse_weight"]) * step_zero.mean()
                 + float(options["u_statistic_weight"]) * step_ustat.mean())
    return {"total": total, "zero_current_mean_mse": step_zero.mean(),
            "u_statistic": step_ustat.mean(),
            "random_offset_rms": rms.detach().mean(), "target_offset_rms": radius.detach().mean(),
            "diversity": diversity, "target_diversity": target_diversity,
            "step_zero_current_mean_mse": step_zero, "step_u_statistic": step_ustat}


def trajectory_targets(
    schedule, clean: Tensor, start: Tensor, steps: int, samples: int, config,
    *, return_moments: bool = False,
):
    """Exact level marginals conditioned on the original input, never a prediction."""
    initial = schedule.sample_training_batch(clean, 1, current_levels=start, answer_jump=0).current
    current = start
    clouds, levels, means, noise_rms = [], [], [], []
    for _ in range(steps):
        current = (current - _transition_jumps(current, config)).clamp_min(0)
        levels.append(current)
        clouds.append(schedule.sample_target_cloud(clean, initial, start, current, samples))
        if return_moments:
            mean, variance = schedule.target_mean_and_variance(clean, initial, start, current)
            means.append(mean)
            # Noise is centered independently in each image channel. Its expected
            # per-pixel variance is reduced by (HW-1)/HW, not by the token count.
            pixels = clean.shape[-2] * clean.shape[-1]
            centering = 1.0 - 1.0 / pixels if schedule.zero_mean_noise else 1.0
            noise_rms.append(schedule.noise_scale * (variance * centering).sqrt())
    if return_moments:
        return (initial, torch.stack(clouds, dim=1), torch.stack(levels, dim=1),
                torch.stack(means, dim=1), torch.stack(noise_rms, dim=1))
    return initial, torch.stack(clouds, dim=1), torch.stack(levels, dim=1)


def rollout_losses(fields: Tensor, targets: Tensor, initial: Tensor, clean: Tensor, distance, zero_weight):
    """Legacy clean-Zero objective retained for old-checkpoint diagnostics."""
    u_stats, zero_mses = [], []
    for step in range(fields.shape[1]):
        u_stats.append(distance(fields[:, step, 1:], targets[:, step], initial))
        zero_mses.append(F.mse_loss(fields[:, step, 0].float(), clean.float()))
    u_stats, zero_mses = torch.stack(u_stats), torch.stack(zero_mses)
    with torch.no_grad():
        diversity = fields[:, :, 1:].float().var(dim=2, unbiased=False).mean()
        target_diversity = targets.float().var(dim=2, unbiased=False).mean()
    return {
        "total": (u_stats + zero_weight * zero_mses).mean(),
        "u_statistic": u_stats.mean(), "zero_token_clean_mse": zero_mses.mean(),
        "diversity": diversity, "target_diversity": target_diversity,
        "step_u_statistic": u_stats, "step_zero_token_clean_mse": zero_mses,
    }


def loss_metric_keys(options):
    mode = options.get("loss_mode", "legacy")
    if mode == "legacy":
        return ("total", "u_statistic", "zero_token_clean_mse")
    if mode == "zero_current_global_vp_ustat":
        return ("total", "zero_current_mean_mse", "u_statistic",
                "random_offset_rms", "target_offset_rms")
    if mode not in ("navigation", "vp_transition_geometry"):
        raise ValueError("loss_mode must be legacy, navigation, vp_transition_geometry or zero_current_global_vp_ustat")
    statistic = ("u_statistic",) if mode == "vp_transition_geometry" else ()
    return ("total", *statistic, "zero_mean_mse", "random_radius", "random_direction",
            "random_offset_rms", "target_offset_rms", "direction_active_fraction")


def sample_rollout_noise(reference, cloud, steps, samples):
    """Keep the actual sampled tokens available to the geometry objective."""
    noise = torch.randn(len(reference), steps, samples + 1, cloud.noise_token_count,
                        cloud.noise_token_dim, device=reference.device, dtype=reference.dtype)
    noise[:, :, 0].zero_()
    return noise


def configured_rollout_losses(
    fields, targets, initial, clean, distance, options, target_mean, posterior_noise_rms, noise,
):
    """Navigation is exactly VP mean MSE + stochastic radius + direction.

    The older vp_transition_geometry mode retains U-stat for old-run diagnostics.

    This is a training objective, not a certificate of conditional diversity or
    rollout contraction. See audit_navigation_loss.py for explicit counterexamples.
    """
    if options.get("loss_mode", "legacy") == "legacy":
        return rollout_losses(fields, targets, initial, clean, distance,
                              float(options["zero_token_clean_mse_weight"]))
    if options.get("loss_mode") == "zero_current_global_vp_ustat":
        return current_mean_global_vp_losses(fields, targets, target_mean,
                                            posterior_noise_rms, distance, options)
    if target_mean.shape != fields.shape:
        raise ValueError("transition objective requires a separate posterior mean for every input path")
    geometry = vp_cloud_geometry_losses(
        fields, target_mean, posterior_noise_rms, noise[:, :, 1:],
        token_scaled_radius=bool(options.get("radius_use_token_rms", True)),
    )
    if options.get("loss_mode") == "navigation":
        total = (float(options["zero_mean_mse_weight"]) * geometry["zero_mean_mse"]
                 + float(options["random_radius_weight"]) * geometry["random_radius"]
                 + float(options["random_direction_weight"]) * geometry["random_direction"])
        with torch.no_grad():
            innovations = fields[:, :, 1:].detach().float() - target_mean[:, :, 1:].detach().float()
            diversity = innovations.var(dim=2, unbiased=False).mean()
            samples = fields.shape[2] - 1
            # Analytic expected finite-ensemble variance; no target cloud sampling.
            target_diversity = posterior_noise_rms.float().square().mean() * (samples - 1) / samples
        return {"total": total, **geometry, "diversity": diversity, "target_diversity": target_diversity}
    # Different parents have different conditional means. Remove each one's
    # teacher drift before comparing their common zero-mean innovation law.
    innovations = fields[:, :, 1:].float() - target_mean[:, :, 1:].detach().float()
    origin = torch.zeros_like(initial, dtype=torch.float32)
    u_stats = torch.stack([
        distance(innovations[:, step], targets[:, step].detach(), origin)
        for step in range(fields.shape[1])
    ])
    total = (float(options.get("u_statistic_weight", 1.0)) * u_stats.mean()
             + float(options["zero_mean_mse_weight"]) * geometry["zero_mean_mse"]
             + float(options["random_radius_weight"]) * geometry["random_radius"]
             + float(options["random_direction_weight"]) * geometry["random_direction"])
    with torch.no_grad():
        diversity = innovations.var(dim=2, unbiased=False).mean()
        target_diversity = targets.float().var(dim=2, unbiased=False).mean()
    return {"total": total, "u_statistic": u_stats.mean(), **geometry,
            "diversity": diversity, "target_diversity": target_diversity,
            "step_u_statistic": u_stats}


def color_preview(clean: Tensor, initial: Tensor, fields: Tensor) -> Tensor:
    """GT / input / final Zero / Random 1-4, one common color scale per molecule."""
    import matplotlib

    panels = torch.cat((clean[:, None], initial[:, None], fields[:, -1]), dim=1)
    rows = []
    for row in panels[:4].detach().float().cpu():
        scale = row.abs().flatten().quantile(0.995).clamp_min(1e-6)
        value = ((row[:, 0] / scale).clamp(-1, 1) + 1) / 2
        rgb = torch.from_numpy(matplotlib.colormaps["coolwarm"](value.numpy())[..., :3]).float()
        rows.append(torch.cat(list(rgb.permute(0, 3, 1, 2)), dim=-1))
    return torch.stack(rows)


def sampled_validation_levels(schedule, batch_size, device, options):
    mode = options.get("validation_start_levels", "pure_noise")
    if mode == "uniform":
        lower = int(options.get("validation_min_start_level", 0))
        upper = int(options.get("validation_max_start_level", len(schedule.alpha_bar) - 1))
        if not 0 <= lower <= upper < len(schedule.alpha_bar):
            raise ValueError("validation level bounds must fit within the VP schedule")
        return torch.randint(lower, upper + 1, (batch_size,), device=device)
    if mode == "pure_noise":
        return torch.full((batch_size,), len(schedule.alpha_bar) - 1, device=device)
    raise ValueError("validation_start_levels must be uniform or pure_noise")


def training_level_probabilities(schedule, device, options):
    mixture = options.get("training_level_mixture")
    difficulty = options.get("training_level_difficulty")
    if difficulty is not None:
        if mixture is not None or options.get("training_start_level") is not None:
            raise ValueError("training_level_difficulty requires no mixture or fixed start")
        floor = float(difficulty.get("signal_floor", .01))
        uniform = float(difficulty.get("uniform_fraction", .2))
        power = float(difficulty.get("power", 1.))
        if not math.isfinite(floor) or floor <= 0 or not 0 < uniform <= 1:
            raise ValueError("difficulty signal_floor must be positive and uniform_fraction in (0, 1]")
        if not math.isfinite(power) or power <= 0:
            raise ValueError("difficulty power must be finite and positive")
        alpha = schedule.alpha_bar.detach().float().to(device)
        weights = (alpha + floor).pow(-power)
        return uniform / len(alpha) + (1 - uniform) * weights / weights.sum()
    if mixture is None:
        return None
    if options.get("training_start_level") is not None:
        raise ValueError("training_level_mixture cannot be combined with training_start_level")
    weights = [0.0] * len(schedule.alpha_bar)
    for component in mixture:
        lower, upper = int(component["min_level"]), int(component["max_level"])
        probability = float(component["probability"])
        if not 0 <= lower <= upper < len(weights):
            raise ValueError("training mixture level bounds must fit within the VP schedule")
        if not math.isfinite(probability) or probability < 0:
            raise ValueError("training mixture probabilities must be finite and non-negative")
        for level in range(lower, upper + 1):
            weights[level] += probability / (upper - lower + 1)
    total = sum(weights)
    if total <= 0:
        raise ValueError("training mixture must have positive total probability")
    return torch.tensor([weight / total for weight in weights], device=device)


def sampled_training_levels(schedule, batch_size, device, options, probabilities=None):
    if probabilities is None:
        probabilities = training_level_probabilities(schedule, device, options)
    if probabilities is not None:
        return torch.multinomial(probabilities, batch_size, replacement=True)
    fixed = options.get("training_start_level")
    if fixed is None:
        return torch.randint(0, len(schedule.alpha_bar), (batch_size,), device=device)
    level = int(fixed)
    if not 0 <= level < len(schedule.alpha_bar):
        raise ValueError("training_start_level must fit within the VP schedule")
    return torch.full((batch_size,), level, device=device, dtype=torch.long)


@torch.no_grad()
def long_rollout_preview(rollout, condition_runner, loader, schedule, options, device):
    """Rank-0, fixed small subset only; never contributes to validation scores."""
    samples, steps = int(options.get("long_rollout_samples", 0)), int(options.get("long_rollout_steps", 0))
    if samples <= 0 or steps <= 0:
        return None
    rollout.eval()
    condition_runner.eval()
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(923001)
        batch = next(iter(loader))
        clean = batch["field"].to(device, non_blocking=True)[:samples]
        start = torch.full((len(clean),), len(schedule.alpha_bar) - 1, device=device)
        initial = schedule.sample_training_batch(clean, 1, current_levels=start, answer_jump=0).current
        noise = sample_rollout_noise(initial, rollout.cloud, steps, int(options["random_samples"]))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            condition = _condition_from_batch(condition_runner, batch, device)
            condition = tuple(value[:len(clean)] for value in condition) if isinstance(condition, tuple) else condition[:len(clean)]
            fields = rollout(initial, **condition_arguments(condition), steps=steps,
                             random_samples=int(options["random_samples"]), noise=noise,
                             **level_arguments(rollout.cloud, start, steps=steps))
        if options.get("loss_mode") == "zero_current_global_vp_ustat":
            # Zero estimates THIS level's signal and is not a transition sampler.
            # Long generation previews therefore show only the Random paths.
            fields = fields[:, :, 1:]
        checkpoints = (1, 2, 4, 8, 12, 16, steps) if steps <= 23 else (1, 2, 4, 8, 16, 32, 48, steps)
        selected = sorted({step for step in checkpoints if step <= steps})
        # One image per molecule/path: GT, initial, then selected rollout states.
        panels = torch.cat((clean[:, None, None].expand(-1, fields.shape[2], 1, -1, -1, -1),
                            initial[:, None, None].expand(-1, fields.shape[2], 1, -1, -1, -1),
                            fields[:, [step - 1 for step in selected]].transpose(1, 2).float()), dim=2)
        import matplotlib

        rows = []
        for molecule in panels.float().cpu():
            scale = molecule.abs().flatten().quantile(0.995).clamp_min(1e-6)
            value = ((molecule[:, :, 0] / scale).clamp(-1, 1) + 1) / 2
            rgb = torch.from_numpy(matplotlib.colormaps["coolwarm"](value.numpy())[..., :3]).float()
            rows.extend(torch.cat(list(path.permute(0, 3, 1, 2)), dim=-1) for path in rgb)
        return torch.stack(rows), selected


def save_long_rollout_preview(preview, output, epoch, paths=5):
    if preview is None:
        return
    import matplotlib.pyplot as plt

    images, selected = preview
    directory = output / "long_rollouts" / f"epoch-{epoch + 1:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    # One PNG per molecule, keeping individual files small enough to display.
    for molecule, path_images in enumerate(images.split(paths), start=1):
        image = torch.cat(list(path_images), dim=1).permute(1, 2, 0).numpy()
        plt.imsave(directory / f"molecule-{molecule:02d}_GT_input_steps-{'-'.join(map(str, selected))}.png", image)


@torch.no_grad()
def condition_control_metrics(
    rollout, condition_runner, loader, schedule, options, device, text_model=None,
):
    """Small fixed-seed long-rollout diagnostic: true vs within-batch shuffled MS.

    Both conditions receive exactly the same initial image and random tokens.
    This is a paired diagnostic on a few validation molecules, not full-set scoring.
    """
    rollout.eval()
    condition_runner.eval()
    if text_model is not None:
        text_model.eval()
    keys = ["true_random_correlation", "shuffled_random_correlation",
            "true_random_delta_correlation", "shuffled_random_delta_correlation",
            "true_random_residual_correlation", "shuffled_random_residual_correlation",
            "true_random_mse", "shuffled_random_mse",
            "true_zero_correlation", "shuffled_zero_correlation",
            "true_zero_delta_correlation", "shuffled_zero_delta_correlation",
            "true_zero_residual_correlation", "shuffled_zero_residual_correlation"]
    if text_model is not None:
        keys.extend(("true_smiles_token_cross_entropy", "shuffled_smiles_token_cross_entropy",
                     "true_smiles_token_accuracy", "shuffled_smiles_token_accuracy",
                     "true_smiles_exact_match", "shuffled_smiles_exact_match",
                     "true_smiles_best_of_4_exact_match", "shuffled_smiles_best_of_4_exact_match"))
    totals = dict.fromkeys(keys, 0.)
    count = 0
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(812451 + int(os.environ.get("RANK", "0")))
        for index, batch in enumerate(loader):
            if index >= int(options.get("condition_control_batches", 2)):
                break
            clean = batch["field"].to(device, non_blocking=True)
            if len(clean) < 2:
                continue
            start = torch.full((len(clean),), len(schedule.alpha_bar) - 1, device=device, dtype=torch.long)
            initial = schedule.sample_training_batch(clean, 1, current_levels=start, answer_jump=0).current
            steps, samples = int(options["long_rollout_steps"]), int(options["random_samples"])
            noise = sample_rollout_noise(initial, rollout.cloud, steps, samples)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                condition = _condition_from_batch(condition_runner, batch, device)
                if not isinstance(condition, tuple):
                    raise TypeError("condition control currently requires MS token conditioning")
                shuffled = tuple(value.roll(1, dims=0) for value in condition)
                for name, current_condition in (("true", condition), ("shuffled", shuffled)):
                    final = rollout(initial, **condition_arguments(current_condition), steps=steps,
                                    random_samples=samples, noise=noise,
                                    **level_arguments(rollout.cloud, start, steps=steps))[:, -1].float()
                    centered = final.flatten(2) - final.flatten(2).mean(dim=2, keepdim=True)
                    truth = clean.float().flatten(1)
                    truth = truth - truth.mean(dim=1, keepdim=True)
                    correlation = (centered * truth[:, None]).sum(dim=2) / (
                        centered.norm(dim=2) * truth.norm(dim=1)[:, None]).clamp_min(1e-12)
                    # The absolute final image still contains the common initial
                    # state.  Compare only the cumulative model correction with
                    # the correction required to reach the clean target.
                    delta = final.float() - initial.float()[:, None]
                    target_delta = clean.float() - initial.float()
                    centered_delta = delta.flatten(2) - delta.flatten(2).mean(dim=2, keepdim=True)
                    centered_target_delta = target_delta.flatten(1)
                    centered_target_delta = (
                        centered_target_delta - centered_target_delta.mean(dim=1, keepdim=True)
                    )
                    delta_correlation = (centered_delta * centered_target_delta[:, None]).sum(dim=2) / (
                        centered_delta.norm(dim=2) * centered_target_delta.norm(dim=1)[:, None]
                    ).clamp_min(1e-12)
                    residual_correlation = (centered_delta * truth[:, None]).sum(dim=2) / (
                        centered_delta.norm(dim=2) * truth.norm(dim=1)[:, None]
                    ).clamp_min(1e-12)
                    totals[f"{name}_random_correlation"] += float(correlation[:, 1:].mean()) * len(clean)
                    totals[f"{name}_zero_correlation"] += float(correlation[:, 0].mean()) * len(clean)
                    totals[f"{name}_random_delta_correlation"] += (
                        float(delta_correlation[:, 1:].mean()) * len(clean)
                    )
                    totals[f"{name}_zero_delta_correlation"] += (
                        float(delta_correlation[:, 0].mean()) * len(clean)
                    )
                    totals[f"{name}_random_residual_correlation"] += (
                        float(residual_correlation[:, 1:].mean()) * len(clean)
                    )
                    totals[f"{name}_zero_residual_correlation"] += (
                        float(residual_correlation[:, 0].mean()) * len(clean)
                    )
                    totals[f"{name}_random_mse"] += float((final[:, 1:] - clean[:, None]).square().mean()) * len(clean)
                    if text_model is not None:
                        token_ids = batch["token_ids"].to(device, non_blocking=True)
                        text_output = text_model(final[:, 1:])
                        text_loss = parallel_smiles_loss(
                            text_output, token_ids,
                            pad_token_id=int(options["smiles_pad_token_id"]),
                            length_weight=float(options.get("smiles_length_loss_weight", 0.1)),
                        )
                        text_metrics = parallel_smiles_metrics(
                            text_output, token_ids, int(options["smiles_pad_token_id"]),
                        )
                        totals[f"{name}_smiles_token_cross_entropy"] += (
                            float(text_loss["token_cross_entropy"]) * len(clean)
                        )
                        for key in ("smiles_token_accuracy", "smiles_exact_match",
                                    "smiles_best_of_4_exact_match"):
                            totals[f"{name}_{key}"] += float(text_metrics[key]) * len(clean)
                    del final, centered, correlation, delta, target_delta
                    del centered_delta, centered_target_delta, delta_correlation, residual_correlation
            count += len(clean)
    result = {key: _distributed_mean(value, count, device) for key, value in totals.items()}
    result["random_correlation_gain"] = result["true_random_correlation"] - result["shuffled_random_correlation"]
    result["random_delta_correlation_gain"] = (
        result["true_random_delta_correlation"] -
        result["shuffled_random_delta_correlation"]
    )
    result["random_residual_correlation_gain"] = (
        result["true_random_residual_correlation"] -
        result["shuffled_random_residual_correlation"]
    )
    result["random_mse_reduction"] = result["shuffled_random_mse"] - result["true_random_mse"]
    if text_model is not None:
        result["smiles_cross_entropy_gain"] = (
            result["shuffled_smiles_token_cross_entropy"] -
            result["true_smiles_token_cross_entropy"]
        )
        result["smiles_token_accuracy_gain"] = (
            result["true_smiles_token_accuracy"] - result["shuffled_smiles_token_accuracy"]
        )
    world = dist.get_world_size() if dist.is_initialized() else 1
    result["molecules"] = count * world
    return result


@torch.no_grad()
def validate(
    rollout, condition_runner, loader, schedule, distance, config, options, device,
    max_batches=None, text_model=None,
):
    rollout.eval()
    condition_runner.eval()
    if text_model is not None:
        text_model.eval()
    metric_keys = [*loss_metric_keys(options), "diversity", "target_diversity"]
    corrected_distance = None
    if options.get("loss_mode") == "vp_transition_geometry":
        # Training omits the target-target term because it is constant with
        # respect to the model.  Report the complete energy distance as well so
        # the irreducible target-cloud floor is not mistaken for model error.
        corrected_distance = FullBandEnergyDistance(
            levels=distance.levels,
            include_target_constant=True,
            unbiased=distance.unbiased,
        )
        metric_keys.extend(("u_statistic_corrected", "u_statistic_target_floor"))
    if text_model is not None:
        metric_keys.extend(("image_total", "smiles_total", "smiles_token_cross_entropy",
                            "smiles_length_cross_entropy", "smiles_token_accuracy",
                            "smiles_exact_match", "smiles_best_of_4_exact_match",
                            "smiles_length_accuracy"))
    totals = dict.fromkeys(metric_keys, 0.0)
    final_totals = dict.fromkeys(("zero_mse", "random_mse", "best_of_4_mse", "zero_correlation",
                                 "random_correlation", "best_of_4_correlation", "random_diversity",
                                 "zero_rms", "random_rms", "clean_rms",
                                 "zero_update_clean_correlation", "random_update_clean_correlation",
                                 "best_of_4_update_clean_correlation",
                                 "zero_update_target_correlation", "random_update_target_correlation",
                                 "best_of_4_update_target_correlation", "zero_update_rms",
                                 "random_update_rms"), 0.0)
    step_keys = (("zero_current_mean_mse", "u_statistic")
                 if options.get("loss_mode") == "zero_current_global_vp_ustat" else
                 (("zero_mean_mse", "random_radius", "random_direction")
                  if options.get("loss_mode") == "navigation" else ("u_statistic",)))
    step_totals = {key: torch.zeros(int(options["validation_steps"]), device=device) for key in step_keys}
    count, preview = 0, None
    # Validation must not advance training's random streams.
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(100003 + int(os.environ.get("RANK", "0")))
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            clean = batch["field"].to(device, non_blocking=True)
            start = sampled_validation_levels(schedule, len(clean), device, options)
            if options.get("loss_mode", "legacy") == "legacy":
                initial, targets, levels, target_mean, posterior_noise_rms = trajectory_targets(
                    schedule, clean, start, int(options["validation_steps"]),
                    int(options["random_samples"]), config, return_moments=True,
                )
            else:
                initial, levels = rollout_initial_and_levels(
                    schedule, clean, start, int(options["validation_steps"]), config,
                )
            noise = sample_rollout_noise(initial, rollout.cloud, int(options["validation_steps"]),
                                         int(options["random_samples"]))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                condition = _condition_from_batch(condition_runner, batch, device)
                fields = rollout(initial, **condition_arguments(condition), steps=int(options["validation_steps"]),
                                 random_samples=int(options["random_samples"]), noise=noise,
                                 **level_arguments(rollout.cloud, start, levels))
                if options.get("loss_mode") == "zero_current_global_vp_ustat":
                    targets, target_mean, posterior_noise_rms = current_mean_global_vp_targets(
                        schedule, clean, start, levels, int(options["random_samples"]))
                elif options.get("loss_mode", "legacy") != "legacy":
                    targets, target_mean, posterior_noise_rms = step_posterior_targets(
                        schedule, clean, initial, fields, start, levels,
                        sample_innovations=options.get("loss_mode") != "navigation",
                    )
                losses = configured_rollout_losses(fields, targets, initial, clean, distance,
                                                    options, target_mean, posterior_noise_rms, noise)
                if corrected_distance is not None:
                    innovations = (
                        fields[:, :, 1:].float()
                        - target_mean[:, :, 1:].detach().float()
                    )
                    origin = torch.zeros_like(initial, dtype=torch.float32)
                    corrected_steps = torch.stack([
                        corrected_distance(
                            innovations[:, step], targets[:, step].detach(), origin,
                        )
                        for step in range(fields.shape[1])
                    ])
                    corrected = corrected_steps.mean()
                    losses["u_statistic_corrected"] = corrected
                    losses["u_statistic_target_floor"] = losses["u_statistic"] - corrected
                if text_model is not None:
                    token_ids = batch["token_ids"].to(device, non_blocking=True)
                    losses, text_output = add_parallel_smiles_supervision(
                        losses, text_model, fields[:, -1, 1:], token_ids, options,
                    )
                    losses.update(parallel_smiles_metrics(
                        text_output, token_ids, int(options["smiles_pad_token_id"]),
                    ))
            for key in totals:
                totals[key] += float(losses[key]) * len(clean)
            final = fields[:, -1].float()
            errors = (final - clean[:, None]).square().mean(dim=(2, 3, 4))
            centered = final.flatten(2) - final.flatten(2).mean(dim=2, keepdim=True)
            truth = clean.flatten(1) - clean.flatten(1).mean(dim=1, keepdim=True)
            correlation = (centered * truth[:, None]).sum(dim=2) / (
                centered.norm(dim=2) * truth.norm(dim=1)[:, None]).clamp_min(1e-12)
            update = final - initial.float()[:, None]
            centered_update = update.flatten(2) - update.flatten(2).mean(dim=2, keepdim=True)
            target_update = clean.float() - initial.float()
            centered_target_update = target_update.flatten(1)
            centered_target_update = (
                centered_target_update - centered_target_update.mean(dim=1, keepdim=True)
            )
            update_clean_correlation = (centered_update * truth[:, None]).sum(dim=2) / (
                centered_update.norm(dim=2) * truth.norm(dim=1)[:, None]
            ).clamp_min(1e-12)
            update_target_correlation = (centered_update * centered_target_update[:, None]).sum(dim=2) / (
                centered_update.norm(dim=2) * centered_target_update.norm(dim=1)[:, None]
            ).clamp_min(1e-12)
            final_metrics = {"zero_mse": errors[:, 0].mean(), "random_mse": errors[:, 1:].mean(),
                "best_of_4_mse": errors[:, 1:].min(dim=1).values.mean(),
                "zero_correlation": correlation[:, 0].mean(), "random_correlation": correlation[:, 1:].mean(),
                "best_of_4_correlation": correlation[:, 1:].max(dim=1).values.mean(),
                "random_diversity": final[:, 1:].var(dim=1, unbiased=False).mean(),
                "zero_rms": final[:, 0].square().mean().sqrt(),
                "random_rms": final[:, 1:].square().mean().sqrt(),
                "clean_rms": clean.square().mean().sqrt(),
                "zero_update_clean_correlation": update_clean_correlation[:, 0].mean(),
                "random_update_clean_correlation": update_clean_correlation[:, 1:].mean(),
                "best_of_4_update_clean_correlation": update_clean_correlation[:, 1:].max(dim=1).values.mean(),
                "zero_update_target_correlation": update_target_correlation[:, 0].mean(),
                "random_update_target_correlation": update_target_correlation[:, 1:].mean(),
                "best_of_4_update_target_correlation": update_target_correlation[:, 1:].max(dim=1).values.mean(),
                "zero_update_rms": update[:, 0].square().mean().sqrt(),
                "random_update_rms": update[:, 1:].square().mean().sqrt()}
            for key, value in final_metrics.items():
                final_totals[key] += float(value) * len(clean)
            for key in step_keys:
                step_totals[key] += losses[f"step_{key}"] * len(clean)
            count += len(clean)
            if preview is None:
                preview = color_preview(clean, initial, fields)
    result = {key: _distributed_mean(value, count, device) for key, value in totals.items()}
    result.update({f"final_{key}": _distributed_mean(value, count, device)
                   for key, value in final_totals.items()})
    for key, values in step_totals.items():
        for step, value in enumerate(values):
            result[f"step_{step + 1}_{key}"] = _distributed_mean(float(value), count, device)
    result["diversity_ratio"] = result["diversity"] / max(result["target_diversity"], 1e-12)
    return result, preview


def validation_suite(rollout, condition_runner, loader, schedule, distance,
                     config, options, device, *, max_batches=None, text_model=None):
    """Primary validation plus fixed-seed retention checks for focused phases."""
    primary_limit = options.get("validation_max_batches") if max_batches is None else max_batches
    result, preview = validate(rollout, condition_runner, loader, schedule, distance,
                               config, options, device, max_batches=primary_limit,
                               text_model=text_model)
    selected = ("total", "image_total", "zero_mean_mse", "u_statistic_corrected",
                "u_statistic_target_floor", "final_zero_correlation",
                "final_random_correlation", "diversity_ratio",
                "smiles_token_accuracy", "smiles_exact_match", "smiles_best_of_4_exact_match")
    for name, overrides in options.get("validation_suites", {}).items():
        suite_options = {**options, **overrides}
        metrics, _ = validate(
            rollout, condition_runner, loader, schedule, distance, config, suite_options,
            device, max_batches=overrides.get("max_batches", primary_limit), text_model=text_model,
        )
        result.update({f"{name}/{key}": metrics[key] for key in selected if key in metrics})
    return result, preview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--from-scratch", action="store_true", help="random initialization; no checkpoint/alignment")
    parser.add_argument("--reset-ms-encoder", action="store_true",
                        help="strictly load token Cloud; initialize MS encoder from --model-config")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/relative_image_rollout.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warm-start", type=Path, help="load aligned student weights, skip alignment")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="two actual updates per phase in an isolated output")
    parser.add_argument("--stress-long-ms", action="store_true", help="prepare-only: largest eligible training MS sequences")
    parser.add_argument("--benchmark-batches", type=int, default=3)
    parser.add_argument("--validation-batches", type=int, default=1,
                        help="scored validation batches in --prepare-only mode")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile-blocks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile-loss", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--uncheckpointed-refine", type=int)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if args.reset_ms_encoder and (args.from_scratch or args.resume or args.warm_start or not args.checkpoint):
        raise ValueError("reset-ms-encoder requires --checkpoint without from-scratch/resume/warm-start")
    if args.resume and (args.from_scratch or args.warm_start):
        raise ValueError("resume cannot be combined with from-scratch or warm-start")
    if args.resume and args.checkpoint and args.resume.resolve() != args.checkpoint.resolve():
        raise ValueError("checkpoint and resume must refer to the same file")
    rank, world, local = (int(os.environ.get(key, default)) for key, default in
                          (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    dataset = SpectrumFieldDataset(args.data)
    seed = int(dataset.manifest.get("seed", 0))
    torch.manual_seed(seed + rank)
    if args.from_scratch:
        if args.checkpoint or args.warm_start:
            raise ValueError("from-scratch cannot load checkpoint or warm-start weights")
        source, source_epoch = None, None
        config = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    else:
        source_path = args.resume or args.checkpoint
        if source_path is None:
            raise ValueError("specify --checkpoint or --from-scratch")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        source_epoch = int(source["epoch"]) + 1
        config = copy.deepcopy(source["config"])
        if args.reset_ms_encoder:
            replacement = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
            config["spectrum_encoder"] = copy.deepcopy(replacement["spectrum_encoder"])
    options = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if "training_seed" in options and not args.resume:
        torch.manual_seed(int(options["training_seed"]) + rank)
    transition_audit = apply_transition_schedule(config, options)
    bind_spectrum_metadata(config["spectrum_encoder"], dataset.manifest)
    loss_metric_keys(options)
    if options.get("loss_mode") == "zero_current_global_vp_ustat":
        for key in ("zero_current_mean_mse_weight", "u_statistic_weight"):
            if key not in options or not math.isfinite(float(options[key])) or float(options[key]) < 0:
                raise ValueError(f"{key} must be a finite, non-negative loss weight")
        if int(options["steps"]) != 1 or int(options["validation_steps"]) != 1:
            raise ValueError("current-mean Zero objective currently supports one-step training/validation only")
        if any(key in options for key in ("random_bound_weight", "random_radius_weight", "random_direction_weight")):
            raise ValueError("this objective has only Zero MSE and U-stat; remove geometry loss weights")
    if options.get("loss_mode", "legacy") in ("navigation", "vp_transition_geometry"):
        if options["loss_mode"] == "navigation" and "u_statistic_weight" in options:
            raise ValueError("Navigation has exactly three terms; remove u_statistic_weight")
        keys = ("zero_mean_mse_weight", "random_radius_weight", "random_direction_weight")
        if options["loss_mode"] == "vp_transition_geometry":
            keys = ("u_statistic_weight", *keys)
        for key in keys:
            if key not in options or not math.isfinite(float(options[key])) or float(options[key]) < 0:
                raise ValueError(f"{key} must be a finite, non-negative loss weight")
    text_enabled = bool(options.get("parallel_smiles", False))
    if text_enabled:
        for key in ("smiles_max_length", "smiles_dim", "smiles_heads", "smiles_layers",
                    "smiles_image_grid_size"):
            if int(options.get(key, 0)) < 1:
                raise ValueError(f"{key} must be positive when parallel_smiles is enabled")
        for key in ("smiles_loss_weight", "smiles_length_loss_weight", "smiles_learning_rate"):
            if not math.isfinite(float(options.get(key, -1))) or float(options[key]) < 0:
                raise ValueError(f"{key} must be finite and non-negative")
        if int(options["smiles_dim"]) % int(options["smiles_heads"]):
            raise ValueError("smiles_dim must be divisible by smiles_heads")
    if args.smoke_test:
        if args.prepare_only or args.resume:
            raise ValueError("smoke-test cannot be combined with prepare-only or resume")
        options.update(alignment_steps=2, epochs=1, log_every=1)
    if args.batch_size is not None:
        options["batch_size_per_gpu"] = args.batch_size
    if args.steps is not None:
        options["steps"] = args.steps
    if args.uncheckpointed_refine is not None:
        options["uncheckpointed_refine_blocks"] = args.uncheckpointed_refine
    for argument, key in ((args.compile, "compile_rollout"),
                          (args.compile_blocks, "compile_blocks"),
                          (args.compile_loss, "compile_loss"),
                          (args.gradient_checkpointing, "gradient_checkpointing")):
        if argument is not None:
            options[key] = argument
    if int(options["steps"]) < 1 or int(options["random_samples"]) < 2:
        raise ValueError("positive steps and at least two random samples required")
    if options.get("loss_mode") == "zero_current_global_vp_ustat" and int(options["steps"]) != 1:
        raise ValueError("current-mean Zero objective currently supports one-step training only")
    config["relative_image_rollout"] = options
    token_mode = bool(options.get("token_conditioning", False))
    build_kwargs = {"gradient_checkpointing": bool(options["gradient_checkpointing"])}
    audit = {"mode": "legacy-vector-plane", "strict_checkpoint_load": True}
    if args.from_scratch:
        if not token_mode or int(options.get("alignment_steps", 0)) != 0:
            raise ValueError("from-scratch requires token_conditioning and alignment_steps=0")
        config["model"].update(refine_depth=int(options.get("refine_depth", 3)), condition_dim=64)
        if "refine_gate_init" in options:
            config["model"]["refine_gate_init"] = float(options["refine_gate_init"])
        cloud, encoder, audit = build_fresh_token_relative_models(
            config, int(dataset.manifest["metadata_dim"]), **build_kwargs)
    elif args.reset_ms_encoder:
        if not token_mode or int(options.get("alignment_steps", 0)) != 0:
            raise ValueError("MS-only reset requires token_conditioning and alignment_steps=0")
        if int(options.get("refine_depth", config["model"]["refine_depth"])) != config["model"]["refine_depth"]:
            raise ValueError("MS-only reset cannot change refine_depth")
        cloud, encoder, audit = build_ms_reset_relative_models(
            source, config, int(dataset.manifest["metadata_dim"]), **build_kwargs)
    elif token_mode and source["config"].get("relative_image_rollout", {}).get("token_conditioning"):
        cloud, encoder, audit = restore_token_relative_models(
            source, int(dataset.manifest["metadata_dim"]), **build_kwargs)
    elif token_mode:
        cloud, encoder, audit = build_token_relative_models(
            source, int(dataset.manifest["metadata_dim"]), **build_kwargs)
    else:
        cloud, encoder = build_relative_models(source, int(dataset.manifest["metadata_dim"]), **build_kwargs)
    if token_mode:
        gaussian_audit = configure_conditional_gaussian_tokens(cloud, options)
        audit = {**audit, **gaussian_audit}
    warm_start_text_state = None
    warm_start_vocabulary = None
    if args.warm_start:
        weights = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        cloud.load_state_dict(weights["cloud"], strict=True)
        encoder.load_state_dict(weights["condition_encoder"], strict=True)
        warm_start_text_state = weights.get("parallel_smiles")
        warm_start_vocabulary = weights.get("smiles_vocabulary")
        del weights
    if not args.from_scratch and "refine_depth" in options:
        depth = int(options["refine_depth"])
        if depth < 1:
            raise ValueError("refine_depth must be positive")
        if depth > len(cloud.refine_blocks):
            audit["refine_expansion"] = expand_relative_refine_depth(cloud, config, depth)
        else:
            if depth < len(cloud.refine_blocks) and getattr(cloud, "level_conditioner", None) is not None:
                raise ValueError("cannot truncate a pretrained level-conditioned refine stack")
            cloud.refine_blocks = torch.nn.ModuleList(list(cloud.refine_blocks[:depth]))
            config["model"]["refine_depth"] = depth
    if "image_cvt_grid_sizes" in options:
        audit["image_cvt_grid_change"] = configure_relative_image_grids(
            cloud, config, options["image_cvt_grid_sizes"],
        )
    if token_mode:
        audit["level_conditioning"] = configure_relative_level_conditioning(cloud, options)
        if options.get("level_embedding", False):
            maximum = int(config["cloud_matching"]["noise_schedule"]["levels"])
            if (maximum != cloud.level_conditioner.max_level or
                    not torch.equal(_transition_jumps(torch.arange(1, maximum + 1), config),
                                    torch.ones(maximum, dtype=torch.long))):
                raise ValueError("level-conditioned training currently requires jump=1 over every level")
    tokenizer, text_model, smiles_token_lengths = None, None, None
    if text_enabled:
        smiles_values = list(dataset.iter_smiles())
        vocabulary = (warm_start_vocabulary or
                      (source.get("smiles_vocabulary") if source is not None else None))
        tokenizer = (SmilesTokenizer(vocabulary) if vocabulary is not None
                     else SmilesTokenizer.from_smiles(smiles_values))
        smiles_token_lengths = torch.tensor(
            [len(tokenizer.encode(value)) - 1 for value in smiles_values], dtype=torch.long
        )
        if len(smiles_token_lengths) != len(dataset):
            raise RuntimeError("SMILES inventory does not match the field dataset")
        options["smiles_pad_token_id"] = tokenizer.pad_id
        text_model = ParallelFieldToSmiles(
            vocab_size=len(tokenizer),
            max_length=int(options["smiles_max_length"]),
            dim=int(options["smiles_dim"]),
            heads=int(options["smiles_heads"]),
            layers=int(options["smiles_layers"]),
            image_grid_size=int(options["smiles_image_grid_size"]),
            ff_multiplier=float(options.get("smiles_ffn_ratio", 4.0)),
            dropout=float(options.get("smiles_dropout", 0.0)),
        )
        text_state = (warm_start_text_state or
                      (source.get("parallel_smiles") if source is not None else None))
        if text_state is not None:
            text_model.load_state_dict(text_state, strict=True)
        del smiles_values
    cloud, encoder = cloud.to(device), encoder.to(device)
    if text_model is not None:
        text_model = text_model.to(device)
    rollout = RelativeImageRolloutCloud(cloud).to(device)
    condition_module = RaggedSpectrumTokenRunner(encoder) if token_mode else RaggedSpectrumConditionRunner(encoder)
    batch_size = int(options["batch_size_per_gpu"])
    accumulation = int(options["gradient_accumulation_steps"])
    sampler_options = {"seed": seed, "rank": rank, "replicas": world,
                       "validation_fraction": float(config["training"]["validation_fraction"]),
                       "split_seed": seed + 17}
    excluded_count, excluded_smiles_count, postpool_counts = 0, 0, None
    eligible = torch.ones(len(dataset), dtype=torch.bool)
    if token_mode:
        postpool_counts = _postpool_peak_token_counts(dataset, 2, device, world > 1, rank)
        ms_eligible = postpool_counts <= int(options["max_postpool_peak_tokens"])
        eligible &= ms_eligible
        excluded_count = int((~ms_eligible).sum())
    if text_enabled:
        smiles_eligible = smiles_token_lengths <= int(options["smiles_max_length"])
        eligible &= smiles_eligible
        excluded_smiles_count = int((~smiles_eligible).sum())
    if token_mode or text_enabled:
        sampler_options["eligible_mask"] = eligible
    train_sampler = ShardShuffleSampler(
        dataset, split="train", batch_size=batch_size,
        bucket_lengths=postpool_counts, bucket_batches=int(options.get("length_bucket_batches", 16)),
        **sampler_options)
    if args.stress_long_ms:
        if not args.prepare_only or not token_mode:
            raise ValueError("stress-long-ms requires token conditioning and prepare-only")
        candidates = [index for shard in train_sampler.local_indices for index in shard]
        longest = set(sorted(candidates, key=lambda index: int(postpool_counts[index]), reverse=True)[:batch_size * world])
        train_sampler.local_indices = [[index for index in shard if index in longest]
                                       for shard in train_sampler.local_indices]
        train_sampler.selected_count = len(longest)
        train_sampler.samples_per_rank = batch_size
    val_batch = int(options["validation_batch_size_per_gpu"])
    val_sampler = ShardShuffleSampler(dataset, split="validation", batch_size=val_batch,
                                     shuffle=False, **sampler_options)
    loader_options = {"num_workers": args.workers, "pin_memory": device.type == "cuda",
                      "collate_fn": partial(collate_spectrum_field_batch,
                                            peak_chunk_size=int(config["training"].get("peak_chunk_size", 256)),
                                            tokenizer=tokenizer,
                                            max_smiles_length=(int(options["smiles_max_length"])
                                                               if text_enabled else None))}
    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size,
                              persistent_workers=args.workers > 0, **loader_options)
    val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=val_batch, **loader_options)
    epoch_updates = math.ceil(len(train_loader) / accumulation)
    updates_per_epoch = int(options.get("training_updates_per_epoch", epoch_updates))
    if not 1 <= updates_per_epoch <= epoch_updates:
        raise ValueError("training_updates_per_epoch must fit within the data loader")
    epoch_microbatches = min(len(train_loader), updates_per_epoch * accumulation)
    epoch_updates = updates_per_epoch
    max_steps = int(options.get("max_steps", int(options["epochs"]) * epoch_updates))
    if max_steps < 1 or max_steps > int(options["epochs"]) * epoch_updates:
        raise ValueError("max_steps must fit within the configured epochs")
    schedule = _make_schedule(dataset.manifest, config, device)
    if not isinstance(schedule, CosineVPSchedule):
        raise TypeError("relative multi-step fine-tuning requires the original cosine VP schedule")
    options["zero_mean_noise"] = schedule.zero_mean_noise
    level_probabilities = training_level_probabilities(schedule, device, options)
    if options.get("training_start_level") is not None:
        fixed_start = sampled_training_levels(schedule, 1, device, options)
        expected = options.get("training_answer_level")
        actual = int((fixed_start - _transition_jumps(fixed_start, config)).clamp_min(0).item())
        if expected is not None and actual != int(expected):
            raise ValueError(f"focused training targets L{actual}, expected L{expected}")
    distance = (None if options.get("loss_mode") == "navigation" else
                FullBandEnergyDistance(levels=int(config["cloud_matching"]["full_band_levels"]),
                                       include_target_constant=False, unbiased=True))
    model_parameters = [p for p in (*cloud.parameters(), *encoder.parameters()) if p.requires_grad]
    text_parameters = ([] if text_model is None else
                       [p for p in text_model.parameters() if p.requires_grad])
    parameters = [*model_parameters, *text_parameters]
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.resume or args.checkpoint
    metadata = {"source_checkpoint": str(checkpoint_path) if checkpoint_path else None, "source_epoch": source_epoch,
                "initialization": ("random" if args.from_scratch else
                                   "pretrained-cloud-new-ms" if args.reset_ms_encoder else "pretrained"),
                "aligned_warm_start": str(args.warm_start) if args.warm_start else None,
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "strict_checkpoint_load": not token_mode, "weight_transfer": audit,
                "image_feedback": True, "level_embedding": bool(options.get("level_embedding", False)),
                "excluded_long_ms": excluded_count,
                "excluded_long_smiles": excluded_smiles_count,
                "train_molecules": train_sampler.selected_count, "validation_molecules": val_sampler.selected_count,
                "source_model": config["model"], "options": options,
                "transition_schedule": transition_audit,
                "refine_residual_gates": [
                    {"block": index, "attention": float(block.attention_gate.detach()),
                     "ffn": float(block.ffn_gate.detach())}
                    for index, block in enumerate(cloud.refine_blocks) if block.attention_gate is not None],
                "smoke_test": args.smoke_test,
                "stress_long_ms": args.stress_long_ms,
                "actual_condition_dim": 64 if token_mode else config["model"]["condition_dim"],
                "condition_attention": "ordinary-token-cross" if token_mode else "plane-CvT",
                "effective_global_batch": batch_size * world * accumulation,
                "training_level_probabilities": (None if level_probabilities is None else
                                                 level_probabilities.cpu().tolist()),
                "cloud_parameters": sum(p.numel() for p in cloud.parameters()),
                "condition_parameters": sum(p.numel() for p in encoder.parameters()),
                "parallel_smiles_parameters": (0 if text_model is None else
                                                sum(p.numel() for p in text_model.parameters())),
                "smiles_vocabulary_size": (0 if tokenizer is None else len(tokenizer))}
    phase_optimizer = None
    parameter_names = optimizer_parameter_names(cloud, encoder, text_model)
    source_parameter_names = None
    if source is not None:
        source_parameter_names = source.get("optimizer_parameter_names")
        if source_parameter_names is None:
            source_parameter_names = [
                [name for name in group if name.split(".", 1)[1] in source[name.split(".", 1)[0]]]
                for group in parameter_names
            ]
    if options.get("warm_start_optimizer", False) and not args.resume:
        if source is None or args.warm_start or args.reset_ms_encoder:
            raise ValueError("warm_start_optimizer requires an unchanged --checkpoint model")
        phase_optimizer = source["optimizer"]
    parent_global_step = (int(source.get("parent_global_step", source.get("global_step", 0)))
                          if args.resume else int(source.get("global_step", 0))) if source is not None else 0
    metadata.update(resume_checkpoint=str(args.resume) if args.resume else None,
                    optimizer_state=("not-created-prepare-only" if args.prepare_only else
                                     "restored" if args.resume else
                                     "restored-new-phase" if phase_optimizer is not None else "fresh"),
                    updates_per_epoch=epoch_updates, max_steps=max_steps,
                    parent_global_step=parent_global_step,
                    source_global_step=int(source["global_step"]) if args.resume else 0)
    if rank == 0:
        (args.output / "preparation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata), flush=True)
    if token_mode and not args.from_scratch and not args.resume and not args.warm_start and int(options.get("alignment_steps", 0)) > 0:
        run_alignment(cloud, encoder, source, train_loader, val_loader, schedule, options,
                      args.output, device, rank, world, local,
                      prepare_only=args.prepare_only, benchmark_batches=args.benchmark_batches)
    resume_state = source if args.resume else None
    del source
    if bool(options.get("compile_blocks", False)) or bool(options.get("compile_pointwise", False)):
        torch._dynamo.config.optimize_ddp = False
    compile_options = configure_relative_execution(cloud, options) if token_mode else {}
    loss_execution = (torch.compile(configured_rollout_losses, fullgraph=True, dynamic=True,
                                    options=compile_options)
                      if bool(options.get("compile_loss", False)) else configured_rollout_losses)
    execution = rollout
    if bool(options["compile_rollout"]):
        torch._dynamo.config.optimize_ddp = False
        execution = torch.compile(rollout, fullgraph=True, dynamic=False)
    runner, condition_runner, text_runner = execution, condition_module, text_model
    if world > 1:
        runner = DDP(execution, device_ids=[local], static_graph=True)
        condition_runner = DDP(condition_module, device_ids=[local])
        if text_model is not None:
            text_runner = DDP(text_model, device_ids=[local], static_graph=True)
    optimizer, scheduler = None, None
    start_epoch, global_step, resume_microbatch = 0, 0, 0
    if not args.prepare_only:
        peak_lr = float(options["learning_rate"])
        parameter_groups = [{"params": model_parameters, "lr": peak_lr,
                             "schedule_peak_lr": peak_lr}]
        if text_parameters:
            text_lr = float(options["smiles_learning_rate"])
            parameter_groups.append({"params": text_parameters, "lr": text_lr,
                                     "schedule_peak_lr": text_lr})
        optimizer = torch.optim.AdamW(parameter_groups, lr=peak_lr,
                                      weight_decay=float(options["weight_decay"]),
                                      fused=device.type == "cuda")
        if phase_optimizer is not None:
            optimizer_audit = restore_optimizer_adding_level_parameters(
                optimizer, phase_optimizer, parameter_names, source_parameter_names)
            if rank == 0:
                (args.output / "optimizer_transfer.json").write_text(json.dumps(optimizer_audit, indent=2))
            phase_lrs = [peak_lr] + ([float(options["smiles_learning_rate"])] if text_parameters else [])
            for group, phase_lr in zip(optimizer.param_groups, phase_lrs, strict=True):
                group["lr"] = group["initial_lr"] = group["schedule_peak_lr"] = phase_lr
        del phase_optimizer
        if args.resume:
            resumed = resume_state
            if resumed.get("training_kind") != "relative-image-feedback-rollout":
                raise ValueError("use --checkpoint, not --resume, to warm-start an old one-step run")
            previous_mode = resumed["config"].get("relative_image_rollout", {}).get("loss_mode", "legacy")
            if previous_mode != options.get("loss_mode", "legacy"):
                raise ValueError("objective changed; use --warm-start and a new output directory, not --resume")
            optimizer_audit = restore_optimizer_adding_level_parameters(
                optimizer, resumed["optimizer"], parameter_names, source_parameter_names)
            if rank == 0:
                (args.output / "optimizer_transfer.json").write_text(json.dumps(optimizer_audit, indent=2))
            complete = resumed.get("epoch_complete", True)
            start_epoch = int(resumed["epoch"]) + int(complete)
            resume_microbatch = 0 if complete else int(resumed["next_microbatch"])
            global_step = int(resumed["global_step"])
            if global_step >= max_steps:
                raise ValueError("checkpoint has already reached max_steps")
            if "rng_states" in resumed:
                if len(resumed["rng_states"]) != world:
                    raise ValueError("exact RNG continuation requires the same world size")
                rng = resumed["rng_states"][rank]
                torch.set_rng_state(rng["torch"])
                if device.type == "cuda":
                    torch.cuda.set_rng_state(rng["cuda"], device)
            else:
                # Older checkpoints omitted RNG; avoid replaying the original seed stream.
                torch.manual_seed(seed + rank + global_step)
        scheduler = make_lr_scheduler(optimizer, options, max_steps, global_step=global_step)
        if rank == 0 and args.resume:
            state_audit = {"source_global_step": global_step, "start_epoch": start_epoch + 1,
                           "optimizer_state_tensors": len(optimizer.state),
                           "optimizer_state_restored": True,
                           "learning_rate": scheduler.get_last_lr()[0], "max_steps": max_steps,
                           "rng_restored": "rng_states" in resume_state}
            (args.output / "resume_state.json").write_text(json.dumps(state_audit, indent=2), encoding="utf-8")
            print(json.dumps(state_audit), flush=True)
        if args.resume:
            del resumed
    del resume_state
    writer = (SummaryWriter(args.output / "tensorboard",
                            purge_step=global_step + 1 if args.resume else None)
              if rank == 0 and not args.prepare_only else None)
    control_every = int(options.get("condition_control_every_epochs", 0))
    if (options.get("validate_before_training", False) and not args.prepare_only
            and (not args.resume or not (args.output / "validation_baseline.json").exists())):
        baseline, preview = validation_suite(
            rollout, condition_module, val_loader, schedule, distance, config, options,
            device, text_model=text_model,
        )
        if writer:
            (args.output / "validation_baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
            for key, value in baseline.items():
                writer.add_scalar(f"validation/{key}", value, global_step)
            writer.add_images("samples/GT_input_finalZero_Random1to4", preview, global_step)
            writer.flush()
            print(f"validation_baseline {json.dumps(baseline)}", flush=True)
    if control_every and (args.resume or options.get("validate_before_training", False)) and not args.prepare_only:
        baseline = condition_control_metrics(
            rollout, condition_module, val_loader, schedule, options, device, text_model,
        )
        if writer:
            baseline_name = (f"condition_control_resume_step-{global_step:07d}.json"
                             if args.resume and (args.output / "condition_control_baseline.json").exists()
                             else "condition_control_baseline.json")
            (args.output / baseline_name).write_text(
                json.dumps(baseline, indent=2), encoding="utf-8")
            for key, value in baseline.items():
                writer.add_scalar(f"condition_control/{key}", value, global_step)
            writer.flush()
            print(f"condition_control step={global_step} {json.dumps(baseline)}", flush=True)
    benchmark = []
    for epoch in range(start_epoch, int(options["epochs"])):
        train_sampler.set_epoch(epoch)
        runner.train()
        condition_runner.train()
        if text_runner is not None:
            text_runner.train()
        train_keys = list(loss_metric_keys(options))
        if text_runner is not None:
            train_keys.extend(("image_total", "smiles_total", "smiles_token_cross_entropy",
                               "smiles_length_cross_entropy"))
        totals = dict.fromkeys(train_keys, 0.0)
        count = 0
        window_start, window_count = time.perf_counter(), 0
        for micro, batch in enumerate(train_loader):
            if micro >= epoch_microbatches:
                break
            if epoch == start_epoch and micro < resume_microbatch:
                continue
            if args.smoke_test and micro >= 2 * accumulation:
                break
            if args.prepare_only and micro >= args.benchmark_batches:
                break
            if args.prepare_only:
                for parameter in parameters:
                    parameter.grad = None
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            clean = batch["field"].to(device, non_blocking=True)
            start = sampled_training_levels(schedule, len(clean), device, options,
                                            probabilities=level_probabilities)
            if options.get("loss_mode", "legacy") == "legacy":
                initial, targets, levels, target_mean, posterior_noise_rms = trajectory_targets(schedule, clean, start,
                    int(options["steps"]), int(options["random_samples"]), config, return_moments=True)
            else:
                initial, levels = rollout_initial_and_levels(schedule, clean, start, int(options["steps"]), config)
            noise = sample_rollout_noise(initial, cloud, int(options["steps"]), int(options["random_samples"]))
            should_step = (micro + 1) % accumulation == 0 or micro + 1 == epoch_microbatches
            group_size = min(accumulation, epoch_microbatches - (micro // accumulation) * accumulation)
            with ExitStack() as stack:
                if world > 1 and not should_step and not args.prepare_only:
                    stack.enter_context(runner.no_sync())
                    stack.enter_context(condition_runner.no_sync())
                    if text_runner is not None:
                        stack.enter_context(text_runner.no_sync())
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    condition = _condition_from_batch(condition_runner, batch, device)
                    fields = runner(initial, **condition_arguments(condition), steps=int(options["steps"]),
                                    random_samples=int(options["random_samples"]), noise=noise,
                                    **level_arguments(cloud, start, levels))
                    if options.get("loss_mode") == "zero_current_global_vp_ustat":
                        targets, target_mean, posterior_noise_rms = current_mean_global_vp_targets(
                            schedule, clean, start, levels, int(options["random_samples"]))
                    elif options.get("loss_mode", "legacy") != "legacy":
                        targets, target_mean, posterior_noise_rms = step_posterior_targets(
                            schedule, clean, initial, fields, start, levels,
                            sample_innovations=options.get("loss_mode") != "navigation",
                        )
                    losses = loss_execution(fields, targets, initial, clean, distance, options,
                                            target_mean, posterior_noise_rms, noise)
                    if text_runner is not None:
                        token_ids = batch["token_ids"].to(device, non_blocking=True)
                        losses, _ = add_parallel_smiles_supervision(
                            losses, text_runner, fields[:, -1, 1:], token_ids, options,
                        )
                (losses["total"] / (1 if args.prepare_only else group_size)).backward()
            for key in totals:
                totals[key] += float(losses[key].detach()) * len(clean)
            count += len(clean)
            window_count += len(clean) * world
            if args.prepare_only:
                grad = torch.nn.utils.clip_grad_norm_(parameters, float("inf"))
                if not torch.isfinite(grad) or not torch.isfinite(losses["total"]):
                    raise RuntimeError("non-finite loss/gradient during preparation")
                if device.type == "cuda":
                    torch.cuda.synchronize()
                measurement = {"batch": micro, "seconds": time.perf_counter() - begin,
                    "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0,
                    "loss": float(losses["total"].detach()), "grad_norm": float(grad),
                    "zero_mean_max_error": float(fields.detach().float().mean(dim=(-2, -1)).abs().max()),
                    "optimizer_updates": 0}
                benchmark.append(measurement)
                if rank == 0:
                    print(json.dumps(measurement), flush=True)
                del fields, losses, condition, initial, targets, target_mean, posterior_noise_rms, noise
                continue
            if should_step:
                grad = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                if not torch.isfinite(grad) or not torch.isfinite(losses["total"]):
                    raise RuntimeError("non-finite rollout loss/gradient; optimizer update cancelled")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if writer and (global_step % int(options["log_every"]) == 0 or global_step == max_steps):
                    for key in totals:
                        writer.add_scalar(f"train/batch_{key}", float(losses[key].detach()), global_step)
                    writer.add_scalar("train/grad_norm", float(grad), global_step)
                    writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)
                    gated_blocks = [block for block in cloud.refine_blocks if block.attention_gate is not None]
                    if gated_blocks:
                        writer.add_scalar("train/refine_attention_gate_mean",
                                          torch.stack([block.attention_gate.detach() for block in gated_blocks]).mean(),
                                          global_step)
                        writer.add_scalar("train/refine_ffn_gate_mean",
                                          torch.stack([block.ffn_gate.detach() for block in gated_blocks]).mean(),
                                          global_step)
                    diversity_key = ("diversity" if options.get("loss_mode", "legacy")
                                     in ("legacy", "zero_current_global_vp_ustat") else "innovation_diversity")
                    writer.add_scalar(f"train/{diversity_key}", float(losses["diversity"]), global_step)
                    speed = window_count / (time.perf_counter() - window_start)
                    writer.add_scalar("train/molecules_per_second", speed, global_step)
                    center_key = ("zero_current_mean_mse" if options.get("loss_mode") == "zero_current_global_vp_ustat"
                                  else ("zero_mean_mse" if options.get("loss_mode") != "legacy"
                                        else "zero_token_clean_mse"))
                    geometry_log = (f" radius={float(losses['random_radius'].detach()):.5f} "
                                    f"direction={float(losses['random_direction'].detach()):.5f}"
                                    if options.get("loss_mode") in ("navigation", "vp_transition_geometry") else "")
                    ustat_log = (f"ustat={float(losses['u_statistic'].detach()):.5f} "
                                 if "u_statistic" in losses else "")
                    smiles_log = (f" smiles_ce={float(losses['smiles_token_cross_entropy'].detach()):.5f}"
                                  if "smiles_token_cross_entropy" in losses else "")
                    print(f"epoch={epoch + 1} step={global_step} loss={float(losses['total'].detach()):.5f} "
                          f"{ustat_log}"
                          f"zero_mse={float(losses[center_key].detach()):.5f}"
                          f"{geometry_log}{smiles_log} speed={speed:.2f}mol/s", flush=True)
                    window_start, window_count = time.perf_counter(), 0
                if global_step >= max_steps or micro + 1 >= epoch_microbatches:
                    break
        if args.prepare_only:
            for parameter in parameters:
                parameter.grad = None
            validation, preview = validate(
                rollout, condition_module, val_loader, schedule, distance, config, options,
                device, max_batches=args.validation_batches, text_model=text_model,
            )
            if rank == 0:
                (args.output / "benchmark.json").write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
                (args.output / "preparation_validation.json").write_text(
                    json.dumps(validation, indent=2), encoding="utf-8")
                if preview is not None:
                    import matplotlib.pyplot as plt

                    image = torch.cat(list(preview), dim=1).permute(1, 2, 0).numpy()
                    plt.imsave(args.output / "preparation_zero_updates_GT_input_Zero_Random1to4.png", image)
                print("Preparation complete: zero optimizer updates; no training checkpoint written.", flush=True)
            if control_every:
                control = condition_control_metrics(
                    rollout, condition_module, val_loader, schedule, options, device, text_model,
                )
                if rank == 0:
                    (args.output / "condition_control.json").write_text(json.dumps(control, indent=2), encoding="utf-8")
            break
        train_means = {key: _distributed_mean(value, count, device) for key, value in totals.items()}
        validation, preview = validation_suite(rollout, condition_module, val_loader, schedule,
                                        distance, config, options, device,
                                        max_batches=1 if args.smoke_test else None,
                                        text_model=text_model)
        control = (condition_control_metrics(
                       rollout, condition_module, val_loader, schedule, options, device, text_model)
                   if control_every and ((epoch + 1) % control_every == 0 or global_step >= max_steps) else None)
        rng = {"torch": torch.get_rng_state(),
               "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
        rng_states = [None] * world
        if world > 1:
            dist.all_gather_object(rng_states, rng)
        else:
            rng_states[0] = rng
        if writer:
            for key, value in train_means.items():
                writer.add_scalar(f"train/epoch_{key}", value, epoch + 1)
            for key, value in validation.items():
                writer.add_scalar(f"validation/{key}", value, epoch + 1)
            sample_tag = ("samples/GT_input_ZeroCurrent_RandomNext1to4"
                          if options.get("loss_mode") == "zero_current_global_vp_ustat"
                          else "samples/GT_input_finalZero_Random1to4")
            writer.add_images(sample_tag, preview, epoch + 1)
            if control is not None:
                for key, value in control.items():
                    writer.add_scalar(f"condition_control/{key}", value, global_step)
                directory = args.output / "condition_controls"
                directory.mkdir(exist_ok=True)
                (directory / f"epoch-{epoch + 1:04d}.json").write_text(
                    json.dumps(control, indent=2), encoding="utf-8")
                print(f"condition_control step={global_step} {json.dumps(control)}", flush=True)
            path = args.output / ("smoke.pt" if args.smoke_test else f"epoch-{epoch + 1:04d}.pt")
            temporary = path.with_suffix(".tmp")
            torch.save({"cloud": cloud.state_dict(), "condition_encoder": encoder.state_dict(),
                        "parallel_smiles": (None if text_model is None else text_model.state_dict()),
                        "smiles_vocabulary": (None if tokenizer is None else tokenizer.id_to_token),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "optimizer_parameter_names": parameter_names,
                        "epoch": epoch, "global_step": global_step, "config": config,
                        "epoch_complete": micro + 1 >= epoch_microbatches, "next_microbatch": micro + 1,
                        "parent_global_step": parent_global_step,
                        "rng_states": rng_states,
                "source_checkpoint": str(checkpoint_path) if checkpoint_path else None,
                        "training_kind": "relative-image-feedback-rollout-smoke" if args.smoke_test else "relative-image-feedback-rollout"}, temporary)
            os.replace(temporary, path)
            _update_latest(path, args.output / "latest.pt")
            (args.output / f"validation-step-{global_step:07d}.json").write_text(
                json.dumps(validation, indent=2), encoding="utf-8")
            writer.flush()
            long_preview = long_rollout_preview(rollout, condition_module, val_loader, schedule, options, device)
            if long_preview is not None:
                images, selected = long_preview
                current_mean_mode = options.get("loss_mode") == "zero_current_global_vp_ustat"
                prefix = "long_random_only" if current_mean_mode else "long_rollout"
                writer.add_images(f"samples/{prefix}_GT_input_steps-{'-'.join(map(str, selected))}", images, epoch + 1)
                save_long_rollout_preview(long_preview, args.output, epoch,
                                         int(options["random_samples"]) + int(not current_mean_mode))
                writer.flush()
            print(f"epoch={epoch + 1} train={train_means['total']:.5f} "
                  f"validation_{options['validation_steps']}step={validation['total']:.5f} saved={path}", flush=True)
        if world > 1:
            dist.barrier()
        if global_step >= max_steps:
            break
    if writer:
        (args.output / "training_finished.json").write_text(json.dumps(
            {"global_step": global_step, "max_steps": max_steps,
             "target_reached": global_step >= max_steps}, indent=2), encoding="utf-8")
        writer.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
