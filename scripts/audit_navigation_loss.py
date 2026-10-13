"""Synthetic failure-mode audit of Navigation loss; no training or weight changes.

These are constructed counterexamples, not measurements of a trained checkpoint.
Run with ``python scripts/audit_navigation_loss.py --device cuda``.
"""

from __future__ import annotations

import argparse
import json

import torch
from train_relative_image_rollout import configured_rollout_losses

from molai.models.bridge import CosineVPSchedule
from molai.models.losses import FullBandEnergyDistance, vp_cloud_geometry_losses

OPTIONS = {"loss_mode": "vp_transition_geometry", "u_statistic_weight": 1.,
           "zero_mean_mse_weight": 1., "random_radius_weight": 1., "random_direction_weight": 1.}


def centered_gaussian(shape, device, generator):
    noise = torch.randn(shape, device=device, generator=generator)
    return noise - noise.mean(dim=(-2, -1), keepdim=True)


def collapsed_gradient_audit(device):
    """A non-optimal exactly collapsed output can be a stationary point."""
    fields = torch.zeros(1, 1, 5, 1, 4, 4, device=device, requires_grad=True)
    mean = torch.zeros_like(fields)
    sigma = torch.ones(1, 1, device=device)
    tokens = torch.tensor([[[[[1., 0.]], [[0., 1.]], [[-1., 0.]], [[0., -1.]]]]], device=device)
    pattern = torch.arange(16, device=device).reshape(1, 1, 1, 4, 4).float() - 7.5
    targets = torch.stack((pattern, -pattern, pattern.flip(-1), -pattern.flip(-1)), dim=2)
    noise = torch.cat((torch.zeros_like(tokens[:, :, :1]), tokens), dim=2)
    origin = fields.new_zeros(1, 1, 4, 4)
    distance = FullBandEnergyDistance(levels=3, include_target_constant=False)
    losses = configured_rollout_losses(fields, targets, origin, origin, distance,
                                        OPTIONS, mean, sigma, noise)
    geometry = losses["random_radius"] + losses["random_direction"]
    geometry_grad = torch.autograd.grad(geometry, fields, retain_graph=True)[0]
    total_grad = torch.autograd.grad(losses["total"], fields)[0]
    return {"radius_loss": float(losses["random_radius"].detach()),
            "direction_loss": float(losses["random_direction"].detach()),
            "geometry_gradient_max": float(geometry_grad.abs().max()),
            "total_gradient_max_symmetric_targets": float(total_grad.abs().max())}


@torch.no_grad()
def inherited_diversity_audit(device, batch=64, size=64):
    """Pooled innovations can appear Gaussian while all next states are equal.

    Parents are independent zero-mean Gaussians scaled so that their analytic VP
    means have exactly the posterior innovation covariance. The deliberately bad
    model emits the same zero next state and ignores fresh tokens. Innovations
    are therefore minus those means: marginally correct Gaussian noise entirely
    inherited from different parents, not conditional diversity at a fixed parent.
    """
    generator = torch.Generator(device=device).manual_seed(98217)
    schedule = CosineVPSchedule(levels=64, noise_scale=3.5, zero_mean_noise=True, device=device)
    clean = torch.zeros(batch, 1, size, size, device=device)
    before = torch.full((batch,), 32, device=device)
    after = torch.full((batch,), 28, device=device)
    _, variance = schedule.target_mean_and_variance(clean, clean, before, after)
    coefficient, _ = schedule.target_mean_and_variance(clean, torch.ones_like(clean), before, after)
    raw_std = schedule.noise_scale * variance.sqrt()[:, None, None, None, None]
    parents = centered_gaussian((batch, 4, 1, size, size), device, generator)
    parents *= raw_std / coefficient[:, None]
    expanded_clean = clean[:, None].expand_as(parents).flatten(0, 1)
    parent_mean, _ = schedule.target_mean_and_variance(
        expanded_clean, parents.flatten(0, 1), before.repeat_interleave(4), after.repeat_interleave(4),
    )
    means = torch.cat((clean[:, None], parent_mean.reshape_as(parents)), dim=1)[:, None]
    fields = torch.zeros_like(means)
    sigma = schedule.noise_scale * (variance * (1. - 1. / size**2)).sqrt()[:, None]
    tokens = torch.randn(batch, 1, 4, size, size, device=device, generator=generator)
    noise = torch.cat((torch.zeros_like(tokens[:, :, :1]), tokens), dim=2)
    targets = centered_gaussian((batch, 4, 1, size, size), device, generator) * raw_std
    targets = targets[:, None]
    # Constant omitted in training is immaterial for gradient, but including it
    # lets the audit report an energy discrepancy whose population optimum is 0.
    distance = FullBandEnergyDistance(levels=3, include_target_constant=True)
    losses = configured_rollout_losses(fields, targets, clean, clean, distance,
                                        OPTIONS, means, sigma, noise)
    fresh_innovations = tokens[:, 0].reshape_as(parents)
    fresh_innovations = fresh_innovations - fresh_innovations.mean(dim=(-2, -1), keepdim=True)
    fresh_innovations *= raw_std
    fresh_fields = means.clone()
    fresh_fields[:, :, 1:] += fresh_innovations[:, None]
    control = configured_rollout_losses(fresh_fields, targets, clean, clean, distance,
                                        OPTIONS, means, sigma, noise)
    return {"zero_mean_mse": float(losses["zero_mean_mse"]),
            "radius_loss": float(losses["random_radius"]),
            "direction_loss": float(losses["random_direction"]),
            "energy_discrepancy_with_target_constant": float(losses["u_statistic"]),
            "objective_with_target_constant": float(losses["total"]),
            "reported_innovation_diversity": float(losses["diversity"]),
            "target_innovation_diversity": float(losses["target_diversity"]),
            "actual_next_state_diversity": float(fields[:, :, 1:].var(dim=2, unbiased=False).mean()),
            "fresh_token_output_sensitivity": 0.0,
            "fresh_token_control_objective_with_target_constant": float(control["total"]),
            "fresh_token_control_radius_loss": float(control["random_radius"]),
            "fresh_token_control_direction_loss": float(control["random_direction"]),
            "pixels": size**2}


def stability_audit(device):
    """Loss values do not determine local input-to-output perturbation gain."""
    tokens = torch.tensor([[[[[1., 0.]], [[0., 1.]], [[-1., 0.]], [[0., -1.]]]]], device=device)
    basis = torch.tensor([[1., -1., 1., -1.], [1., 1., -1., -1.]], device=device)
    offsets = (tokens.flatten(3) @ basis / 2**0.5).reshape(1, 1, 4, 1, 1, 4)
    scalar_input = torch.zeros((), device=device, requires_grad=True)
    pattern = basis[0].reshape(1, 1, 1, 1, 1, 4)
    means = (0.5 * scalar_input * pattern).expand(1, 1, 5, 1, 1, 4)
    baseline = torch.cat((means[:, :, :1], means[:, :, 1:] + offsets), dim=2)
    unstable = baseline + 0.001 * torch.sin(1_000_000 * scalar_input) * pattern
    sigma = torch.ones(1, 1, device=device)
    origin = torch.zeros(1, 1, 1, 4, device=device)
    noise = torch.cat((torch.zeros_like(tokens[:, :, :1]), tokens), dim=2)
    distance = FullBandEnergyDistance(levels=3, include_target_constant=False)
    good = configured_rollout_losses(baseline, offsets, origin, origin, distance,
                                     OPTIONS, means, sigma, noise)
    bad = configured_rollout_losses(unstable, offsets, origin, origin, distance,
                                    OPTIONS, means, sigma, noise)
    good_gain = torch.autograd.grad(baseline[0, 0, 0, 0, 0, 0], scalar_input, retain_graph=True)[0]
    bad_gain = torch.autograd.grad(unstable[0, 0, 0, 0, 0, 0], scalar_input)[0]
    return {"loss_difference_at_audited_input": float((good["total"] - bad["total"]).detach()),
            "stable_local_gain": float(good_gain), "unstable_local_gain": float(bad_gain),
            "maximum_output_error_amplitude": 0.001}


def terminal_diversity_audit(device):
    """L0 radial supervision penalizes independent Random variance about clean."""
    tokens = torch.tensor([[[[[1., 0.]], [[0., 1.]], [[-1., 0.]], [[0., -1.]]]]], device=device)
    pattern = torch.tensor([1., -1., 1., -1.], device=device).reshape(1, 1, 1, 1, 4)
    mean = torch.zeros(1, 1, 5, 1, 1, 4, device=device)
    sigma = torch.zeros(1, 1, device=device)
    results = []
    for amplitude in (0., 1., 2.):
        fields = mean.clone()
        fields[:, :, 1:] = amplitude * pattern[:, :, None] * torch.tensor(
            [1., -1., 1., -1.], device=device,
        )[None, None, :, None, None, None]
        loss = vp_cloud_geometry_losses(fields, mean, sigma, tokens)
        results.append({"random_amplitude": amplitude, "radius_loss": float(loss["random_radius"]),
                        "direction_loss": float(loss["random_direction"])})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    report = {"kind": "synthetic_navigation_loss_audit_not_a_checkpoint_evaluation",
              "device": str(device), "collapsed_stationary_point": collapsed_gradient_audit(device),
              "inherited_diversity_counterexample": inherited_diversity_audit(device),
              "stability_counterexample": stability_audit(device),
              "terminal_radius": terminal_diversity_audit(device)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
