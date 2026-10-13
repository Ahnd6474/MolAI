import importlib
from pathlib import Path

import pytest
import torch

from molai.models.bridge import CosineVPSchedule
from molai.models.losses import FullBandEnergyDistance


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("train_relative_image_rollout")


def make_targets(trainer, size=8, samples=4):
    schedule = CosineVPSchedule(levels=64, noise_scale=3.5, zero_mean_noise=True)
    clean = torch.randn(3, 1, size, size)
    clean -= clean.mean(dim=(-2, -1), keepdim=True)
    start, levels = torch.tensor([64, 8, 0]), torch.tensor([[56], [7], [0]])
    targets, means, radius = trainer.current_mean_global_vp_targets(schedule, clean, start, levels, samples)
    return schedule, clean, start, levels, targets, means, radius


def test_zero_is_current_signal_and_random_is_next_global_not_posterior(trainer):
    schedule, clean, start, levels, targets, means, radius = make_targets(trainer)
    torch.testing.assert_close(means[:, 0, 0], schedule.alpha_bar[start].sqrt()[:, None, None, None] * clean)
    next_mean = schedule.alpha_bar[levels[:, 0]].sqrt()[:, None, None, None] * clean
    torch.testing.assert_close(means[:, 0, 1], next_mean)
    assert means[0, 0, 0].count_nonzero() == 0  # terminal pure-noise input
    assert means[0, 0, 1].abs().sum() > 0       # but Random must generate signal
    torch.testing.assert_close(targets[-1, 0], clean[-1:].expand(4, -1, -1, -1))
    assert radius[-1].item() == 0
    torch.testing.assert_close((targets - means[:, :, 1:]).mean(dim=(-2, -1)),
                               torch.zeros(3, 1, 4, 1), atol=2e-7, rtol=0)
    expected = schedule.noise_scale * (1 - schedule.alpha_bar[levels]).sqrt() * (63 / 64) ** .5
    torch.testing.assert_close(radius, expected)
    assert not targets.requires_grad and not means.requires_grad


def test_only_zero_mse_and_ustat_are_computed_and_compilable(trainer):
    _, clean, _, _, targets, means, radius = make_targets(trainer, size=4)
    fields = torch.randn_like(means, requires_grad=True)
    options = {"loss_mode": "zero_current_global_vp_ustat", "zero_current_mean_mse_weight": 1.,
               "u_statistic_weight": 1.}
    distance = FullBandEnergyDistance(levels=1, include_target_constant=False, unbiased=True)
    compiled = torch.compile(trainer.configured_rollout_losses, backend="aot_eager", fullgraph=True)
    result = compiled(fields, targets, clean, clean, distance, options, means, radius, None)
    torch.testing.assert_close(result["total"], result["zero_current_mean_mse"] + result["u_statistic"])
    assert not any("bound" in k or "direction" in k or "random_radius" in k for k in result)
    result["total"].backward()
    assert torch.isfinite(fields.grad).all()
    assert fields.grad[:, :, 0].abs().sum() > 0
    assert fields.grad[:, :, 1:].abs().sum() > 0
    keys = trainer.loss_metric_keys(options)
    assert "zero_current_mean_mse" in keys and "u_statistic" in keys


def test_zero_mse_does_not_supervise_random_or_use_clean_target(trainer):
    _, clean, _, _, targets, means, radius = make_targets(trainer)
    fields = means.clone().requires_grad_(True)
    fields.data[:, :, 1:] += 12
    options = {"zero_current_mean_mse_weight": 1., "u_statistic_weight": 0.}
    distance = FullBandEnergyDistance(levels=1, include_target_constant=False)
    result = trainer.current_mean_global_vp_losses(fields, targets, means, radius, distance, options)
    assert result["zero_current_mean_mse"].item() == 0
    assert (fields[0, 0, 0] - clean[0]).square().mean() > 0
    result["total"].backward()
    torch.testing.assert_close(fields.grad[:, :, 1:], torch.zeros_like(fields.grad[:, :, 1:]))


def test_global_targets_have_expected_noise_energy_and_independent_samples(trainer):
    torch.manual_seed(39)
    _, _, _, _, targets, means, radius = make_targets(trainer, size=64, samples=256)
    innovation = targets[:, :, :, 0] - means[:, :, 1:, 0]
    energy = innovation[:2].square().mean(dim=(2, 3, 4))
    torch.testing.assert_close(energy, radius[:2].square(), atol=.01, rtol=.01)
    cross = (innovation[0, 0, :-1] * innovation[0, 0, 1:]).mean()
    assert cross.abs() < .05


def test_target_sampling_is_reproducible(trainer):
    torch.manual_seed(1729)
    first = make_targets(trainer)
    torch.manual_seed(1729)
    second = make_targets(trainer)
    for left, right in zip(first[1:], second[1:], strict=True):
        torch.testing.assert_close(left, right)


def test_model_backward_validation_and_random_only_long_preview(trainer, monkeypatch):
    from test_navigation_training_setup import small_config

    from molai.models.relative_rollout import (
        RelativeImageRolloutCloud,
        build_fresh_token_relative_models,
    )

    cloud, _, _ = build_fresh_token_relative_models(small_config(), 3, gradient_checkpointing=False)
    rollout = RelativeImageRolloutCloud(cloud)
    schedule, clean, start, _levels, targets, means, radius = make_targets(trainer, size=4)
    initial = schedule.sample_training_batch(clean, 1, current_levels=start, answer_jump=0).current
    condition = torch.randn(3, 2, 64, requires_grad=True)
    mask = torch.ones(3, 2, dtype=torch.bool)
    fields = rollout(initial, condition, steps=1, condition_mask=mask)
    options = {"loss_mode": "zero_current_global_vp_ustat", "zero_current_mean_mse_weight": 1.,
               "u_statistic_weight": 1., "random_samples": 4, "validation_steps": 1,
               "validation_start_levels": "uniform", "long_rollout_samples": 1, "long_rollout_steps": 4}
    distance = FullBandEnergyDistance(levels=1, include_target_constant=False)
    loss = trainer.configured_rollout_losses(fields, targets, initial, clean, distance,
                                            options, means, radius, None)
    loss["total"].backward()
    assert condition.grad.abs().sum() > 0
    assert torch.isfinite(cloud.output_head.weight.grad).all()
    batch = {"field": clean, "condition": condition.detach()}
    monkeypatch.setattr(trainer, "_condition_from_batch", lambda *_args: (batch["condition"], mask))
    state = torch.random.get_rng_state().clone()
    validation, preview = trainer.validate(rollout, torch.nn.Identity(), [batch], schedule, distance,
                                          {"cloud_matching": {"answer_jump": 8}}, options, torch.device("cpu"))
    torch.testing.assert_close(state, torch.random.get_rng_state())
    assert "step_1_zero_current_mean_mse" in validation and "step_1_u_statistic" in validation
    assert not any("random_radius" in k or "direction" in k or "bound" in k for k in validation)
    assert preview.shape == (3, 3, 4, 28)
    images, selected = trainer.long_rollout_preview(rollout, torch.nn.Identity(), [batch], schedule,
                                                  options, torch.device("cpu"))
    assert len(images) == 4  # Random only: Zero is a denoiser, not a transition.
    assert selected == [1, 2, 4]
