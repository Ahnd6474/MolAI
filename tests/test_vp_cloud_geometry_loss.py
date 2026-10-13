import importlib
from pathlib import Path

import pytest
import torch

from molai.models.bridge import CosineVPSchedule
from molai.models.losses import FullBandEnergyDistance, vp_cloud_geometry_losses


def example_geometry():
    # An isometry from two token coordinates to a spatially zero-mean image.
    basis = torch.tensor([[1., -1., 1., -1.], [1., 1., -1., -1.]])
    tokens = torch.tensor([[[[[1., 0.]], [[0., 1.]], [[-1., 0.]], [[0., -1.]]]]])
    mean = torch.tensor([[[[[2., -2., 3., -3.]]]]])
    sigma = torch.tensor([[2.]])
    offsets = (tokens.flatten(3) @ basis / 2**0.5) * sigma[:, :, None, None]
    fields = torch.cat((mean[:, :, None], mean[:, :, None] + offsets.reshape(1, 1, 4, 1, 1, 4)), dim=2)
    return fields, mean, sigma, tokens


def test_geometry_matches_center_radius_and_angles_with_exact_constructed_map():
    fields, mean, sigma, tokens = example_geometry()
    losses = vp_cloud_geometry_losses(fields, mean, sigma, tokens)
    assert losses["zero_mean_mse"].item() == 0
    assert losses["random_radius"].item() < 1e-10
    assert losses["random_direction"].item() < 1e-10
    assert losses["direction_active_fraction"].item() == 1


def test_radius_penalizes_single_outlier_that_energy_u_statistic_can_miss():
    fields, mean, _, tokens = example_geometry()
    sigma = torch.zeros(1, 1)
    fields = mean[:, :, None].repeat(1, 1, 5, 1, 1, 1)
    pattern = torch.tensor([[[[1., -1., 1., -1.]]]])
    distance = FullBandEnergyDistance(levels=1, include_target_constant=False)
    amplitude = torch.tensor(20., requires_grad=True)
    outlier = fields.clone()
    outlier[:, :, -1] = mean + amplitude * pattern
    targets = mean[:, 0, None].repeat(1, 4, 1, 1, 1)
    baseline = distance(fields[:, 0, 1:], targets, mean[:, 0])
    escaped = distance(outlier[:, 0, 1:], targets, mean[:, 0])
    torch.testing.assert_close(escaped, baseline, atol=1e-5, rtol=0)
    geometry = vp_cloud_geometry_losses(outlier, mean, sigma, tokens)
    assert geometry["random_radius"].item() > 99
    geometry["random_radius"].backward()
    assert amplitude.grad.item() > 9


def test_zero_supervises_posterior_mean_not_clean_and_cannot_move_radius_center():
    fields, mean, sigma, tokens = example_geometry()
    shifted = fields.clone()
    shifted[:, :, 0] += 3
    correct = vp_cloud_geometry_losses(fields, mean, sigma, tokens)
    wrong = vp_cloud_geometry_losses(shifted, mean, sigma, tokens)
    torch.testing.assert_close(wrong["zero_mean_mse"], torch.tensor(9.))
    torch.testing.assert_close(wrong["random_radius"], correct["random_radius"])
    torch.testing.assert_close(wrong["random_direction"], correct["random_direction"])


def test_wrong_angles_are_penalized_even_with_correct_radius():
    fields, mean, sigma, tokens = example_geometry()
    fields = fields.clone()
    fields[:, :, 2] = fields[:, :, 1]
    losses = vp_cloud_geometry_losses(fields, mean, sigma, tokens)
    assert losses["random_radius"].item() < 1e-10
    assert losses["random_direction"].item() > 0.1


@pytest.mark.parametrize("zero_tokens", [False, True])
def test_clean_and_collapsed_states_have_finite_losses_and_gradients(zero_tokens):
    fields = torch.zeros(2, 3, 5, 1, 4, 4, requires_grad=True)
    mean = torch.zeros(2, 3, 1, 4, 4)
    sigma = torch.zeros(2, 3)
    tokens = torch.randn(2, 3, 4, 2, 4)
    if zero_tokens:
        tokens.zero_()
        sigma.fill_(1.)
    losses = vp_cloud_geometry_losses(fields, mean, sigma, tokens)
    total = losses["zero_mean_mse"] + losses["random_radius"] + losses["random_direction"]
    assert total.item() == 0
    total.backward()
    assert torch.isfinite(fields.grad).all()
    assert losses["random_direction"].item() == 0


def test_nonzero_noise_with_collapsed_output_is_finite_and_not_radius_optimum():
    fields = torch.zeros(1, 2, 5, 1, 2, 2, requires_grad=True)
    mean = torch.zeros(1, 2, 1, 2, 2)
    sigma = torch.ones(1, 2)
    tokens = torch.randn(1, 2, 4, 2, 4)
    losses = vp_cloud_geometry_losses(fields, mean, sigma, tokens)
    assert losses["random_radius"].item() > 0
    (losses["random_radius"] + losses["random_direction"]).backward()
    assert torch.isfinite(fields.grad).all()


def test_direction_and_radius_use_only_current_step_tokens():
    fields = torch.randn(2, 3, 5, 1, 4, 4)
    mean = torch.randn(2, 3, 1, 4, 4)
    sigma = torch.ones(2, 3)
    tokens = torch.randn(2, 3, 4, 2, 4)
    actual = vp_cloud_geometry_losses(fields, mean, sigma, tokens)
    for step in range(3):
        expected = vp_cloud_geometry_losses(fields[:, step:step + 1], mean[:, step:step + 1],
                                           sigma[:, step:step + 1], tokens[:, step:step + 1])
        torch.testing.assert_close(actual["step_random_direction"][step], expected["random_direction"])
        torch.testing.assert_close(actual["step_random_radius"][step], expected["random_radius"])
    changed = tokens.clone()
    changed[:, 0] *= -100
    other = vp_cloud_geometry_losses(fields, mean, sigma, changed)
    for key in ("step_random_direction", "step_random_radius"):
        torch.testing.assert_close(other[key][1:], actual[key][1:])


def test_exact_posterior_moments_and_centered_noise_radius(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    trainer = importlib.import_module("train_relative_image_rollout")
    schedule = CosineVPSchedule(levels=64, noise_scale=3.5, zero_mean_noise=True)
    clean = torch.randn(3, 1, 4, 4)
    clean -= clean.mean(dim=(-2, -1), keepdim=True)
    start = torch.tensor([0, 4, 64])
    config = {"cloud_matching": {"transition_jumps": [
        {"max_level": 8, "jump": 1}, {"max_level": 16, "jump": 2},
        {"max_level": 32, "jump": 4}, {"max_level": 64, "jump": 8},
    ]}}
    initial, targets, levels, mean, sigma = trainer.trajectory_targets(
        schedule, clean, start, 4, 4, config, return_moments=True,
    )
    for step in range(4):
        expected_mean, variance = schedule.target_mean_and_variance(clean, initial, start, levels[:, step])
        torch.testing.assert_close(mean[:, step], expected_mean)
        torch.testing.assert_close(sigma[:, step], 3.5 * (variance * 15 / 16).sqrt())
    torch.testing.assert_close(mean[0], clean[0, None].expand(4, 1, 4, 4))
    assert sigma[0].count_nonzero() == 0
    assert sigma[1, -1].item() == 0
    # Full-noise input is independent of x0; the posterior is the VP marginal.
    alpha = schedule.alpha_bar[levels[2]]
    torch.testing.assert_close(mean[2], alpha.sqrt()[:, None, None, None] * clean[2])
    torch.testing.assert_close(sigma[2], 3.5 * ((1 - alpha) * 15 / 16).sqrt())
    assert targets.shape == (3, 4, 4, 1, 4, 4)


@pytest.mark.parametrize("center_noise", [True, False])
def test_transition_teachers_condition_on_actual_branch_inputs_and_detach_only_targets(monkeypatch, center_noise):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    trainer = importlib.import_module("train_relative_image_rollout")
    schedule = CosineVPSchedule(levels=64, noise_scale=3.5, zero_mean_noise=center_noise)
    clean = torch.randn(3, 1, 4, 4, requires_grad=True)
    start = torch.tensor([0, 4, 64])
    initial, levels = trainer.rollout_initial_and_levels(schedule, clean, start, 3,
                                                       {"cloud_matching": {"answer_jump": 2}})
    fields = torch.randn(3, 3, 5, 1, 4, 4, requires_grad=True)
    innovations, means, sigma = trainer.step_posterior_targets(schedule, clean, initial, fields, start, levels)
    assert means.shape == fields.shape
    assert innovations.shape == (3, 3, 4, 1, 4, 4)
    assert not any(value.requires_grad for value in (initial, innovations, means, sigma))
    before = start
    for step in range(3):
        for path in range(5):
            actual_input = initial if step == 0 else fields[:, step - 1, path].detach()
            expected, variance = schedule.target_mean_and_variance(clean, actual_input, before, levels[:, step])
            torch.testing.assert_close(means[:, step, path], expected)
            centering = 15 / 16 if center_noise else 1.
            torch.testing.assert_close(sigma[:, step], 3.5 * (variance * centering).sqrt())
        before = levels[:, step]
    if center_noise:
        torch.testing.assert_close(innovations.mean(dim=(-2, -1)), torch.zeros(3, 3, 4, 1), atol=5e-7, rtol=0)
    assert sigma[0].count_nonzero() == 0
    assert sigma[1, 1:].count_nonzero() == 0
    torch.testing.assert_close(means[0], clean[0].expand(3, 5, 1, 4, 4))
    assert innovations[0].count_nonzero() == 0
    modified = fields.detach().clone()
    modified[2, 0, 3] += 10
    _, new_means, new_sigma = trainer.step_posterior_targets(schedule, clean, initial, modified, start, levels)
    assert not torch.allclose(means[2, 1, 3], new_means[2, 1, 3])
    changed = torch.zeros(3, 3, 5, dtype=torch.bool)
    changed[2, 1, 3] = True
    torch.testing.assert_close(means[~changed], new_means[~changed])
    torch.testing.assert_close(sigma, new_sigma)


def test_per_branch_drift_does_not_change_innovation_geometry_or_u_statistic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    trainer = importlib.import_module("train_relative_image_rollout")
    fields, shared_mean, sigma, tokens = example_geometry()
    means = shared_mean[:, :, None].expand_as(fields).clone()
    targets = torch.randn_like(fields[:, :, 1:])
    noise = torch.cat((torch.zeros_like(tokens[:, :, :1]), tokens), dim=2)
    options = {"loss_mode": "vp_transition_geometry", "u_statistic_weight": 1.,
               "zero_mean_mse_weight": 1., "random_radius_weight": 1., "random_direction_weight": 1.}
    distance = FullBandEnergyDistance(levels=1, include_target_constant=False)
    original = trainer.configured_rollout_losses(fields, targets, shared_mean[:, 0], shared_mean[:, 0],
                                                distance, options, means, sigma, noise)
    drift = torch.randn_like(fields) * 10
    translated = trainer.configured_rollout_losses(fields + drift, targets, shared_mean[:, 0], shared_mean[:, 0],
                                                  distance, options, means + drift, sigma, noise)
    for key in original:
        torch.testing.assert_close(translated[key], original[key], atol=3e-6, rtol=1e-5)
    assert original["zero_mean_mse"].item() == 0
    assert original["random_radius"].item() < 1e-10
    assert original["random_direction"].item() < 1e-10


@pytest.mark.parametrize("zero_weight,radius_weight,direction_weight", [(0.05, 1., 0.1), (1., 1., 1.)])
def test_configured_objective_has_separate_terms_and_matches_weights(
    monkeypatch, zero_weight, radius_weight, direction_weight,
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    trainer = importlib.import_module("train_relative_image_rollout")
    fields, mean, sigma, tokens = example_geometry()
    fields = fields.clone()
    fields[:, :, 0] += 0.5
    fields[:, :, 2] = fields[:, :, 1] + 0.5
    fields.requires_grad_()
    targets = torch.zeros_like(fields[:, :, 1:])
    noise = torch.cat((torch.zeros_like(tokens[:, :, :1]), tokens), dim=2)
    options = {"loss_mode": "vp_transition_geometry", "u_statistic_weight": 1.,
               "zero_mean_mse_weight": zero_weight, "random_radius_weight": radius_weight,
               "random_direction_weight": direction_weight}
    distance = FullBandEnergyDistance(levels=1, include_target_constant=False)
    losses = trainer.configured_rollout_losses(fields, targets, mean[:, 0], mean[:, 0] + 100,
                                               distance, options, mean[:, :, None].expand_as(fields), sigma, noise)
    expected = (losses["u_statistic"] + zero_weight * losses["zero_mean_mse"]
                + radius_weight * losses["random_radius"] + direction_weight * losses["random_direction"])
    torch.testing.assert_close(losses["total"], expected)
    assert all(losses[key].item() > 0 for key in ("zero_mean_mse", "random_radius", "random_direction"))
    assert "zero_token_clean_mse" not in losses
    assert set(trainer.loss_metric_keys(options)).issubset(losses)
    losses["total"].backward()
    assert torch.isfinite(fields.grad).all()


def test_geometry_runs_in_fp32_under_autocast_and_compiles():
    fields, mean, sigma, tokens = example_geometry()
    fields = fields.clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        eager = vp_cloud_geometry_losses(fields, mean, sigma, tokens)
    assert eager["random_direction"].dtype == torch.float32
    # aot_eager traces the whole loss/backward without requiring a Windows C++ compiler.
    compiled = torch.compile(vp_cloud_geometry_losses, backend="aot_eager", fullgraph=True, dynamic=True)
    actual = compiled(fields, mean, sigma, tokens)
    for key in eager:
        torch.testing.assert_close(actual[key], eager[key])
    (actual["zero_mean_mse"] + actual["random_radius"] + actual["random_direction"]).backward()
    assert torch.isfinite(fields.grad).all()


def test_geometry_rejects_misaligned_token_paths():
    fields, mean, sigma, tokens = example_geometry()
    with pytest.raises(ValueError, match="excluding Zero"):
        vp_cloud_geometry_losses(fields, mean, sigma, tokens[:, :, :3])


def test_transition_mode_is_explicit_and_default_coefficients_remain_one(monkeypatch):
    import yaml

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    trainer = importlib.import_module("train_relative_image_rollout")
    options = yaml.safe_load((root / "configs/relative_image_rollout.yaml").read_text(encoding="utf-8"))
    assert options["loss_mode"] == "zero_current_global_vp_ustat"
    trainer.loss_metric_keys(options)
    assert all(options[key] == 1. for key in (
        "zero_current_mean_mse_weight", "u_statistic_weight",
    ))
    # The old cumulative geometry objective must not be silently reinterpreted.
    with pytest.raises(ValueError, match="vp_transition_geometry"):
        trainer.loss_metric_keys({"loss_mode": "vp_geometry"})


@pytest.mark.parametrize("mode", ["vp_transition_geometry", "navigation"])
def test_full_rollout_objective_backpropagates_and_validation_logs_geometry(monkeypatch, mode):
    from molai.models.cloud import MolecularFieldCloud
    from molai.models.relative_rollout import RelativeImageRolloutCloud

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    trainer = importlib.import_module("train_relative_image_rollout")
    cloud = MolecularFieldCloud(
        condition_dim=16, dim=16, heads=4, condition_cross_depth=1,
        noise_token_count=4, noise_token_dim=8, refine_depth=1,
        cvt_kernel_sizes=(3, 5), cvt_grid_sizes=(2, 1), max_resolution=8,
        zero_mean_output=True, gradient_checkpointing=False,
    )
    rollout = RelativeImageRolloutCloud(cloud)
    condition = torch.randn(1, 16, requires_grad=True)
    clean = torch.randn(1, 1, 4, 4)
    clean -= clean.mean(dim=(-2, -1), keepdim=True)
    schedule = CosineVPSchedule(levels=64, noise_scale=3.5, zero_mean_noise=True)
    config = {"cloud_matching": {"answer_jump": 8}}
    options = {"loss_mode": mode, "u_statistic_weight": 1.,
               "zero_mean_mse_weight": 1., "random_radius_weight": 1.,
               "random_direction_weight": 1., "validation_steps": 2, "random_samples": 4}
    start = torch.tensor([64])
    initial, levels = trainer.rollout_initial_and_levels(schedule, clean, start, 2, config)
    noise = trainer.sample_rollout_noise(initial, cloud, 2, 4)
    assert noise[:, :, 0].count_nonzero() == 0
    first = []

    def capture(_module, _inputs, outputs):
        if not first:
            outputs[0].retain_grad()
            first.append(outputs[0])

    handle = cloud.register_forward_hook(capture)
    fields = rollout(initial, condition, steps=2, noise=noise)
    handle.remove()
    targets, mean, sigma = trainer.step_posterior_targets(
        schedule, clean, initial, fields, start, levels, sample_innovations=mode != "navigation")
    assert not any(value.requires_grad for value in (targets, mean, sigma) if value is not None)
    distance = FullBandEnergyDistance(levels=1, include_target_constant=False)
    # Trace both the configured U-statistic and geometry branches together.
    objective = torch.compile(trainer.configured_rollout_losses, backend="aot_eager", fullgraph=True)
    losses = objective(fields, targets, initial, clean, distance, options, mean, sigma, noise)
    final_geometry = (losses["step_zero_mean_mse"][-1] + losses["step_random_radius"][-1]
                      + losses["step_random_direction"][-1])
    if mode != "navigation":
        final_geometry = final_geometry + losses["step_u_statistic"][-1]
    final_geometry.backward()
    assert first[0].grad.abs().sum() > 0
    assert condition.grad.abs().sum() > 0
    assert torch.isfinite(cloud.output_head.weight.grad).all()
    monkeypatch.setattr(trainer, "_condition_from_batch", lambda _runner, batch, _device: batch["condition"])
    before = torch.random.get_rng_state().clone()
    validation, preview = trainer.validate(
        rollout, torch.nn.Identity(), [{"field": clean, "condition": condition.detach()}],
        schedule, distance, config, options, torch.device("cpu"),
    )
    torch.testing.assert_close(before, torch.random.get_rng_state())
    assert set(trainer.loss_metric_keys(options)).issubset(validation)
    assert all(torch.isfinite(torch.tensor(value)) for value in validation.values())
    assert preview.shape == (1, 3, 4, 28)
    if mode == "navigation":
        assert not any("statistic" in key for key in validation)
        for step in (1, 2):
            for key in ("zero_mean_mse", "random_radius", "random_direction"):
                assert f"step_{step}_{key}" in validation


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_bf16_inputs_and_autocast_match_fp32_reference_gradients():
    fields = torch.randn(2, 4, 5, 1, 8, 8).bfloat16()
    mean = torch.randn(2, 4, 5, 1, 8, 8)
    sigma = torch.tensor([[1., 0.5, 0.1, 0.], [2., 1., 0.2, 0.]])
    tokens = torch.randn(2, 4, 4, 4, 8)
    cpu_fields = fields.float().requires_grad_()
    cpu = vp_cloud_geometry_losses(cpu_fields, mean, sigma, tokens)
    (cpu["zero_mean_mse"] + cpu["random_radius"] + cpu["random_direction"]).backward()
    gpu_fields = fields.cuda().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        gpu = vp_cloud_geometry_losses(gpu_fields, mean.cuda(), sigma.cuda(), tokens.cuda())
    (gpu["zero_mean_mse"] + gpu["random_radius"] + gpu["random_direction"]).backward()
    for key in cpu:
        torch.testing.assert_close(gpu[key].cpu(), cpu[key], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(gpu_fields.grad.float().cpu(), cpu_fields.grad, atol=1e-4, rtol=0.02)
