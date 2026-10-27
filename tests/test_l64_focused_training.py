"""Ensure focused training is pure-noise input with the original VP teacher."""
import importlib
from pathlib import Path

import pytest
import torch

from molai.models.bridge import CosineVPSchedule


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("train_relative_image_rollout")


def test_focused_start_is_pure_noise_and_teacher_is_l62(trainer):
    schedule = CosineVPSchedule(levels=64, noise_scale=2.0, zero_mean_noise=True)
    clean = torch.randn(3, 1, 8, 8)
    clean -= clean.mean((-2, -1), keepdim=True)
    config = {"cloud_matching": {"transition_jumps": [{"max_level": 64, "jump": 2}]}}
    start = trainer.sampled_training_levels(schedule, 3, torch.device("cpu"),
                                           {"training_start_level": 64})
    torch.manual_seed(901)
    initial, after = trainer.rollout_initial_and_levels(schedule, clean, start, 1, config)
    torch.manual_seed(901)
    alternate, _ = trainer.rollout_initial_and_levels(schedule, clean * 5, start, 1, config)
    torch.testing.assert_close(initial, alternate, rtol=0, atol=0)
    assert after.eq(62).all()
    assert initial.mean((-2, -1)).abs().max() < 1e-6
    dummy = torch.zeros(3, 1, 5, 1, 8, 8)
    targets, mean, rms = trainer.step_posterior_targets(
        schedule, clean, initial, dummy, start, after)
    expected_mean = (schedule.alpha_bar[62].sqrt() * clean)[:, None, None].expand_as(mean)
    torch.testing.assert_close(mean, expected_mean)
    assert targets.shape == (3, 1, 4, 1, 8, 8)
    assert rms.gt(0).all()


def test_training_sampler_defaults_unchanged_and_validation_bounds(trainer):
    schedule = CosineVPSchedule(levels=64)
    device = torch.device("cpu")
    torch.manual_seed(7)
    actual = trainer.sampled_training_levels(schedule, 20, device, {})
    torch.manual_seed(7)
    torch.testing.assert_close(actual, torch.randint(0, 65, (20,)))
    low = trainer.sampled_validation_levels(schedule, 1000, device,
        {"validation_start_levels": "uniform", "validation_max_start_level": 16})
    assert int(low.min()) == 0 and int(low.max()) == 16
    with pytest.raises(ValueError, match="training_start_level"):
        trainer.sampled_training_levels(schedule, 1, device, {"training_start_level": 65})
    with pytest.raises(ValueError, match="validation level bounds"):
        trainer.sampled_validation_levels(schedule, 1, device,
            {"validation_start_levels": "uniform", "validation_min_start_level": 17,
             "validation_max_start_level": 16})


def test_weighted_levels_preserve_all_levels_with_half_pure_noise(trainer):
    schedule = CosineVPSchedule(levels=64)
    device = torch.device("cpu")
    options = {"training_level_mixture": [
        {"min_level": 64, "max_level": 64, "probability": .5},
        {"min_level": 0, "max_level": 63, "probability": .5},
    ]}
    weights = trainer.training_level_probabilities(schedule, device, options)
    assert weights[64].item() == .5
    torch.testing.assert_close(weights[:64], torch.full((64,), .5 / 64))
    torch.manual_seed(321)
    sampled = trainer.sampled_training_levels(schedule, 100000, device, options, weights)
    assert .49 < float(sampled.eq(64).float().mean()) < .51
    assert len(sampled.unique()) == 65
    with pytest.raises(ValueError, match="cannot be combined"):
        trainer.training_level_probabilities(schedule, device, {**options, "training_start_level": 64})
    with pytest.raises(ValueError, match="positive total"):
        trainer.training_level_probabilities(schedule, device, {"training_level_mixture": []})


def test_difficulty_sampling_keeps_l62_frequent_and_clean_levels_nonzero(trainer):
    schedule = CosineVPSchedule(levels=64)
    device = torch.device("cpu")
    options = {"training_level_difficulty": {
        "signal_floor": .01, "uniform_fraction": .2, "power": 1.,
    }}
    weights = trainer.training_level_probabilities(schedule, device, options)
    assert float(weights.sum()) == pytest.approx(1.)
    assert bool((weights > 0).all()) and bool((weights[1:] >= weights[:-1]).all())
    assert .11 < float(weights[64]) < .13
    assert .09 < float(weights[62]) < .11
    assert float(weights[62]) > 5 / 65
    assert float(weights[:17].sum()) > .07
    torch.manual_seed(97)
    sampled = trainer.sampled_training_levels(schedule, 100000, device, options, weights)
    assert abs(float(sampled.eq(62).float().mean()) - float(weights[62])) < .005
    with pytest.raises(ValueError, match="requires no mixture"):
        trainer.training_level_probabilities(schedule, device, {**options, "training_start_level": 64})
    with pytest.raises(ValueError, match="signal_floor"):
        trainer.training_level_probabilities(schedule, device,
            {"training_level_difficulty": {"signal_floor": 0}})
