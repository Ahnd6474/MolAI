import importlib
from pathlib import Path

import pytest
import torch

from molai.models.bridge import CosineVPSchedule


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("diagnose_first_transition_oracle")


def test_first_oracle_is_exact_posterior_and_zero_mean_noise(diagnostic):
    schedule = CosineVPSchedule(levels=64, noise_scale=3.59, zero_mean_noise=True)
    clean = torch.randn(2, 1, 8, 8)
    clean -= clean.mean((-2, -1), keepdim=True)
    initial = torch.randn_like(clean)
    make = lambda: torch.Generator().manual_seed(123)
    fields = diagnostic.oracle_first_state(schedule, clean, initial, next_level=56,
                                            paths=5, generator=make())
    expected_mean = schedule.alpha_bar[56].sqrt() * clean
    torch.testing.assert_close(fields[:, 0], expected_mean)
    torch.testing.assert_close(fields.mean((-2, -1)), torch.zeros(2, 5, 1), atol=2e-7, rtol=0)
    changed_input = diagnostic.oracle_first_state(schedule, clean, initial * 12, next_level=56,
                                                   paths=5, generator=make())
    torch.testing.assert_close(fields, changed_input, rtol=0, atol=0)
    assert not torch.equal(fields[:, 1], fields[:, 2])


def test_replaces_only_first_output_and_keeps_later_tokens(diagnostic):
    class Cloud:
        cache_condition_kv = False

        def __init__(self):
            self.calls = []

        def __call__(self, current, condition, *, samples, noise, condition_mask):
            self.calls.append((current.clone(), noise.clone()))
            increment = noise.sum((-2, -1))[:, :, None, None, None]
            return current[:, None].expand(-1, samples, -1, -1, -1) + increment, None, None

    cloud = Cloud()
    initial = torch.randn(1, 1, 4, 4)
    condition, mask = torch.randn(1, 3, 8), torch.ones(1, 3, dtype=torch.bool)
    tokens = torch.randn(1, 4, 5, 2, 3)
    saved = tokens.clone()
    baseline = diagnostic.paired_rollout(cloud, initial, condition, mask, tokens)
    calls = cloud.calls
    assert len(calls) == 4
    cloud.calls = []
    replacement = torch.randn(1, 5, 1, 4, 4)
    oracle = diagnostic.paired_rollout(cloud, initial, condition, mask, tokens, replacement)
    assert len(cloud.calls) == 3
    torch.testing.assert_close(oracle[:, 0], replacement, rtol=0, atol=0)
    for call, reference in zip(cloud.calls, calls[1:], strict=True):
        torch.testing.assert_close(call[1], reference[1], rtol=0, atol=0)
    torch.testing.assert_close(oracle[:, 1:] - oracle[:, :-1],
                               baseline[:, 1:] - baseline[:, :-1])
    torch.testing.assert_close(tokens, saved, rtol=0, atol=0)
    torch.testing.assert_close(oracle[:, 1:, 0], replacement[:, :1].expand(-1, 3, -1, -1, -1))
