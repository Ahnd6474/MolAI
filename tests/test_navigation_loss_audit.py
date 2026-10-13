"""Preserve counterexamples: passing arithmetic tests is not a collapse guarantee."""

import importlib
from pathlib import Path

import pytest
import torch


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("audit_navigation_loss")


def test_collapsed_output_is_penalized_but_geometry_can_have_zero_gradient(audit):
    result = audit.collapsed_gradient_audit(torch.device("cpu"))
    assert result["radius_loss"] > 0.4
    assert result["direction_loss"] > 0.3
    assert result["geometry_gradient_max"] == 0
    assert result["total_gradient_max_symmetric_targets"] < 1e-6


def test_equal_loss_values_do_not_certify_rollout_contraction(audit):
    result = audit.stability_audit(torch.device("cpu"))
    assert result["loss_difference_at_audited_input"] == 0
    assert result["stable_local_gain"] < 1
    assert result["unstable_local_gain"] > 1000


def test_clean_endpoint_does_not_require_random_diversity(audit):
    result = audit.terminal_diversity_audit(torch.device("cpu"))
    assert result[0]["radius_loss"] == 0
    assert result[1]["radius_loss"] > 0.99
    assert result[2]["radius_loss"] > 3.99
    assert all(row["direction_loss"] == 0 for row in result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_pooled_innovations_can_hide_next_state_collapse(audit):
    result = audit.inherited_diversity_audit(torch.device("cuda"))
    assert result["zero_mean_mse"] == 0
    assert result["radius_loss"] < 0.002
    assert result["direction_loss"] < 0.002
    assert abs(result["energy_discrepancy_with_target_constant"]) < 0.01
    assert abs(result["reported_innovation_diversity"] / result["target_innovation_diversity"] - 1) < 0.02
    assert result["actual_next_state_diversity"] == 0
    assert result["fresh_token_output_sensitivity"] == 0
