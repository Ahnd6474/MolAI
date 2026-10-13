import copy

import pytest
import torch
from test_navigation_training_setup import small_config

from molai.models.attention import CvTMixerBlock
from molai.models.relative_rollout import (
    RelativeImageRolloutCloud,
    build_fresh_token_relative_models,
)


def block(gate=None):
    return CvTMixerBlock(16, 4, kernel_sizes=[3, 5], grid_sizes=[2, 1], residual_gate_init=gate)


def test_legacy_refine_retains_checkpoint_keys_and_exact_formula():
    model = block()
    assert not any("gate" in name for name in model.state_dict())
    other = block()
    other.load_state_dict(model.state_dict(), strict=True)
    value = torch.randn(2, 4, 4, 16)
    hidden = value + model.attention(model.norm(value))
    torch.testing.assert_close(model(value), hidden + model.ff(hidden), rtol=0, atol=0)


def test_zero_gates_are_identity_but_receive_gradient():
    model = block(0.)
    value = torch.randn(2, 4, 4, 16, requires_grad=True)
    output = model(value)
    torch.testing.assert_close(output, value, rtol=0, atol=0)
    output.square().mean().backward()
    for gate in (model.attention_gate, model.ffn_gate):
        assert gate.ndim == 0 and gate.requires_grad
        assert torch.isfinite(gate.grad) and gate.grad.abs() > 0
    torch.testing.assert_close(model.attention.attention.in_proj_weight.grad,
                               torch.zeros_like(model.attention.attention.in_proj_weight))


def test_both_gates_scale_their_own_updates_and_are_compilable():
    model = block(.1)
    with torch.no_grad():
        model.ffn_gate.fill_(.2)
    value = torch.randn(2, 4, 4, 16)
    hidden = value + .1 * model.attention(model.norm(value))
    expected = hidden + .2 * model.ff(hidden)
    torch.testing.assert_close(model(value), expected, rtol=0, atol=0)
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    actual = compiled(value)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert model.attention_gate.grad is not None and model.ffn_gate.grad is not None


def test_gated_scratch_rollout_registers_six_optimizer_parameters_and_roundtrips():
    config = copy.deepcopy(small_config())
    config["model"]["refine_gate_init"] = .1
    cloud, encoder, _ = build_fresh_token_relative_models(config, 3, gradient_checkpointing=True)
    gates = [(name, parameter) for name, parameter in cloud.named_parameters()
             if name.startswith("refine_blocks.") and name.endswith("_gate")]
    assert len(gates) == 6
    assert len({id(parameter) for _, parameter in gates}) == 6
    assert all(parameter.ndim == 0 and parameter.requires_grad for _, parameter in gates)
    for _, parameter in gates:
        assert float(parameter.detach()) == pytest.approx(.1)
    initial = torch.randn(1, 1, 4, 4)
    fields = RelativeImageRolloutCloud(cloud)(initial, torch.randn(1, 3, 64), steps=2, random_samples=4,
                                            condition_mask=torch.ones(1, 3, dtype=torch.bool))
    assert fields.shape == (1, 2, 5, 1, 4, 4)
    fields.square().mean().backward()
    for _, parameter in gates:
        assert parameter.grad is not None and torch.isfinite(parameter.grad)
    restored, restored_encoder, _ = build_fresh_token_relative_models(config, 3, gradient_checkpointing=False)
    restored.load_state_dict(cloud.state_dict(), strict=True)
    restored_encoder.load_state_dict(encoder.state_dict(), strict=True)


@pytest.mark.parametrize("gate", [float("nan"), float("inf")])
def test_nonfinite_gate_initialization_is_rejected(gate):
    with pytest.raises(ValueError, match="finite"):
        block(gate)
