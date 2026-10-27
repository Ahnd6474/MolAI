import copy
import importlib
from pathlib import Path

import pytest
import torch
import yaml

from molai.models.relative_rollout import (
    build_fresh_token_relative_models, expand_relative_refine_depth, restore_token_relative_models,
)
from test_navigation_training_setup import small_config


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("train_relative_image_rollout")


def trained_model():
    config = small_config()
    config["model"]["refine_gate_init"] = .1
    config["relative_image_rollout"] = {"conditional_gaussian_tokens": True, "level_embedding": True}
    cloud, encoder, _ = build_fresh_token_relative_models(config, 3, gradient_checkpointing=True)
    with torch.no_grad():
        cloud.level_conditioner.mlp[-1].weight.normal_(std=.01)
        cloud.level_conditioner.mlp[-1].bias.normal_(std=.01)
    return config, cloud, encoder


def test_refine8_identity_old_weights_level_rows_and_new_gate_gradients():
    config, cloud, encoder = trained_model()
    cloud.eval()
    initial, ms = torch.randn(2, 1, 4, 4), torch.randn(2, 3, 64)
    kwargs = {"samples": 3, "condition_mask": torch.ones(2, 3, dtype=torch.bool),
              "noise": torch.randn(2, 3, 4, 8), "current_levels": torch.tensor([64, 1])}
    expected = cloud(initial, ms, **kwargs)[0].detach()
    before = copy.deepcopy(cloud.state_dict())
    identities = [id(p) for block in cloud.refine_blocks for p in block.parameters()]
    audit = expand_relative_refine_depth(cloud, config, 8)
    assert audit["added_blocks"] == 5 and len(cloud.refine_blocks) == 8
    assert identities == [id(p) for block in cloud.refine_blocks[:3] for p in block.parameters()]
    for name, old in before.items():
        value = cloud.state_dict()[name]
        if name.startswith("level_conditioner.mlp.2."):
            assert torch.equal(value[:old.shape[0]], old)
            assert value[old.shape[0]:].count_nonzero() == 0
        else:
            assert torch.equal(value, old)
    for block in cloud.refine_blocks[3:]:
        assert block.attention_gate.item() == 0 and block.ffn_gate.item() == 0
        assert block.attention.grid_sizes == cloud.refine_blocks[0].attention.grid_sizes
    cloud.train()
    actual = cloud(initial, ms, **kwargs)[0]
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    actual.square().mean().backward()
    for block in cloud.refine_blocks[3:]:
        assert torch.isfinite(block.attention_gate.grad) and block.attention_gate.grad.abs() > 0
        assert torch.isfinite(block.ffn_gate.grad) and block.ffn_gate.grad.abs() > 0
    restored, _, restored_audit = restore_token_relative_models(
        {"config": config, "cloud": cloud.state_dict(), "condition_encoder": encoder.state_dict()},
        3, gradient_checkpointing=False)
    assert len(restored.refine_blocks) == 8 and restored_audit["all_loaded_tensors_identical"]
    with pytest.raises(ValueError, match="cannot remove"):
        expand_relative_refine_depth(cloud, config, 3)


def test_refine8_adam_restores_old_states_pads_level_moments_and_updates(trainer):
    config, cloud, encoder = trained_model()
    parameters = [p for p in (*cloud.parameters(), *encoder.parameters()) if p.requires_grad]
    old = torch.optim.AdamW(parameters, lr=1e-5)
    sum(p.square().mean() for p in parameters).backward()
    old.step()
    saved = copy.deepcopy(old.state_dict())
    names = trainer.optimizer_parameter_names(cloud, encoder, None)
    expand_relative_refine_depth(cloud, config, 8)
    all_parameters = [p for p in (*cloud.parameters(), *encoder.parameters()) if p.requires_grad]
    new = torch.optim.AdamW(all_parameters, lr=1e-5)
    new_names = trainer.optimizer_parameter_names(cloud, encoder, None)
    audit = trainer.restore_optimizer_adding_level_parameters(new, saved, new_names, names)
    assert len(audit["expanded_parameter_states"]) == 2
    assert all(name.startswith("cloud.refine_blocks.") for name in audit["fresh_parameters"])
    by_name = {**{f"cloud.{n}": p for n, p in cloud.named_parameters()},
               **{f"condition_encoder.{n}": p for n, p in encoder.named_parameters()}}
    for old_id, name in zip(saved["param_groups"][0]["params"], names[0], strict=True):
        if old_id not in saved["state"]:
            continue
        state = new.state[by_name[name]]
        for key, value in saved["state"][old_id].items():
            actual = state[key]
            if key != "step" and actual.shape != value.shape:
                assert torch.equal(actual[:value.shape[0]], value)
                assert actual[value.shape[0]:].count_nonzero() == 0
            else:
                assert torch.equal(actual, value)
    new.zero_grad()
    sum(p.square().mean() for p in all_parameters).backward()
    new.step()
    assert all(torch.isfinite(p).all() for p in all_parameters)
    restored, restored_encoder, _ = restore_token_relative_models(
        {"config": config, "cloud": cloud.state_dict(), "condition_encoder": encoder.state_dict()},
        3, gradient_checkpointing=False)
    restored_optimizer = torch.optim.AdamW(
        [p for p in (*restored.parameters(), *restored_encoder.parameters()) if p.requires_grad], lr=1e-5)
    resume_audit = trainer.restore_optimizer_adding_level_parameters(
        restored_optimizer, new.state_dict(), trainer.optimizer_parameter_names(restored, restored_encoder, None),
        new_names)
    assert not resume_audit["fresh_parameters"] and not resume_audit["expanded_parameter_states"]
    assert len(restored_optimizer.state) == len(new.state)


def test_refine8_phase_changes_only_depth():
    root = Path(__file__).resolve().parents[1] / "configs"
    previous = yaml.safe_load((root / "navigation_ustat_level64.yaml").read_text())
    actual = yaml.safe_load((root / "navigation_ustat_level64_refine8.yaml").read_text())
    assert actual == {**previous, "refine_depth": 8}
