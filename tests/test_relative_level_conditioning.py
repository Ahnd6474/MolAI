import copy
import importlib
from pathlib import Path

import pytest
import torch
import yaml

from molai.models.bridge import CosineVPSchedule
from molai.models.relative_rollout import (
    RelativeImageRolloutCloud, build_fresh_token_relative_models,
    configure_relative_level_conditioning, restore_token_relative_models,
)
from test_navigation_training_setup import small_config


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("train_relative_image_rollout")


def test_level_adapter_preserves_backbone_and_zero_init_output_then_learns():
    config = small_config()
    config["relative_image_rollout"] = {"conditional_gaussian_tokens": True}
    cloud, encoder, _ = build_fresh_token_relative_models(config, 3, gradient_checkpointing=True)
    cloud.train()
    initial, ms = torch.randn(2, 1, 4, 4), torch.randn(2, 3, 64)
    mask, noise = torch.ones(2, 3, dtype=torch.bool), torch.randn(2, 3, 4, 8)
    before = copy.deepcopy(cloud.state_dict())
    refine_ids = [id(p) for block in cloud.refine_blocks for p in block.parameters()]
    expected = cloud(initial, ms, samples=3, condition_mask=mask, noise=noise)[0]
    options = {"conditional_gaussian_tokens": True, "level_embedding": True}
    configure_relative_level_conditioning(cloud, options)
    actual = cloud(initial, ms, samples=3, condition_mask=mask, noise=noise,
                   current_levels=torch.tensor([64, 1]))[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert refine_ids == [id(p) for block in cloud.refine_blocks for p in block.parameters()]
    for name, value in before.items():
        assert torch.equal(value, cloud.state_dict()[name])
    actual.square().mean().backward()
    assert cloud.level_conditioner.mlp[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        cloud.level_conditioner.mlp[-1].weight.normal_(std=.01)
    embeddings = cloud.level_conditioner(torch.tensor([64, 63]))
    assert not torch.equal(embeddings[0], embeddings[1])
    with pytest.raises(ValueError, match="requires current_levels"):
        cloud(initial, ms, condition_mask=mask)
    config["relative_image_rollout"] = options
    restored, _, audit = restore_token_relative_models(
        {"config": config, "cloud": cloud.state_dict(), "condition_encoder": encoder.state_dict()},
        3, gradient_checkpointing=False)
    assert audit["all_loaded_tensors_identical"]
    assert restored.level_conditioner is not None


def test_level_schedule_reaches_clean_in_exactly_64_steps(trainer):
    config = {"cloud_matching": {"transition_jumps": [{"max_level": 64, "jump": 1}]}}
    schedule = CosineVPSchedule(levels=64, zero_mean_noise=True)
    cloud, _, _ = build_fresh_token_relative_models(small_config(), 3, gradient_checkpointing=False)
    configure_relative_level_conditioning(cloud, {"level_embedding": True})
    clean, start = torch.randn(2, 1, 4, 4), torch.tensor([64, 2])
    _, answers = trainer.rollout_initial_and_levels(schedule, clean, start, 64, config)
    assert answers[0].tolist() == list(range(63, -1, -1))
    assert answers[1, :4].tolist() == [1, 0, 0, 0]
    actual = trainer.level_arguments(cloud, start, answers)["current_levels"]
    assert actual[0].tolist() == list(range(64, 0, -1))
    torch.testing.assert_close(actual, trainer.level_arguments(cloud, start, steps=64)["current_levels"])
    captured = []
    handle = cloud.register_forward_pre_hook(lambda _m, _a, kw: captured.append(kw["current_levels"].clone()),
                                              with_kwargs=True)
    rollout = RelativeImageRolloutCloud(cloud)
    rollout(clean, torch.randn(2, 3, 64), condition_mask=torch.ones(2, 3, dtype=torch.bool),
            steps=2, random_samples=2, current_levels=actual[:, :2])
    handle.remove()
    assert captured[0].tolist() == [64, 2]
    assert captured[1].tolist() == [63, 63, 63, 1, 1, 1]
    with pytest.raises(ValueError, match="requires current_levels"):
        rollout(clean, torch.randn(2, 3, 64), condition_mask=torch.ones(2, 3, dtype=torch.bool))


def test_adam_state_migration_preserves_all_old_moments_and_only_adds_time_layers(trainer):
    cloud, encoder, _ = build_fresh_token_relative_models(small_config(), 3, gradient_checkpointing=False)
    parameters = [p for p in (*cloud.parameters(), *encoder.parameters()) if p.requires_grad]
    old = torch.optim.AdamW(parameters, lr=1e-5)
    sum(p.square().mean() for p in parameters).backward()
    old.step()
    saved = copy.deepcopy(old.state_dict())
    names = trainer.optimizer_parameter_names(cloud, encoder, None)
    configure_relative_level_conditioning(cloud, {"level_embedding": True})
    new = torch.optim.AdamW([p for p in (*cloud.parameters(), *encoder.parameters()) if p.requires_grad], lr=1e-5)
    new_names = trainer.optimizer_parameter_names(cloud, encoder, None)
    audit = trainer.restore_optimizer_adding_level_parameters(new, saved, new_names, names)
    assert audit["restored_parameter_states"] == len(old.state)
    assert len(audit["fresh_parameters"]) == 4
    for parameter in parameters:
        for key, value in old.state[parameter].items():
            assert torch.equal(value, new.state[parameter][key])
    assert all(p not in new.state for p in cloud.level_conditioner.parameters())


def test_level64_config_changes_only_time_conditioning_schedule_and_preview():
    root = Path(__file__).resolve().parents[1] / "configs"
    old = yaml.safe_load((root / "navigation_ustat_difficulty_grid32.yaml").read_text())
    new = yaml.safe_load((root / "navigation_ustat_level64.yaml").read_text())
    expected = {**old, "level_embedding": True, "level_embedding_max_level": 64,
                "long_rollout_steps": 64, "transition_jumps": [{"max_level": 64, "jump": 1}]}
    assert new == expected
