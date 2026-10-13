import copy
import importlib
from pathlib import Path

import pytest
import torch
import yaml


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("train_relative_image_rollout")


def configuration():
    root = Path(__file__).resolve().parents[1]
    model = yaml.safe_load((root / "configs/model.yaml").read_text(encoding="utf-8"))
    phase = yaml.safe_load((root / "configs/navigation_ustat_million_jump2.yaml").read_text(encoding="utf-8"))
    return model, phase


def test_changes_only_levels_above_56_and_reaches_clean_in_23_steps(trainer):
    config, options = configuration()
    old = copy.deepcopy(config)
    before = trainer._transition_jumps(torch.arange(65), config)
    audit = trainer.apply_transition_schedule(config, options)
    after = trainer._transition_jumps(torch.arange(65), config)
    torch.testing.assert_close(after[:57], before[:57], rtol=0, atol=0)
    assert torch.equal(after[57:], torch.full((8,), 2))
    assert audit["overridden"] and audit["previous"] == old["cloud_matching"]["transition_jumps"]
    expected = copy.deepcopy(old)
    expected["cloud_matching"]["transition_jumps"] = options["transition_jumps"]
    assert config == expected
    current, path = torch.tensor([64]), []
    while current.item() > 0:
        current = (current - trainer._transition_jumps(current, config)).clamp_min(0)
        path.append(current.item())
    assert path[:4] == [62, 60, 58, 56]
    assert path == [62, 60, 58, 56, 48, 40, 32, 28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert len(path) == options["long_rollout_steps"] == 23


def test_phase_preserves_architecture_objective_optimizer_and_budget():
    root = Path(__file__).resolve().parents[1]
    old = yaml.safe_load((root / "configs/navigation_ustat_million.yaml").read_text(encoding="utf-8"))
    _, new = configuration()
    for key, value in old.items():
        if key != "long_rollout_steps":
            assert new[key] == value, key
    assert set(new) - set(old) == {"transition_jumps"}


def test_override_rejects_invalid_schedule_without_mutating_config(trainer):
    config, _ = configuration()
    previous = copy.deepcopy(config)
    for bands in ([], [{"max_level": 56, "jump": 2}],
                  [{"max_level": 64, "jump": 0}],
                  [{"max_level": 64, "jump": 2}, {"max_level": 56, "jump": 8}]):
        with pytest.raises(ValueError):
            trainer.apply_transition_schedule(config, {"transition_jumps": bands})
        assert config == previous
    audit = trainer.apply_transition_schedule(config, {})
    assert config == previous and not audit["overridden"]
