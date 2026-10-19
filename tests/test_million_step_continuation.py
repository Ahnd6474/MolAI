import copy
import importlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from molai.models.relative_rollout import (
    build_fresh_token_relative_models,
    restore_token_relative_models,
)


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("train_relative_image_rollout")


def test_reanchored_lr_retains_adam_moments_and_resumes_exactly(trainer):
    parameter = torch.nn.Parameter(torch.tensor([.2, -.3]))
    original = torch.optim.AdamW([parameter], lr=1e-5)
    parameter.square().sum().backward()
    original.step()
    saved = copy.deepcopy(original.state_dict())
    optimizer = torch.optim.AdamW([parameter], lr=3e-5)
    optimizer.load_state_dict(saved)
    options = {"learning_rate": 3e-5, "min_learning_rate": 1e-5,
               "warmup_start_learning_rate": 1e-5, "warmup_steps": 1000,
               "lr_schedule_start_step": 101316}
    scheduler = trainer.make_lr_scheduler(optimizer, options, 1000000, global_step=101316)
    assert scheduler.last_epoch == 101316
    assert scheduler.get_last_lr()[0] == pytest.approx(1e-5)
    for key, value in saved["state"][0].items():
        assert torch.equal(value, optimizer.state[parameter][key])
    assert scheduler.lr_lambdas[0](102316) * 3e-5 == pytest.approx(3e-5)
    assert scheduler.lr_lambdas[0](1000000) * 3e-5 == pytest.approx(1e-5)
    # Reconstructing at a later checkpoint must not repeat the warmup or reset LR.
    continued = trainer.make_lr_scheduler(optimizer, options, 1000000, global_step=234567)
    assert continued.get_last_lr()[0] == pytest.approx(scheduler.lr_lambdas[0](234567) * 3e-5)
    assert continued.get_last_lr()[0] > 1e-5
    optimizer.zero_grad()
    parameter.square().sum().backward()
    optimizer.step()
    continued.step()
    assert continued.last_epoch == 234568
    assert optimizer.state[parameter]["step"] == 2
    with pytest.raises(ValueError, match="outside"):
        trainer.make_lr_scheduler(optimizer, options, 1000000, global_step=1)


def test_strict_token_resume_restores_both_models_without_reset():
    from test_navigation_training_setup import small_config

    config = small_config()
    config["model"]["condition_dim"] = 64
    cloud, encoder, _ = build_fresh_token_relative_models(config, 3, gradient_checkpointing=False)
    with torch.no_grad():
        for parameter in (*cloud.parameters(), *encoder.parameters()):
            parameter.add_(.234)
    checkpoint = {"config": config, "cloud": cloud.state_dict(), "condition_encoder": encoder.state_dict()}
    restored, ms, audit = restore_token_relative_models(checkpoint, 3, gradient_checkpointing=False)
    assert audit["all_loaded_tensors_identical"]
    for actual, expected in ((restored, cloud), (ms, encoder)):
        for key, value in expected.state_dict().items():
            assert torch.equal(actual.state_dict()[key], value)
    with pytest.raises(RuntimeError, match="size mismatch"):
        restore_token_relative_models(checkpoint, 4)


def test_strict_gaussian_token_resume_restores_controller_and_attention():
    from test_navigation_training_setup import small_config

    config = small_config()
    config["model"]["condition_dim"] = 64
    config["relative_image_rollout"] = {
        "conditional_gaussian_tokens": True,
        "gaussian_image_grid_size": 32,
        "gaussian_controller_heads": 4,
        "gaussian_sigma_min": 0.05,
        "gaussian_sigma_max": 2.0,
        "gaussian_sigma_init": 1.0,
    }
    cloud, encoder, _ = build_fresh_token_relative_models(
        config, 3, gradient_checkpointing=False,
    )
    with torch.no_grad():
        for parameter in (*cloud.parameters(), *encoder.parameters()):
            parameter.add_(0.071)
    checkpoint = {
        "config": config,
        "cloud": cloud.state_dict(),
        "condition_encoder": encoder.state_dict(),
    }
    restored, ms, audit = restore_token_relative_models(
        checkpoint, 3, gradient_checkpointing=False,
    )
    assert audit["all_loaded_tensors_identical"]
    assert restored.random_token_controller.image_grid_size == 32
    assert not hasattr(restored.noise_attention, "key_norm")
    for actual, expected in ((restored, cloud), (ms, encoder)):
        for key, value in expected.state_dict().items():
            assert torch.equal(actual.state_dict()[key], value)


def test_million_config_preserves_objective_and_exact_budget():
    root = Path(__file__).resolve().parents[1]
    original = yaml.safe_load((root / "configs/navigation_ustat.yaml").read_text(encoding="utf-8"))
    current = yaml.safe_load((root / "configs/navigation_ustat_million.yaml").read_text(encoding="utf-8"))
    for key in ("steps", "validation_steps", "random_samples", "batch_size_per_gpu", "refine_depth",
                "loss_mode", "u_statistic_weight", "zero_mean_mse_weight", "random_radius_weight",
                "random_direction_weight", "radius_use_token_rms"):
        assert current[key] == original[key]
    assert current["max_steps"] == 1000000
    assert current["lr_schedule_start_step"] == 101316
    assert current["epochs"] * 8443 >= current["max_steps"]
    assert (current["epochs"] - 1) * 8443 < current["max_steps"]


def test_paired_ms_control_uses_same_noise_and_does_not_advance_rng(trainer, monkeypatch):
    from molai.models.bridge import CosineVPSchedule

    class Runner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cloud = type("Cloud", (), {"noise_token_count": 4, "noise_token_dim": 8})()
            self.calls = []

        def forward(self, initial, condition, condition_mask, steps, random_samples, noise):
            self.calls.append((initial.clone(), condition.clone(), condition_mask.clone(), noise.clone()))
            return condition[:, :1, :16].reshape(2, 1, 1, 1, 4, 4).expand(-1, steps, 5, -1, -1, -1)

    clean = torch.randn(2, 1, 4, 4)
    tokens = torch.randn(2, 2, 16)
    tokens[:, 0] = clean.flatten(1)
    mask = torch.ones(2, 2, dtype=torch.bool)
    monkeypatch.setattr(trainer, "_condition_from_batch", lambda *_: (tokens, mask))
    runner = Runner()
    schedule = CosineVPSchedule(levels=64, noise_scale=3.5, zero_mean_noise=True)
    options = {"long_rollout_steps": 20, "random_samples": 4, "condition_control_batches": 1}
    before = torch.random.get_rng_state().clone()
    result = trainer.condition_control_metrics(runner, torch.nn.Identity(), [{"field": clean}],
                                               schedule, options, torch.device("cpu"))
    assert torch.equal(before, torch.random.get_rng_state())
    assert len(runner.calls) == 2
    for index in (0, 2, 3):
        assert torch.equal(runner.calls[0][index], runner.calls[1][index])
    assert torch.equal(runner.calls[1][1], runner.calls[0][1].roll(1, 0))
    assert result["molecules"] == 2
    assert result["true_random_mse"] == 0
    assert result["true_random_correlation"] == pytest.approx(1.)
    assert result["random_mse_reduction"] > 0


@pytest.mark.parametrize("split_first_transition", [False, True])
def test_actual_training_stops_exactly_at_update_budget_and_preserves_resume(
    trainer, monkeypatch, tmp_path, split_first_transition,
):
    from test_navigation_training_setup import small_config

    fields, spectra, output = (tmp_path / name for name in ("fields", "spectra", "output"))
    fields.mkdir()
    spectra.mkdir()
    records = 20
    torch.save({"field": torch.randn(records, 1, 4, 4), "smiles": ["C"] * records,
                "inchikey14": [f"KEY{i:011d}" for i in range(records)],
                "electron_count": torch.ones(records), "formal_charge": torch.zeros(records)},
               fields / "fields.pt")
    (fields / "manifest.json").write_text(json.dumps({"shards": [{"file": "fields.pt", "records": records}]}))
    torch.save({"mz": torch.ones(records * 4), "intensity": torch.ones(records * 4),
                "peak_offsets": torch.arange(records + 1) * 4,
                "spectrum_offsets": torch.arange(records + 1),
                "metadata": torch.zeros(records, 3), "precursor_mz": torch.ones(records)}, spectra / "spectra.pt")
    (spectra / "manifest.json").write_text(json.dumps({"format": "molai-spectrum-field-v1",
        "field_source": "../fields", "records": records, "metadata_dim": 3, "seed": 11,
        "shards": [{"file": "spectra.pt", "records": records}]}))
    config = small_config()
    config["model"]["condition_dim"] = 64
    config["training"].update(validation_fraction=.2)
    config["cloud_matching"] = {"full_band_levels": 3, "transition_jumps": [{"max_level": 64, "jump": 8}],
        "noise_schedule": {"type": "cosine_vp", "levels": 64, "noise_scale": 3.5, "zero_mean_noise": True}}
    options = {"token_conditioning": True, "alignment_steps": 0, "max_postpool_peak_tokens": 8192,
               "steps": 1, "validation_steps": 1, "random_samples": 4, "batch_size_per_gpu": 2,
               "validation_batch_size_per_gpu": 2, "gradient_accumulation_steps": 1,
               "gradient_checkpointing": False, "compile_rollout": False,
               "epochs": 1, "max_steps": 3, "lr_schedule_start_step": 1,
               "warmup_start_learning_rate": 1e-5, "learning_rate": 3e-5, "min_learning_rate": 1e-5,
               "warmup_steps": 1, "weight_decay": .0001, "loss_mode": "vp_transition_geometry",
               "zero_mean_mse_weight": 1., "random_radius_weight": 1., "random_direction_weight": 1.,
               "u_statistic_weight": 1., "log_every": 1}
    config["relative_image_rollout"] = options
    cloud, encoder, _ = build_fresh_token_relative_models(config, 3, gradient_checkpointing=False)
    parameters = [p for p in (*cloud.parameters(), *encoder.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)
    sum(p.square().mean() for p in parameters).backward()
    optimizer.step()
    checkpoint = tmp_path / "source.pt"
    torch.save({"cloud": cloud.state_dict(), "condition_encoder": encoder.state_dict(), "config": config,
                "optimizer": optimizer.state_dict(), "scheduler": {}, "epoch": 0, "global_step": 1,
                "epoch_complete": False, "next_microbatch": 1,
                "training_kind": "relative-image-feedback-rollout"}, checkpoint)
    phase = tmp_path / "phase.yaml"
    if split_first_transition:
        root = Path(__file__).resolve().parents[1]
        split_options = yaml.safe_load(
            (root / "configs/navigation_ustat_million_jump2.yaml").read_text(encoding="utf-8"))
        # Change only the new phase; the checkpoint must retain its old schedule.
        options = {**options, "transition_jumps": split_options["transition_jumps"]}
    phase.write_text(yaml.safe_dump(options))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(trainer, "_postpool_peak_token_counts", lambda dataset, *_: torch.ones(len(dataset)))
    monkeypatch.setattr(trainer, "validate", lambda *_args, **_kwargs: ({"total": 0.}, torch.zeros(1, 3, 4, 28)))
    monkeypatch.setattr(trainer, "long_rollout_preview", lambda *_: None)
    monkeypatch.setattr("sys.argv", ["train", "--data", str(spectra), "--resume", str(checkpoint),
                                     "--config", str(phase), "--output", str(output), "--workers", "0"])
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        trainer.main()
    finally:
        torch.set_num_threads(previous_threads)
    result = torch.load(output / "latest.pt", weights_only=False)
    assert result["global_step"] == 3
    assert result["next_microbatch"] == 3 and not result["epoch_complete"]
    assert result["scheduler"]["last_epoch"] == 3
    assert len(result["rng_states"]) == 1
    assert max(float(state["step"]) for state in result["optimizer"]["state"].values()) == 3
    assert json.loads((output / "training_finished.json").read_text())["target_reached"]
    audit = json.loads((output / "resume_state.json").read_text())
    assert audit["source_global_step"] == 1 and audit["optimizer_state_restored"]
    if split_first_transition:
        assert result["config"]["cloud_matching"]["transition_jumps"] == options["transition_jumps"]
        preparation = json.loads((output / "preparation.json").read_text())
        assert preparation["transition_schedule"]["overridden"]
        assert preparation["transition_schedule"]["previous"] == [{"max_level": 64, "jump": 8}]
