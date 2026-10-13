import copy
import importlib
from pathlib import Path

import pytest
import torch
import yaml

from molai.models.bridge import CosineVPSchedule
from molai.models.relative_rollout import (
    RelativeImageRolloutCloud,
    ScalarGatedRandomMemoryAttention,
    build_fresh_token_relative_models,
    build_ms_reset_relative_models,
)


def small_config():
    return {"model": {"field_channels": 1, "condition_dim": 384, "fullres_dim": 16,
                      "heads": 4, "condition_cross_depth": 1, "refine_depth": 3,
                      "noise_token_count": 4, "noise_token_dim": 8, "noise_temperature": 0.8,
                      "noise_gate_init": 0.02, "condition_gate_init": 0.5, "ffn_ratio": 2.,
                      "cvt_grid_sizes": [2, 1], "cvt_kernel_sizes": [3, 5], "zero_mean_output": True},
            "spectrum_encoder": {"heads": 4, "peak_conv_stages": 2, "spectrum_layers": 2,
                                 "peak_position_dim": 8, "mz_bin_width": 1., "mz_upper_bound": 10.},
            "training": {"peak_chunk_size": 8}}


def test_scratch_model_keeps_relative_topology_and_initializes_new_weights():
    config = small_config()
    before = copy.deepcopy(config)
    torch.manual_seed(117)
    cloud, encoder, audit = build_fresh_token_relative_models(config, 3, gradient_checkpointing=False)
    torch.manual_seed(118)
    second, _, _ = build_fresh_token_relative_models(config, 3, gradient_checkpointing=False)
    assert config == before
    assert audit["loaded_parameter_tensors"] == 0
    assert audit["initialization"] == "random"
    assert len(cloud.refine_blocks) == 3
    assert len(encoder.peak_conv_blocks) == 2
    assert len(encoder.spectrum_attention_blocks) == 2
    assert cloud.condition_blocks[0].attention.kdim == 64
    assert cloud.condition_blocks[0].attention.vdim == 64
    assert isinstance(cloud.noise_attention, ScalarGatedRandomMemoryAttention)
    assert not hasattr(cloud, "condition_plane")
    assert not torch.equal(cloud.output_head.weight, second.output_head.weight)
    assert not any(parameter.requires_grad for parameter in cloud.readout.parameters())
    rollout = RelativeImageRolloutCloud(cloud)
    initial = torch.randn(1, 1, 4, 4)
    initial -= initial.mean(dim=(-2, -1), keepdim=True)
    fields = rollout(initial, torch.randn(1, 3, 64), condition_mask=torch.ones(1, 3, dtype=torch.bool),
                     steps=2, random_samples=4)
    assert fields.shape == (1, 2, 5, 1, 4, 4)
    fields[:, -1].square().mean().backward()
    assert torch.isfinite(cloud.output_head.weight.grad).all()
    assert cloud.refine_blocks[0].attention.attention.in_proj_weight.grad is not None


def test_ms_only_reset_preserves_every_cloud_tensor_and_reinitializes_encoder():
    from molai.data.spectrum_metadata import SpectrumMetadataSchema
    from molai.models.condition import bind_spectrum_metadata

    original = small_config()
    original["model"]["condition_dim"] = 64
    cloud, old_encoder, _ = build_fresh_token_relative_models(original, 3, gradient_checkpointing=False)
    # Sentinel weights also detect accidental resets of condition cross/refine gates.
    with torch.no_grad():
        for parameter in cloud.parameters():
            parameter.add_(.123)
        for parameter in old_encoder.parameters():
            parameter.fill_(7.)
    checkpoint = {"config": copy.deepcopy(original), "cloud": cloud.state_dict(),
                  "condition_encoder": old_encoder.state_dict()}
    config = copy.deepcopy(original)
    schema = SpectrumMetadataSchema({"ionization_mode": ["positive"], "adduct": ["[m+h]+"],
                                     "instrument_type": ["timstof"],
                                     "collision_energy_orig_units": ["ev"]}, 2, 2)
    config["spectrum_encoder"].update(metadata_fusion="preconv_cross", metadata_cross_gate_init=.1)
    bind_spectrum_metadata(config["spectrum_encoder"], {"metadata_dim": len(schema.feature_names),
                                                       "metadata_schema": schema.to_dict(),
                                                       "metadata_features": schema.feature_names})
    restored, encoder, audit = build_ms_reset_relative_models(
        checkpoint, config, len(schema.feature_names), gradient_checkpointing=False)
    assert audit["loaded_ms_tensors"] == 0
    assert audit["all_cloud_tensors_identical"]
    for key, value in cloud.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value), key
    assert encoder.metadata_fusion == "preconv_cross"
    assert not torch.equal(encoder.peak_embedding.output.weight, old_encoder.peak_embedding.output.weight)
    assert restored.output_head.weight.requires_grad
    assert encoder.peak_embedding.output.weight.requires_grad
    # The new metadata layout must remain reconstructible from the saved config.
    second_cloud, second_encoder, _ = build_fresh_token_relative_models(
        config, len(schema.feature_names), gradient_checkpointing=False)
    second_cloud.load_state_dict(restored.state_dict(), strict=True)
    second_encoder.load_state_dict(encoder.state_dict(), strict=True)
    changed = copy.deepcopy(config)
    changed["model"]["refine_depth"] = 2
    with pytest.raises(ValueError, match="architecture"):
        build_ms_reset_relative_models(checkpoint, changed, len(schema.feature_names))
    broken = dict(checkpoint, cloud={key: value for key, value in checkpoint["cloud"].items()
                                    if key != "output_head.weight"})
    with pytest.raises(RuntimeError, match="Missing key"):
        build_ms_reset_relative_models(broken, config, len(schema.feature_names))
    legacy = dict(checkpoint, cloud={**checkpoint["cloud"], "condition_plane.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="token-conditioned"):
        build_ms_reset_relative_models(legacy, config, len(schema.feature_names))


def test_current_mean_ustat_run_settings_are_one_step_and_four_plus_zero():
    root = Path(__file__).resolve().parents[1]
    options = yaml.safe_load((root / "configs/relative_image_rollout.yaml").read_text(encoding="utf-8"))
    assert options["steps"] == options["validation_steps"] == 1
    assert options["refine_depth"] == 3
    assert options["refine_gate_init"] == 0.1
    assert options["random_samples"] == 4
    assert options["alignment_steps"] == 0
    assert options["long_rollout_samples"] == 4
    assert options["long_rollout_steps"] == 20
    assert options["loss_mode"] == "zero_current_global_vp_ustat"
    assert options["u_statistic_weight"] == 1.0
    assert not any(key in options for key in ("random_bound_weight", "random_radius_weight",
                                              "random_direction_weight"))


def test_navigation_ustat_fresh_run_config_keeps_four_unit_weight_terms(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    trainer = importlib.import_module("train_relative_image_rollout")
    options = yaml.safe_load((root / "configs/navigation_ustat.yaml").read_text(encoding="utf-8"))
    assert options["loss_name"] == "navigation_ustat"
    assert options["loss_mode"] == "vp_transition_geometry"
    assert options["steps"] == options["validation_steps"] == 1
    assert options["alignment_steps"] == 0
    assert options["random_samples"] == 4
    assert options["batch_size_per_gpu"] == 16
    assert options["refine_depth"] == 3
    assert options["refine_gate_init"] == .1
    assert options["radius_use_token_rms"] is True
    for key in ("u_statistic_weight", "zero_mean_mse_weight", "random_radius_weight",
                "random_direction_weight"):
        assert options[key] == 1.
    assert "zero_current_mean_mse_weight" not in options
    assert "zero_token_clean_mse_weight" not in options
    assert {"u_statistic", "zero_mean_mse", "random_radius", "random_direction"}.issubset(
        trainer.loss_metric_keys(options))


def test_navigation_is_exactly_three_terms_and_never_calls_distance(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    trainer = importlib.import_module("train_relative_image_rollout")
    fields = torch.randn(2, 2, 5, 1, 4, 4, requires_grad=True)
    means = torch.randn_like(fields)
    sigma = torch.ones(2, 2)
    noise = torch.randn(2, 2, 5, 4, 8)
    noise[:, :, 0] = 0
    options = {"loss_mode": "navigation", "zero_mean_mse_weight": 1.,
               "random_radius_weight": 1., "random_direction_weight": 1.}

    def forbidden_distance(*_args):
        raise AssertionError("U-statistic must never be computed")

    compiled = torch.compile(trainer.configured_rollout_losses, backend="aot_eager", fullgraph=True)
    result = compiled(fields, None, means[:, 0, 0], means[:, 0, 0], forbidden_distance,
                      options, means, sigma, noise)
    assert not any("statistic" in key for key in result)
    assert "u_statistic" not in trainer.loss_metric_keys(options)
    expected = result["zero_mean_mse"] + result["random_radius"] + result["random_direction"]
    torch.testing.assert_close(result["total"], expected)
    result["total"].backward()
    assert torch.isfinite(fields.grad).all()
    for key in ("step_zero_mean_mse", "step_random_radius", "step_random_direction"):
        assert result[key].shape == (2,)


def test_navigation_teacher_does_not_sample_reference_cloud_or_advance_rng(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    trainer = importlib.import_module("train_relative_image_rollout")
    schedule = CosineVPSchedule(levels=64, noise_scale=3.5, zero_mean_noise=True)
    clean, initial = torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 4)
    fields = torch.randn(2, 2, 5, 1, 4, 4)
    before = torch.random.get_rng_state().clone()
    targets, means, sigma = trainer.step_posterior_targets(
        schedule, clean, initial, fields, torch.tensor([64, 16]),
        torch.tensor([[56, 48], [14, 12]]), sample_innovations=False,
    )
    assert targets is None
    assert means.shape == fields.shape and sigma.shape == (2, 2)
    torch.testing.assert_close(before, torch.random.get_rng_state())


def test_uniform_validation_and_small_long_preview_are_rng_isolated(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    trainer = importlib.import_module("train_relative_image_rollout")
    cloud, _, _ = build_fresh_token_relative_models(small_config(), 3, gradient_checkpointing=False)
    rollout = RelativeImageRolloutCloud(cloud)
    schedule = CosineVPSchedule(levels=64, noise_scale=3.5, zero_mean_noise=True)
    device = torch.device("cpu")
    options = {"validation_start_levels": "uniform", "random_samples": 4,
               "long_rollout_samples": 1, "long_rollout_steps": 4}
    levels = trainer.sampled_validation_levels(schedule, 2048, device, options)
    assert levels.min() == 0 and levels.max() == 64
    pure = trainer.sampled_validation_levels(schedule, 3, device, {})
    assert pure.tolist() == [64, 64, 64]
    batch = {"field": torch.randn(2, 1, 4, 4), "condition": torch.randn(2, 3, 64)}
    batch["field"] -= batch["field"].mean(dim=(-2, -1), keepdim=True)
    mask = torch.ones(2, 3, dtype=torch.bool)
    monkeypatch.setattr(trainer, "_condition_from_batch", lambda *_args: (batch["condition"], mask))
    before = torch.random.get_rng_state().clone()
    preview = trainer.long_rollout_preview(rollout, torch.nn.Identity(), [batch], schedule, options, device)
    torch.testing.assert_close(before, torch.random.get_rng_state())
    images, selected = preview
    assert selected == [1, 2, 4]
    assert images.shape == (5, 3, 4, 20)
    assert torch.isfinite(images).all()
    trainer.save_long_rollout_preview(preview, tmp_path, 0)
    assert len(list((tmp_path / "long_rollouts" / "epoch-0001").glob("*.png"))) == 1
