import importlib
from pathlib import Path

import pytest
import torch

from molai.models.bridge import CosineVPSchedule
from molai.models.cloud import MolecularFieldCloud
from molai.models.relative_rollout import (
    RelativeImageRolloutCloud,
    ScalarGatedRandomMemoryAttention,
    TokenConditionRelativeCloud,
    build_relative_models,
    build_token_relative_models,
    configure_conditional_gaussian_tokens,
)


def small_cloud():
    return MolecularFieldCloud(
        condition_dim=16, dim=16, heads=4, condition_cross_depth=1,
        noise_token_count=4, noise_token_dim=8, refine_depth=1,
        cvt_grid_sizes=(4, 2, 1), max_resolution=16,
        zero_mean_output=True, gradient_checkpointing=False,
    )


def test_relative_rollout_matches_explicit_image_feedback_and_preserves_gradient():
    cloud = small_cloud()
    rollout = RelativeImageRolloutCloud(cloud)
    initial = torch.randn(2, 1, 8, 8, requires_grad=True)
    condition = torch.randn(2, 16, requires_grad=True)
    noise = torch.randn(2, 4, 5, 4, 8)
    first_outputs = []

    def capture(_module, _inputs, outputs):
        if not first_outputs:
            outputs[0].retain_grad()
            first_outputs.append(outputs[0])

    handle = cloud.register_forward_hook(capture)
    actual = rollout(initial, condition, steps=4, noise=noise)
    handle.remove()
    expected_noise = noise.clone()
    expected_noise[:, :, 0].zero_()
    current, _, _ = cloud(initial, condition, samples=5, noise=expected_noise[:, 0])
    expected = [current]
    for step in range(1, 4):
        current, _, _ = cloud(current.flatten(0, 1), condition.repeat_interleave(5, 0),
                             samples=1, noise=expected_noise[:, step].reshape(10, 1, 4, 8))
        current = current[:, 0].reshape(2, 5, 1, 8, 8)
        expected.append(current)
    torch.testing.assert_close(actual, torch.stack(expected, dim=1))
    actual[:, -1].square().mean().backward()
    assert first_outputs[0].grad.abs().sum() > 0
    assert initial.grad.abs().sum() > 0
    assert condition.grad.abs().sum() > 0
    assert actual.shape == (2, 4, 5, 1, 8, 8)
    torch.testing.assert_close(actual.mean(dim=(-2, -1)), torch.zeros(2, 4, 5, 1), atol=1e-6, rtol=0)


def test_relative_zero_path_is_deterministic_and_zero_correction_is_identity():
    cloud = small_cloud()
    rollout = RelativeImageRolloutCloud(cloud)
    initial = torch.randn(1, 1, 8, 8)
    initial -= initial.mean(dim=(-2, -1), keepdim=True)
    condition = torch.randn(1, 16)
    first = rollout(initial, condition, steps=4)
    second = rollout(initial, condition, steps=4)
    torch.testing.assert_close(first[:, :, 0], second[:, :, 0])
    torch.nn.init.zeros_(cloud.output_head.weight)
    torch.nn.init.zeros_(cloud.output_head.bias)
    fields = rollout(initial, condition, steps=20)
    torch.testing.assert_close(fields, initial[:, None, None].expand(1, 20, 5, 1, 8, 8))


def test_conditional_gaussian_tokens_use_ms_query_32_image_kv_and_mean_zero_path():
    from molai.models.attention import TokenCrossBlock2d

    cloud = TokenConditionRelativeCloud(
        condition_dim=64, dim=16, heads=4, condition_cross_depth=1,
        noise_token_count=4, noise_token_dim=8, refine_depth=1,
        cvt_grid_sizes=(8, 4, 2), max_resolution=32,
        zero_mean_output=True, gradient_checkpointing=False,
    )
    del cloud.condition_plane
    cloud.condition_blocks = torch.nn.ModuleList([TokenCrossBlock2d(16, 64, 4)])
    cloud.noise_attention = ScalarGatedRandomMemoryAttention(16, 4, 8, 0.8, 0.02)
    audit = configure_conditional_gaussian_tokens(cloud, {
        "conditional_gaussian_tokens": True,
        "gaussian_controller_heads": 4,
        "gaussian_image_grid_size": 32,
        "gaussian_sigma_min": 0.05,
        "gaussian_sigma_max": 2.0,
        "gaussian_sigma_init": 1.0,
    })
    assert audit["random_attention_kv_norm"] == "removed"
    assert not hasattr(cloud.noise_attention, "key_norm")
    initial = torch.randn(2, 1, 32, 32, requires_grad=True)
    ms_tokens = torch.randn(2, 7, 64, requires_grad=True)
    ms_mask = torch.tensor([[True] * 4 + [False] * 3, [True] * 7])
    epsilon = torch.randn(2, 3, 4, 8)
    epsilon[:, 0].zero_()
    captured = {}

    def capture_cross(_module, inputs):
        captured["image_token_count"] = inputs[1].shape[1]

    def capture_random(_module, inputs):
        captured["random_tokens"] = inputs[1].detach().reshape(2, 3, 4, 8)

    cross_handle = cloud.random_token_controller.cross_attention.register_forward_pre_hook(capture_cross)
    random_handle = cloud.noise_attention.register_forward_pre_hook(capture_random)
    image_hidden = cloud.encode_image(initial)
    mean, sigma = cloud.random_token_controller(ms_tokens, ms_mask, image_hidden)
    fields, _, _ = cloud(
        initial, ms_tokens, samples=3, noise=epsilon, condition_mask=ms_mask,
    )
    cross_handle.remove()
    random_handle.remove()

    assert captured["image_token_count"] == 32 * 32
    expected = mean[:, None, None] + sigma[:, None, None] * epsilon
    torch.testing.assert_close(captured["random_tokens"], expected.detach())
    torch.testing.assert_close(
        captured["random_tokens"][:, 0],
        mean[:, None].expand(-1, cloud.noise_token_count, -1).detach(),
    )
    assert float(sigma.detach().min()) >= 0.05 and float(sigma.detach().max()) <= 2.0
    fields.square().mean().backward()
    assert cloud.random_token_controller.mean_head.weight.grad.abs().sum() > 0
    assert ms_tokens.grad is not None and ms_tokens.grad.abs().sum() > 0
    assert initial.grad is not None and initial.grad.abs().sum() > 0

    # After the first recurrent step there are multiple image paths per cached
    # spectrum.  Their moments must remain path-specific while sharing MS.
    cloud.cache_condition_kv = True
    rollout = RelativeImageRolloutCloud(cloud)
    with torch.no_grad():
        recurrent = rollout(
            initial.detach(), ms_tokens.detach(), steps=2, random_samples=2,
            condition_mask=ms_mask,
        )
    assert recurrent.shape == (2, 2, 3, 1, 32, 32)


def test_relative_targets_are_original_input_marginals(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    trainer = importlib.import_module("train_relative_image_rollout")
    schedule = CosineVPSchedule(levels=64, noise_scale=3.5, zero_mean_noise=True)
    clean = torch.randn(3, 1, 8, 8)
    clean -= clean.mean(dim=(-2, -1), keepdim=True)
    start = torch.tensor([0, 8, 64])
    config = {"cloud_matching": {"transition_jumps": [
        {"max_level": 8, "jump": 1}, {"max_level": 16, "jump": 2},
        {"max_level": 32, "jump": 4}, {"max_level": 64, "jump": 8},
    ]}}
    initial, targets, levels = trainer.trajectory_targets(schedule, clean, start, 4, 4, config)
    assert levels.tolist() == [[0, 0, 0, 0], [7, 6, 5, 4], [56, 48, 40, 32]]
    torch.testing.assert_close(initial[0], clean[0])
    torch.testing.assert_close(targets[0], clean[0][None, None].expand(4, 4, 1, 8, 8))
    targets_before = targets.clone()
    prediction = torch.randn(3, 4, 5, 1, 8, 8, requires_grad=True)
    losses = trainer.rollout_losses(prediction, targets, initial, clean,
                                    trainer.FullBandEnergyDistance(include_target_constant=False), 0.05)
    losses["total"].backward()
    assert prediction.grad.abs().sum() > 0
    torch.testing.assert_close(targets, targets_before)


def test_restore_scalar_gate_checkpoint_without_architecture_conversion():
    cloud = small_cloud()
    cloud.noise_attention = ScalarGatedRandomMemoryAttention(16, 4, 8, 0.8, 0.297)
    from molai.models.condition import SpectrumConditionEncoder

    spectrum = {"heads": 4, "peak_conv_stages": 2, "spectrum_layers": 1,
                "peak_position_dim": 8, "mz_bin_width": 1.0, "mz_upper_bound": 1000.0}
    encoder = SpectrumConditionEncoder(metadata_dim=3, dim=16, peak_chunk_size=8, **spectrum)
    config = {"model": {
        "field_channels": 1, "condition_dim": 16, "fullres_dim": 16, "heads": 4,
        "condition_cross_depth": 1, "noise_token_count": 4, "noise_token_dim": 8,
        "refine_depth": 1, "cvt_output_sizes": [4, 2, 1], "max_resolution": 16,
        "zero_mean_output": True, "noise_temperature": 0.8, "noise_gate_init": 0.02,
    }, "training": {"peak_chunk_size": 8}, "spectrum_encoder": spectrum}
    restored, restored_encoder = build_relative_models(
        {"cloud": cloud.state_dict(), "condition_encoder": encoder.state_dict(), "config": config}, 3,
    )
    assert isinstance(restored.noise_attention, ScalarGatedRandomMemoryAttention)
    assert restored.field_encoder.attention.grid_sizes == (4, 2, 1)
    for key, value in cloud.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(restored_encoder.state_dict()[key], value)
    x, condition, noise = torch.randn(1, 1, 8, 8), torch.randn(1, 16), torch.randn(1, 4, 4, 8)
    torch.testing.assert_close(cloud(x, condition, noise=noise)[0], restored(x, condition, noise=noise)[0])


def test_token_condition_replacement_preserves_backbone_and_masks_padding():
    from molai.models.attention import TokenCrossBlock2d
    from molai.models.condition import SpectrumConditionEncoder

    cloud = small_cloud()
    cloud.noise_attention = ScalarGatedRandomMemoryAttention(16, 4, 8, 0.8, 0.297)
    spectrum = {"heads": 4, "peak_conv_stages": 2, "spectrum_layers": 1,
                "peak_position_dim": 8, "mz_bin_width": 1.0, "mz_upper_bound": 100.0,
                "dropout": 0.0}
    old_encoder = SpectrumConditionEncoder(metadata_dim=3, dim=16, peak_chunk_size=8, **spectrum)
    config = {"model": {
        "field_channels": 1, "condition_dim": 16, "fullres_dim": 16, "heads": 4,
        "condition_cross_depth": 1, "noise_token_count": 4, "noise_token_dim": 8,
        "refine_depth": 1, "cvt_output_sizes": [4, 2, 1], "max_resolution": 16,
        "zero_mean_output": True, "noise_temperature": 0.8, "noise_gate_init": 0.02,
        "ffn_ratio": 2.0, "condition_gate_init": 0.5,
    }, "training": {"peak_chunk_size": 8}, "spectrum_encoder": spectrum}
    student, encoder, audit = build_token_relative_models(
        {"cloud": cloud.state_dict(), "condition_encoder": old_encoder.state_dict(), "config": config},
        3, gradient_checkpointing=True,
    )
    assert not hasattr(student, "condition_plane")
    assert isinstance(student.condition_blocks[0], TokenCrossBlock2d)
    assert len(encoder.spectrum_attention_blocks) == 2
    assert len(encoder.peak_conv_blocks) == 2
    assert encoder.output[-1].out_features == 64
    assert "peak_embedding.slope.weight" in audit["reused_ms_keys"]
    for key, value in cloud.state_dict().items():
        if not key.startswith(("condition_plane.", "condition_blocks.")):
            torch.testing.assert_close(student.state_dict()[key], value)
    peaks = torch.rand(2, 8, 2)
    peaks[..., 0] = torch.arange(8)
    mask = torch.tensor([[True] * 4 + [False] * 4, [True] * 8])
    tokens, token_mask = encoder.forward_ragged_tokens(
        peaks, mask, torch.arange(2), torch.arange(2), torch.randn(2, 3),
        torch.tensor([10., 20.]), 2,
    )
    assert tokens.shape == (2, 2, 64)
    assert token_mask.tolist() == [[True, False], [True, True]]
    initial, noise = torch.randn(2, 1, 8, 8), torch.randn(2, 4, 5, 4, 8)
    rollout = RelativeImageRolloutCloud(student)
    fields = rollout(initial, tokens, condition_mask=token_mask, noise=noise)
    poisoned = tokens.masked_fill(~token_mask[..., None], 1e6)
    same = rollout(initial, poisoned, condition_mask=token_mask, noise=noise)
    torch.testing.assert_close(fields, same)
    fields[:, -1].square().mean().backward()
    assert encoder.peak_embedding.output.weight.grad.abs().sum() > 0
    assert student.condition_blocks[0].attention.in_proj_weight is None
    assert student.condition_blocks[0].attention.k_proj_weight.grad.abs().sum() > 0
    assert student.field_encoder.input_projection.weight.grad.abs().sum() > 0
    torch.testing.assert_close(fields.mean(dim=(-2, -1)), torch.zeros(2, 4, 5, 1), atol=1e-6, rtol=0)


def test_cached_ms_kv_matches_outputs_and_all_gradients():
    from molai.models.attention import TokenCrossBlock2d
    from molai.models.relative_rollout import TokenConditionRelativeCloud

    cloud = TokenConditionRelativeCloud(
        condition_dim=16, dim=16, heads=4, condition_cross_depth=1,
        noise_token_count=4, noise_token_dim=8, refine_depth=1,
        cvt_grid_sizes=(4, 2, 1), max_resolution=16,
        zero_mean_output=True, gradient_checkpointing=True,
    )
    del cloud.condition_plane
    cloud.condition_blocks = torch.nn.ModuleList([TokenCrossBlock2d(16, 64, 4)])
    cloud.readout.requires_grad_(False)
    rollout = RelativeImageRolloutCloud(cloud)
    initial = torch.randn(2, 1, 8, 8, requires_grad=True)
    tokens = torch.randn(2, 7, 64, requires_grad=True)
    mask = torch.tensor([[True] * 3 + [False] * 4, [True] * 7])
    noise = torch.randn(2, 4, 5, 4, 8)
    variables = [initial, tokens, *[p for p in cloud.parameters() if p.requires_grad]]
    original = rollout(initial, tokens, condition_mask=mask, noise=noise)
    original_gradients = torch.autograd.grad(original.square().mean(), variables)
    cloud.cache_condition_kv = True
    cached = rollout(initial, tokens, condition_mask=mask, noise=noise)
    cached_gradients = torch.autograd.grad(cached.square().mean(), variables)
    torch.testing.assert_close(cached, original, atol=1e-6, rtol=1e-5)
    for cached_gradient, original_gradient in zip(cached_gradients, original_gradients, strict=True):
        torch.testing.assert_close(cached_gradient, original_gradient, atol=1e-6, rtol=1e-4)
    assert torch.count_nonzero(cached_gradients[1][~mask]) == 0
    cloud.cache_condition_kv = False
    cloud.uncheckpointed_refine_blocks = 1
    selective = rollout(initial, tokens, condition_mask=mask, noise=noise)
    selective_gradients = torch.autograd.grad(selective.square().mean(), variables)
    torch.testing.assert_close(selective, original, atol=0, rtol=0)
    for selective_gradient, original_gradient in zip(selective_gradients, original_gradients, strict=True):
        torch.testing.assert_close(selective_gradient, original_gradient, atol=1e-6, rtol=1e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.backends.cuda.is_flash_attention_available(),
    reason="packed Flash Attention requires a CUDA build with Flash kernels",
)
def test_packed_ms_attention_cuda_outputs_gradients_and_padding():
    from molai.models.attention import TokenCrossBlock2d

    block = TokenCrossBlock2d(64, 64, 4).cuda()
    query = torch.randn(6, 4, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    context = torch.randn(2, 11, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.tensor([[True] * 7 + [False] * 4, [True] * 11], device="cuda")
    variables = [query, context, *block.parameters()]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        original = block(query, context.repeat_interleave(3, 0), mask.repeat_interleave(3, 0))
    original_gradients = torch.autograd.grad(original.float().square().mean(), variables)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        key, value = block.project_context(context)
        counts = mask.sum(dim=1, dtype=torch.int32)
        cumulative = torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))
        packed = block.forward_varlen(query, key.transpose(1, 2)[mask],
                                     value.transpose(1, 2)[mask], cumulative, 11)
    packed_gradients = torch.autograd.grad(packed.float().square().mean(), variables)
    torch.testing.assert_close(packed, original, atol=0.02, rtol=0.02)
    original_vector = torch.cat([value.float().flatten() for value in original_gradients])
    packed_vector = torch.cat([value.float().flatten() for value in packed_gradients])
    assert float((packed_vector - original_vector).norm() / original_vector.norm()) < 0.03
    assert torch.count_nonzero(packed_gradients[1][~mask]) == 0
