import pytest
import torch

from molai.models.attention import MultiscaleCvTAttention2d, RandomMemoryAttention
from molai.models.bridge import CosineVPSchedule, GeometricVESchedule, VPSchedule
from molai.models.cloud import (
    AbsoluteHiddenRolloutCloud,
    AbsoluteMolecularFieldCloud,
    AbsoluteTrajectoryRolloutCloud,
    EncoderAnchoredHiddenUpdate,
    HiddenRolloutCloud,
    MolecularCloudModel,
    MolecularFieldCloud,
)
from molai.models.condition import (
    ExponentialDistanceConvBlock,
    PositionwiseAffinePeakEmbedding,
    SpectrumConditionEncoder,
    SpectrumTokenEncoder,
)
from molai.models.image_smiles import FieldToSmiles
from molai.models.losses import (
    FullBandEnergyDistance,
    _off_diagonal_mean,
    full_band_distance,
)
from molai.models.smiles import SmilesTokenizer


def test_full_resolution_cloud_forward_and_backward() -> None:
    model = MolecularCloudModel(
        smiles_vocab_size=16,
        smiles_pad_token_id=0,
        condition_dim=32,
        field_channels=1,
        dim=32,
        heads=4,
        condition_cross_depth=1,
        noise_cross_depth=1,
        refine_depth=4,
        window_size=4,
        max_resolution=32,
        gradient_checkpointing=False,
    )
    current = torch.randn(2, 1, 16, 16).clamp(-1, 1)
    condition = torch.randn(2, 32)
    decoder_input = torch.randint(0, 16, (2, 12))
    output = model(current, condition, samples=2, smiles_input_ids=decoder_input)

    assert output.fields.shape == (2, 2, 1, 16, 16)
    assert output.spatial_noise_energy.shape == (2, 16, 16)
    assert output.molecular_embeddings.shape == (2, 2, 32)
    assert output.smiles_logits is not None
    assert output.smiles_logits.shape == (2, 2, 12, 16)
    output.fields.mean().backward()


def test_cvt_attention_keeps_queries_and_compresses_only_context() -> None:
    attention = MultiscaleCvTAttention2d(
        dim=32,
        heads=4,
        kernel_sizes=(3, 5, 7),
        grid_sizes=(8, 4, 2),
    )
    query = torch.randn(2, 16, 16, 32, requires_grad=True)
    context = torch.randn(2, 16, 16, 32)
    pooled = attention.pool_context(context)
    result = attention(query, context)

    assert pooled.shape == (2, 84, 32)
    assert result.shape == query.shape
    result.square().mean().backward()
    assert query.grad is not None


def test_bridge_and_energy_distance() -> None:
    clean = torch.randn(2, 1, 16, 16).tanh()
    bridge = VPSchedule(steps=20).sample_training_batch(clean, samples=3)
    legacy_loss = FullBandEnergyDistance(levels=2, unbiased=False)(
        bridge.target_cloud, bridge.target_cloud, bridge.current
    )
    unbiased_loss = FullBandEnergyDistance(levels=2)(
        bridge.target_cloud, bridge.target_cloud, bridge.current
    )

    assert bridge.current.shape == clean.shape
    assert bridge.target_cloud.shape == (2, 3, 1, 16, 16)
    assert abs(float(legacy_loss)) < 1e-5
    assert torch.isfinite(unbiased_loss)


def test_off_diagonal_mean_excludes_self_pairs() -> None:
    distances = torch.tensor(
        [
            [
                [0.001, 2.0, 4.0],
                [2.0, 0.001, 6.0],
                [4.0, 6.0, 0.001],
            ]
        ]
    )

    torch.testing.assert_close(_off_diagonal_mean(distances), torch.tensor(4.0))


def test_unbiased_energy_distance_requires_two_samples() -> None:
    cloud = torch.zeros(2, 1, 1, 8, 8)
    current = torch.zeros(2, 1, 8, 8)

    with pytest.raises(ValueError, match="at least two samples"):
        FullBandEnergyDistance(levels=2)(cloud, cloud, current)


def test_geometric_noise_bridge_uses_requested_levels() -> None:
    clean = torch.zeros(3, 1, 8, 8)
    levels = torch.tensor([1, 16, 64])
    schedule = GeometricVESchedule(levels=64, sigma_min=0.005, sigma_max=0.5)

    bridge = schedule.sample_training_batch(
        clean,
        samples=2,
        current_levels=levels,
        clean_answer_probability=1.0,
    )

    assert bridge.current.shape == clean.shape
    assert bridge.target_cloud.shape == (3, 2, 1, 8, 8)
    torch.testing.assert_close(bridge.current_levels, levels)
    torch.testing.assert_close(bridge.target_cloud, clean[:, None].expand(-1, 2, -1, -1, -1))
    assert bridge.current[-1].std() > bridge.current[0].std() * 20


def test_cosine_vp_terminal_level_contains_no_clean_signal() -> None:
    schedule = CosineVPSchedule(levels=64, noise_scale=3.5)
    clean = torch.randn(2, 1, 32, 32) * 4.0
    terminal = torch.full((2,), 64)

    torch.manual_seed(11)
    first = schedule.sample_training_batch(
        clean,
        samples=2,
        current_levels=terminal,
        answer_jump=8,
        clean_answer_probability=0.0,
    )
    torch.manual_seed(11)
    second = schedule.sample_training_batch(
        clean * 3.0,
        samples=2,
        current_levels=terminal,
        answer_jump=8,
        clean_answer_probability=0.0,
    )

    assert float(schedule.alpha_bar[0]) == 1.0
    assert float(schedule.alpha_bar[-1]) == 0.0
    torch.testing.assert_close(first.current, second.current)
    torch.testing.assert_close(first.current.std(), torch.tensor(3.5), atol=0.12, rtol=0.0)
    torch.testing.assert_close(first.answer_levels, torch.full((2,), 56))


def test_cosine_vp_centers_noise_and_supports_variable_jumps_and_fixed_points() -> None:
    schedule = CosineVPSchedule(levels=64, noise_scale=3.5, zero_mean_noise=True)
    clean = torch.randn(5, 1, 16, 16)
    clean = clean - clean.mean(dim=(-2, -1), keepdim=True)
    levels = torch.tensor([0, 4, 12, 24, 64])
    jumps = torch.tensor([0, 1, 2, 4, 8])

    bridge = schedule.sample_training_batch(
        clean,
        samples=3,
        current_levels=levels,
        answer_jump=jumps,
    )

    torch.testing.assert_close(bridge.answer_levels, torch.tensor([0, 3, 10, 20, 56]))
    torch.testing.assert_close(bridge.current[0], clean[0])
    torch.testing.assert_close(
        bridge.target_cloud[0], clean[0].expand_as(bridge.target_cloud[0])
    )
    torch.testing.assert_close(
        bridge.current.mean(dim=(-2, -1)),
        torch.zeros(5, 1),
        atol=2e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        bridge.target_cloud.mean(dim=(-2, -1)),
        torch.zeros(5, 3, 1),
        atol=2e-6,
        rtol=0.0,
    )


def test_cloud_raw_residual_is_not_limited_to_legacy_range() -> None:
    model = MolecularFieldCloud(
        condition_dim=32,
        dim=32,
        heads=4,
        condition_cross_depth=1,
        noise_cross_depth=1,
        noise_token_count=8,
        refine_depth=1,
        max_resolution=16,
        gradient_checkpointing=False,
    ).eval()
    torch.nn.init.zeros_(model.output_head.weight)
    torch.nn.init.constant_(model.output_head.bias, 12.0)
    fields, _, _ = model(
        torch.zeros(1, 1, 8, 8),
        torch.zeros(1, 32),
        samples=1,
    )

    torch.testing.assert_close(fields, torch.full_like(fields, 12.0))


def test_cloud_can_project_every_output_to_zero_spatial_mean() -> None:
    model = MolecularFieldCloud(
        condition_dim=32,
        dim=32,
        heads=4,
        condition_cross_depth=1,
        noise_cross_depth=1,
        noise_token_count=8,
        refine_depth=1,
        max_resolution=16,
        zero_mean_output=True,
        gradient_checkpointing=False,
    ).eval()
    current = torch.randn(2, 1, 8, 8)
    fields, _, _ = model(current, torch.randn(2, 32), samples=3)

    torch.testing.assert_close(
        fields.float().mean(dim=(-2, -1)),
        torch.zeros(2, 3, 1),
        atol=2e-6,
        rtol=0.0,
    )


def test_reused_band_pyramid_matches_direct_error_filtering() -> None:
    torch.manual_seed(4)
    first = torch.randn(3, 1, 16, 16)
    second = torch.randn(3, 1, 16, 16)
    error = first - second
    direct_distances = []
    kernel_1d = first.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    kernel = torch.outer(kernel_1d, kernel_1d)[None, None]
    for level in range(3):
        low = torch.nn.functional.conv2d(error, kernel, padding=2 * 2**level, dilation=2**level)
        direct_distances.append(torch.sqrt((error - low).square() + 1e-6).mean(dim=(1, 2, 3)))
        error = low
    direct_distances.append(torch.sqrt(error.square() + 1e-6).mean(dim=(1, 2, 3)))
    expected = torch.stack(direct_distances).mean(dim=0)

    torch.testing.assert_close(full_band_distance(first, second, levels=3), expected)


def test_noise_token_cloud_is_deterministic_and_sample_independent() -> None:
    model = MolecularFieldCloud(
        condition_dim=32,
        field_channels=1,
        dim=32,
        heads=4,
        condition_cross_depth=1,
        noise_cross_depth=1,
        noise_token_count=8,
        refine_depth=1,
        window_size=4,
        max_resolution=16,
        gradient_checkpointing=False,
    ).eval()
    current = torch.zeros(1, 1, 8, 8)
    condition = torch.randn(1, 32)
    noise = torch.randn(1, 2, 8, 32)

    first, _, _ = model(current, condition, samples=2, noise=noise)
    second, _, _ = model(current, condition, samples=2, noise=noise)

    torch.testing.assert_close(first, second)
    assert not torch.allclose(first[:, 0], first[:, 1])


def test_cloud_uses_one_random_attention_before_cvt_refinement() -> None:
    model = MolecularFieldCloud(
        condition_dim=32,
        dim=32,
        heads=4,
        condition_cross_depth=1,
        noise_cross_depth=1,
        noise_token_count=8,
        refine_depth=2,
        max_resolution=16,
        gradient_checkpointing=False,
    )

    assert model.noise_attention.random_dim == 32
    assert len(model.refine_blocks) == 2
    assert all(block.attention.pooled_token_count(16, 16) == 84 for block in model.refine_blocks)


def test_random_attention_uses_channelwise_glu_gate() -> None:
    attention = RandomMemoryAttention(
        dim=8,
        heads=2,
        random_dim=4,
        gate_init=0.02,
    )
    query = torch.randn(2, 4, 4, 8, requires_grad=True)
    random_tokens = torch.randn(2, 6, 4)
    output = attention(query, random_tokens, torch.ones(2, 4, 4))

    torch.testing.assert_close(
        torch.sigmoid(attention.gate_projection.bias),
        torch.full((8,), 0.02),
    )
    output.square().mean().backward()
    assert attention.gate_projection.weight.grad is not None


def test_hidden_rollout_encodes_once_and_backpropagates_through_shared_steps() -> None:
    cloud = MolecularFieldCloud(
        condition_dim=32,
        dim=32,
        heads=4,
        condition_cross_depth=1,
        noise_cross_depth=1,
        noise_token_count=8,
        refine_depth=1,
        max_resolution=16,
        zero_mean_output=True,
        gradient_checkpointing=False,
    )
    model = HiddenRolloutCloud(
        cloud,
        max_level=8,
        gate_init=0.02,
        decoder_dim=16,
    )
    assert model.state_update.attention.grid_sizes == (32, 16, 8)
    assert model.state_update.attention.pooled_token_count(128, 128) == 1344
    encoder_calls = 0

    def count_encoder_calls(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    handle = cloud.field_encoder.register_forward_pre_hook(count_encoder_calls)
    initial = torch.randn(2, 1, 8, 8)
    condition = torch.randn(2, 32)
    levels = torch.tensor([8, 4, 0])
    noise = torch.randn(2, 2, 3, 8, 32)
    output = model(initial, condition, levels, samples=2, noise=noise)
    handle.remove()

    assert encoder_calls == 1
    assert output.fields.shape == (2, 2, 3, 1, 8, 8)
    assert output.anchor_reconstruction.shape == initial.shape
    assert output.final_hidden.shape == (2, 2, 8, 8, 32)
    assert output.gate_means.shape == (2, 2, 3)
    assert output.update_rms.shape == (2, 2, 3)
    assert output.spatial_noise_energy.shape == (2, 2, 3, 8, 8)
    torch.testing.assert_close(
        output.fields.float().mean(dim=(-2, -1)),
        torch.zeros(2, 2, 3, 1),
        atol=2e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output.gate_means,
        torch.full_like(output.gate_means, 0.02),
        atol=1e-6,
        rtol=0.0,
    )

    output.fields.square().mean().backward()
    assert model.state_update.gate_bias.grad is not None
    assert model.state_update.attention.attention.in_proj_weight.grad is not None
    assert any(parameter.grad is not None for parameter in cloud.field_encoder.parameters())
    assert cloud.condition_blocks[0].gate_projection.weight.grad is not None
    assert model.output_head.output_projection.weight.grad is not None


def test_absolute_decoder_accepts_encoder_anchor_directly() -> None:
    cloud = MolecularFieldCloud(
        field_channels=1,
        condition_dim=16,
        dim=16,
        heads=4,
        condition_cross_depth=1,
        noise_cross_depth=1,
        noise_token_count=4,
        refine_depth=1,
        max_resolution=8,
        zero_mean_output=True,
        gradient_checkpointing=False,
    )
    model = HiddenRolloutCloud(cloud, max_level=4, decoder_dim=8)
    field = torch.randn(2, 1, 8, 8)

    anchor = model.encode_anchor(field)
    reconstruction = model.decode_absolute(anchor)

    assert reconstruction.shape == field.shape
    torch.testing.assert_close(
        reconstruction.mean(dim=(-2, -1)),
        torch.zeros(2, 1),
        atol=1e-6,
        rtol=0.0,
    )
    reconstruction.square().mean().backward()
    assert model.output_head.direct_projection.weight.grad is not None


def test_hidden_rollout_rejects_mismatched_noise_shape() -> None:
    cloud = MolecularFieldCloud(
        condition_dim=16,
        dim=16,
        heads=4,
        condition_cross_depth=1,
        noise_cross_depth=1,
        noise_token_count=4,
        refine_depth=1,
        max_resolution=8,
        gradient_checkpointing=False,
    )
    model = HiddenRolloutCloud(cloud, max_level=4)

    with pytest.raises(ValueError, match=r"\[B,M,T,K,R\]"):
        model(
            torch.randn(1, 1, 8, 8),
            torch.randn(1, 16),
            torch.tensor([4, 0]),
            samples=2,
            noise=torch.randn(1, 2, 4, 16),
        )


def test_hidden_state_anchor_uses_encoder_as_query_and_model_as_context() -> None:
    update = EncoderAnchoredHiddenUpdate(
        dim=16,
        heads=4,
        max_level=4,
        gate_init=0.02,
        kernel_sizes=(3,),
        grid_sizes=(2,),
    )
    encoder_hidden = torch.randn(2, 8, 8, 16)
    model_hidden = torch.randn(2, 8, 8, 16)
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture_attention_inputs(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        captured.append((inputs[0].detach(), inputs[1].detach()))

    handle = update.attention.register_forward_pre_hook(capture_attention_inputs)
    anchored, _, _ = update(encoder_hidden, model_hidden, torch.tensor([4, 0]))
    handle.remove()

    torch.testing.assert_close(captured[0][0], update.query_norm(encoder_hidden))
    torch.testing.assert_close(captured[0][1], update.context_norm(model_hidden))
    assert anchored.shape == encoder_hidden.shape

    torch.nn.init.zeros_(update.candidate_projection.weight)
    anchored, _, _ = update(encoder_hidden, model_hidden, torch.tensor([4, 0]))
    torch.testing.assert_close(anchored, encoder_hidden)


def test_smiles_tokenizer_round_trip() -> None:
    tokenizer = SmilesTokenizer.from_smiles(["CC(=O)O", "c1ccccc1Cl"])
    value = "c1ccccc1Cl"
    assert tokenizer.decode(tokenizer.encode(value)) == value


def test_field_to_smiles_probe_shapes() -> None:
    model = FieldToSmiles(vocab_size=20, embedding_dim=32, decoder_hidden_dim=32)
    fields = torch.randn(2, 1, 64, 64)
    input_ids = torch.randint(0, 20, (2, 11))
    assert model(fields, input_ids).shape == (2, 11, 20)
    assert model.generate(fields, bos_token_id=1, eos_token_id=2, max_length=7).shape[0] == 2


def _spectrum_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(7)
    peaks = torch.rand(2, 3, 6, 2)
    peaks[..., 0] = 50.0 + 600.0 * peaks[..., 0]
    peak_mask = torch.tensor(
        [
            [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]],
        ],
        dtype=torch.bool,
    )
    spectrum_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    metadata = torch.randn(2, 3, 5)
    precursor_mz = torch.tensor([[700.0, 500.0, 0.0], [800.0, 450.0, 650.0]])
    return peaks, peak_mask, spectrum_mask, metadata, precursor_mz


def test_spectrum_condition_attention_shapes_and_backward() -> None:
    encoder = SpectrumConditionEncoder(
        metadata_dim=5,
        dim=32,
        heads=4,
        peak_conv_stages=2,
        spectrum_layers=1,
        dropout=0.0,
        peak_position_dim=8,
        mz_bin_width=1.0,
        mz_upper_bound=1_000.0,
    )
    inputs = _spectrum_inputs()
    condition = encoder(*inputs)

    assert condition.shape == (2, 32)
    assert torch.isfinite(condition).all()
    condition.square().mean().backward()
    assert encoder.peak_embedding.slope.weight.grad is not None
    assert encoder.peak_conv_blocks[0].raw_tau.grad is not None


def test_spectrum_condition_attention_supports_bfloat16_autocast() -> None:
    encoder = SpectrumConditionEncoder(
        metadata_dim=5,
        dim=32,
        heads=4,
        peak_conv_stages=1,
        spectrum_layers=1,
        dropout=0.0,
        peak_position_dim=8,
        mz_bin_width=1.0,
        mz_upper_bound=1_000.0,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        condition = encoder(*_spectrum_inputs())

    assert condition.shape == (2, 32)
    assert torch.isfinite(condition).all()


def test_spectrum_token_encoder_keeps_every_twice_pooled_peak() -> None:
    encoder = SpectrumTokenEncoder(
        metadata_dim=3,
        dim=16,
        heads=4,
        peak_conv_stages=2,
        spectrum_layers=1,
        dropout=0.0,
        peak_position_dim=8,
        mz_bin_width=1.0,
        mz_upper_bound=1_000.0,
        peak_chunk_size=8,
    )
    peak_chunks = torch.rand(3, 8, 2)
    peak_chunks[..., 0] = peak_chunks[..., 0] * 800.0
    peak_mask = torch.ones(3, 8, dtype=torch.bool)
    chunk_to_spectrum = torch.tensor([0, 1, 2])
    spectrum_to_molecule = torch.tensor([0, 1, 1])
    metadata = torch.randn(3, 3)
    precursor_mz = torch.tensor([500.0, 700.0, 650.0])

    tokens, mask = encoder.forward_ragged_tokens(
        peak_chunks,
        peak_mask,
        chunk_to_spectrum,
        spectrum_to_molecule,
        metadata,
        precursor_mz,
        molecule_count=2,
    )

    # Every eight-peak chunk leaves two tokens after two 2x pools. Molecule 1
    # has two spectra/chunks and therefore retains all four tokens.
    assert tokens.shape == (2, 4, 16)
    assert mask.tolist() == [[True, True, False, False], [True, True, True, True]]
    tokens[mask].square().mean().backward()
    assert encoder.peak_conv_blocks[0].raw_tau.grad is not None


def test_absolute_cloud_uses_masked_ms_tokens_and_backpropagates() -> None:
    model = AbsoluteMolecularFieldCloud(
        field_channels=1,
        condition_dim=16,
        dim=16,
        heads=4,
        condition_cross_depth=1,
        noise_token_count=4,
        noise_token_dim=8,
        refine_depth=1,
        cvt_kernel_sizes=(3, 3, 3),
        cvt_grid_sizes=(4, 2, 1),
        max_level=8,
        decoder_dim=16,
        gradient_checkpointing=False,
    )
    current = torch.randn(2, 1, 8, 8)
    condition = torch.randn(2, 5, 16, requires_grad=True)
    condition_mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    output = model(
        current,
        condition,
        torch.tensor([4, 8]),
        samples=2,
        condition_mask=condition_mask,
        return_anchor_reconstruction=True,
    )

    assert output.fields.shape == (2, 2, 1, 8, 8)
    assert output.anchor_reconstruction is not None
    assert output.anchor_reconstruction.shape == current.shape
    torch.testing.assert_close(
        output.fields.mean(dim=(-2, -1)),
        torch.zeros(2, 2, 1),
        atol=2e-6,
        rtol=0.0,
    )
    (output.fields.square().mean() + output.anchor_reconstruction.square().mean()).backward()
    assert condition.grad is not None


def test_absolute_hidden_rollout_branches_only_at_final_step() -> None:
    cloud = AbsoluteMolecularFieldCloud(
        field_channels=1,
        condition_dim=16,
        dim=16,
        heads=4,
        condition_cross_depth=1,
        noise_token_count=4,
        noise_token_dim=8,
        refine_depth=1,
        cvt_kernel_sizes=(3, 3, 3),
        cvt_grid_sizes=(4, 2, 1),
        max_level=8,
        decoder_dim=16,
        gradient_checkpointing=False,
    )
    model = AbsoluteHiddenRolloutCloud(cloud)
    initial = torch.randn(2, 1, 8, 8)
    condition = torch.randn(2, 5, 16, requires_grad=True)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    levels = torch.tensor([8, 6, 4, 2])
    intermediate_noise = torch.randn(2, 3, 4, 8)
    final_noise = torch.randn(2, 5, 4, 8)
    final_noise[:, 0].zero_()
    encoder_calls = 0

    def count_encoder_calls(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    handle = cloud.field_encoder.register_forward_pre_hook(count_encoder_calls)
    output = model(
        initial,
        condition,
        levels,
        final_samples=5,
        intermediate_noise=intermediate_noise,
        final_noise=final_noise,
        condition_mask=mask,
    )
    handle.remove()

    assert encoder_calls == 1
    assert output.fields.shape == (2, 5, 1, 8, 8)
    assert output.final_hidden.shape == (2, 5, 8, 8, 16)
    assert output.intermediate_gate_means.shape == (2, 3)
    assert output.intermediate_update_rms.shape == (2, 3)
    assert output.final_gate_means.shape == (2, 5)
    assert output.final_update_rms.shape == (2, 5)
    assert output.spatial_noise_energy.shape == (2, 4, 8, 8)
    torch.testing.assert_close(
        output.fields.mean(dim=(-2, -1)),
        torch.zeros(2, 5, 1),
        atol=2e-6,
        rtol=0.0,
    )

    output.fields.square().mean().backward()
    assert condition.grad is not None
    assert cloud.field_encoder.input_projection.weight.grad is not None
    assert cloud.state_update.gate_bias.grad is not None


def test_absolute_hidden_rollout_accumulates_gated_updates() -> None:
    cloud = AbsoluteMolecularFieldCloud(
        field_channels=1,
        condition_dim=16,
        dim=16,
        heads=4,
        condition_cross_depth=1,
        noise_token_count=4,
        noise_token_dim=8,
        refine_depth=1,
        cvt_kernel_sizes=(3, 3, 3),
        cvt_grid_sizes=(4, 2, 1),
        max_level=8,
        decoder_dim=16,
        gradient_checkpointing=False,
    )

    class ConstantUpdate(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_inputs: list[torch.Tensor] = []

        def forward(
            self,
            encoder_hidden: torch.Tensor,
            model_hidden: torch.Tensor,
            levels: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            del model_hidden, levels
            self.encoder_inputs.append(encoder_hidden.detach().clone())
            update = torch.full_like(encoder_hidden, 0.25)
            gate = torch.ones_like(encoder_hidden)
            return encoder_hidden + update, gate, update

    state_update = ConstantUpdate()
    cloud.state_update = state_update
    model = AbsoluteHiddenRolloutCloud(cloud)
    initial = torch.randn(1, 1, 8, 8)
    condition = torch.randn(1, 5, 16)
    levels = torch.tensor([8, 6, 4, 2])
    intermediate_noise = torch.randn(1, 3, 4, 8)
    final_noise = torch.randn(1, 2, 4, 8)
    anchor = cloud.encode_anchor(initial)

    output = model(
        initial,
        condition,
        levels,
        final_samples=2,
        intermediate_noise=intermediate_noise,
        final_noise=final_noise,
    )

    expected = anchor + 4 * 0.25
    torch.testing.assert_close(output.final_hidden[:, 0], expected)
    torch.testing.assert_close(output.final_hidden[:, 1], expected)
    assert len(state_update.encoder_inputs) == 4
    torch.testing.assert_close(state_update.encoder_inputs[0], anchor)
    torch.testing.assert_close(state_update.encoder_inputs[1], anchor + 0.25)
    torch.testing.assert_close(state_update.encoder_inputs[2], anchor + 0.50)
    torch.testing.assert_close(
        state_update.encoder_inputs[3],
        (anchor + 0.75).expand(2, -1, -1, -1),
    )


def test_absolute_hidden_rollout_can_remove_all_level_conditioning() -> None:
    cloud = AbsoluteMolecularFieldCloud(
        field_channels=1,
        condition_dim=16,
        dim=16,
        heads=4,
        condition_cross_depth=1,
        noise_token_count=4,
        noise_token_dim=8,
        refine_depth=1,
        cvt_kernel_sizes=(3, 3, 3),
        cvt_grid_sizes=(4, 2, 1),
        max_level=8,
        decoder_dim=16,
        gradient_checkpointing=False,
    )
    model = AbsoluteHiddenRolloutCloud(cloud, use_level_conditioning=False).eval()
    initial = torch.randn(1, 1, 8, 8)
    condition = torch.randn(1, 5, 16)
    intermediate_noise = torch.randn(1, 3, 4, 8)
    final_noise = torch.randn(1, 2, 4, 8)

    with torch.no_grad():
        first = model(
            initial,
            condition,
            torch.tensor([8, 6, 4, 2]),
            final_samples=2,
            intermediate_noise=intermediate_noise,
            final_noise=final_noise,
        )
        second = model(
            initial,
            condition,
            torch.tensor([7, 5, 3, 1]),
            final_samples=2,
            intermediate_noise=intermediate_noise,
            final_noise=final_noise,
        )

    torch.testing.assert_close(first.fields, second.fields)
    assert not cloud.level_embedding.weight.requires_grad
    assert not cloud.state_update.level_embedding.weight.requires_grad
    assert not cloud.state_update.level_gate.weight.requires_grad
    assert not cloud.state_update.level_amplitude.weight.requires_grad


def test_absolute_trajectory_rollout_keeps_four_paths_for_every_step() -> None:
    cloud = AbsoluteMolecularFieldCloud(
        field_channels=1,
        condition_dim=16,
        dim=16,
        heads=4,
        condition_cross_depth=1,
        noise_token_count=4,
        noise_token_dim=8,
        refine_depth=1,
        cvt_kernel_sizes=(3, 3, 3),
        cvt_grid_sizes=(4, 2, 1),
        max_level=8,
        decoder_dim=16,
        gradient_checkpointing=False,
    )
    model = AbsoluteTrajectoryRolloutCloud(
        cloud, intermediate_refine_depth=1, use_level_conditioning=False
    )
    initial = torch.randn(2, 1, 8, 8)
    condition = torch.randn(2, 5, 16, requires_grad=True)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    levels = torch.tensor([[8, 6, 4, 2], [7, 5, 3, 1]])
    noise = torch.randn(2, 4, 4, 4, 8)
    condition_batch_sizes: list[int] = []

    def record_condition_batch(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        condition_batch_sizes.append(inputs[0].shape[0])

    handle = cloud.condition_blocks[0].register_forward_pre_hook(
        record_condition_batch
    )

    output = model(
        initial,
        condition,
        levels,
        samples=4,
        noise=noise,
        condition_mask=mask,
        compute_hidden_consistency=True,
    )
    handle.remove()

    assert output.fields.shape == (2, 4, 4, 1, 8, 8)
    assert output.final_hidden.shape == (2, 4, 8, 8, 16)
    assert output.hidden_consistency_mse is not None
    assert output.hidden_consistency_mse.shape == (2, 4, 4)
    assert output.hidden_consistency_mse.requires_grad
    assert output.gate_means.shape == (2, 4, 4)
    assert output.update_rms.shape == (2, 4, 4)
    assert output.spatial_noise_energy.shape == (2, 4, 4, 8, 8)
    assert condition_batch_sizes == [2, 8, 8, 8]
    torch.testing.assert_close(
        output.fields.mean(dim=(-2, -1)),
        torch.zeros(2, 4, 4, 1),
        atol=2e-6,
        rtol=0.0,
    )

    decoder_gradient = torch.autograd.grad(
        output.hidden_consistency_mse.mean(),
        cloud.output_head.direct_projection.weight,
        retain_graph=True,
    )[0]
    assert decoder_gradient.abs().sum() > 0
    (output.fields.square().mean() + output.hidden_consistency_mse.mean()).backward()
    assert condition.grad is not None
    assert cloud.field_encoder.input_projection.weight.grad is not None
    assert cloud.state_update.gate_bias.grad is not None


def test_absolute_trajectory_rollout_can_reencode_every_four_steps() -> None:
    cloud = AbsoluteMolecularFieldCloud(
        field_channels=1,
        condition_dim=16,
        dim=16,
        heads=4,
        condition_cross_depth=1,
        noise_token_count=4,
        noise_token_dim=8,
        refine_depth=1,
        cvt_kernel_sizes=(3, 3, 3),
        cvt_grid_sizes=(4, 2, 1),
        max_level=8,
        decoder_dim=16,
        gradient_checkpointing=False,
    )
    model = AbsoluteTrajectoryRolloutCloud(
        cloud, intermediate_refine_depth=1, use_level_conditioning=False
    ).eval()
    encoder_calls = 0

    def count_encoder_calls(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    handle = cloud.field_encoder.register_forward_pre_hook(count_encoder_calls)
    with torch.no_grad():
        output = model(
            torch.randn(1, 1, 8, 8),
            torch.randn(1, 5, 16),
            torch.tensor([8, 7, 6, 5, 4, 3, 2, 1]),
            samples=4,
            noise=torch.randn(1, 8, 4, 4, 8),
            reencode_every=4,
        )
    handle.remove()

    assert output.fields.shape == (1, 8, 4, 1, 8, 8)
    assert encoder_calls == 2


def test_peak_embedding_is_affine_in_raw_intensity() -> None:
    embedding = PositionwiseAffinePeakEmbedding(
        dim=16,
        position_dim=8,
        mz_bin_width=0.01,
        mz_upper_bound=1_000.0,
    )
    mz = torch.tensor([[100.0, 250.0, 900.0]])
    zero = embedding(mz, torch.zeros_like(mz))
    one = embedding(mz, torch.ones_like(mz))
    two = embedding(mz, torch.full_like(mz, 2.0))

    torch.testing.assert_close(two - zero, 2.0 * (one - zero))
    _, intercept = embedding.coefficients(mz)
    torch.testing.assert_close(zero, intercept)
    assert embedding.position_indices(torch.tensor([100.0, 100.01])).tolist() == [10000, 10001]


def test_distance_convolution_treats_far_index_neighbor_as_zero() -> None:
    torch.manual_seed(23)
    block = ExponentialDistanceConvBlock(
        dim=8,
        heads=2,
        kernel_size=3,
        tau_min=0.01,
        tau_max=0.02,
        cutoff_multiplier=4.0,
        dropout=0.0,
    ).eval()
    tokens = torch.randn(1, 2, 8)
    masses = torch.tensor([[100.0, 200.0]])
    both = block(tokens, masses, masses, torch.tensor([[True, True]]))
    isolated = block(tokens, masses, masses, torch.tensor([[True, False]]))

    torch.testing.assert_close(both[:, 0], isolated[:, 0])


def test_spectrum_condition_attention_is_set_invariant() -> None:
    encoder = SpectrumConditionEncoder(
        metadata_dim=5,
        dim=32,
        heads=4,
        peak_conv_stages=2,
        spectrum_layers=2,
        dropout=0.0,
        peak_position_dim=8,
        mz_bin_width=1.0,
        mz_upper_bound=1_000.0,
    ).eval()
    peaks, peak_mask, spectrum_mask, metadata, precursor_mz = _spectrum_inputs()
    expected = encoder(peaks, peak_mask, spectrum_mask, metadata, precursor_mz)

    peak_order = torch.tensor([2, 0, 5, 1, 4, 3])
    peak_permuted = encoder(
        peaks[:, :, peak_order],
        peak_mask[:, :, peak_order],
        spectrum_mask,
        metadata,
        precursor_mz,
    )
    spectrum_order = torch.tensor([2, 0, 1])
    spectrum_permuted = encoder(
        peaks[:, spectrum_order],
        peak_mask[:, spectrum_order],
        spectrum_mask[:, spectrum_order],
        metadata[:, spectrum_order],
        precursor_mz[:, spectrum_order],
    )

    torch.testing.assert_close(peak_permuted, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(spectrum_permuted, expected, atol=2e-6, rtol=2e-6)


def test_spectrum_condition_attention_uses_all_peak_chunks() -> None:
    encoder = SpectrumConditionEncoder(
        metadata_dim=2,
        dim=16,
        heads=4,
        peak_conv_stages=1,
        spectrum_layers=1,
        dropout=0.0,
        peak_position_dim=8,
        mz_bin_width=1.0,
        mz_upper_bound=1_000.0,
        peak_chunk_size=4,
    ).eval()
    peaks = torch.tensor(
        [[[[100.0, 1.0], [150.0, 0.8], [200.0, 0.6], [250.0, 0.4], [300.0, 0.01], [350.0, 0.001]]]]
    )
    peak_mask = torch.ones(1, 1, 6, dtype=torch.bool)
    spectrum_mask = torch.ones(1, 1, dtype=torch.bool)
    metadata = torch.zeros(1, 1, 2)
    precursor_mz = torch.tensor([[400.0]])

    truncated = encoder(
        peaks[:, :, :4],
        peak_mask[:, :, :4],
        spectrum_mask,
        metadata,
        precursor_mz,
    )
    complete = encoder(peaks, peak_mask, spectrum_mask, metadata, precursor_mz)

    assert not torch.allclose(complete, truncated, atol=2e-6, rtol=2e-6)
