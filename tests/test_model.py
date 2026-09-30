import torch

from molai.models.bridge import GeometricVESchedule, VPSchedule
from molai.models.cloud import MolecularCloudModel, MolecularFieldCloud
from molai.models.condition import PositionwiseAffinePeakEmbedding, SpectrumConditionEncoder
from molai.models.image_smiles import FieldToSmiles
from molai.models.losses import FullBandEnergyDistance, full_band_distance
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


def test_bridge_and_energy_distance() -> None:
    clean = torch.randn(2, 1, 16, 16).tanh()
    bridge = VPSchedule(steps=20).sample_training_batch(clean, samples=3)
    loss = FullBandEnergyDistance(levels=2)(
        bridge.target_cloud, bridge.target_cloud, bridge.current
    )

    assert bridge.current.shape == clean.shape
    assert bridge.target_cloud.shape == (2, 3, 1, 16, 16)
    assert abs(float(loss)) < 1e-5


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


def test_reused_band_pyramid_matches_direct_error_filtering() -> None:
    torch.manual_seed(4)
    first = torch.randn(3, 1, 16, 16)
    second = torch.randn(3, 1, 16, 16)
    error = first - second
    direct_distances = []
    kernel_1d = first.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    kernel = torch.outer(kernel_1d, kernel_1d)[None, None]
    for level in range(3):
        low = torch.nn.functional.conv2d(
            error, kernel, padding=2 * 2**level, dilation=2**level
        )
        direct_distances.append(
            torch.sqrt((error - low).square() + 1e-6).mean(dim=(1, 2, 3))
        )
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
        peak_layers=2,
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
    assert encoder.peak_blocks[0].attention.relative_projection.weight.grad is not None


def test_spectrum_condition_attention_supports_bfloat16_autocast() -> None:
    encoder = SpectrumConditionEncoder(
        metadata_dim=5,
        dim=32,
        heads=4,
        peak_layers=1,
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


def test_spectrum_condition_attention_is_set_invariant() -> None:
    encoder = SpectrumConditionEncoder(
        metadata_dim=5,
        dim=32,
        heads=4,
        peak_layers=2,
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
        peak_layers=1,
        spectrum_layers=1,
        dropout=0.0,
        peak_position_dim=8,
        mz_bin_width=1.0,
        mz_upper_bound=1_000.0,
        peak_chunk_size=4,
    ).eval()
    peaks = torch.tensor(
        [[[[100.0, 1.0], [150.0, 0.8], [200.0, 0.6], [250.0, 0.4],
           [300.0, 0.01], [350.0, 0.001]]]]
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
