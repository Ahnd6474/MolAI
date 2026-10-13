import copy

import pytest
import torch

from molai.data.spectrum_metadata import SpectrumMetadataSchema
from molai.models.condition import (
    AcquisitionMetadataTokens,
    PeakMetadataCrossAttention,
    SpectrumTokenEncoder,
    bind_spectrum_metadata,
)


@pytest.fixture
def schema():
    return SpectrumMetadataSchema({
        "ionization_mode": ["negative", "positive"], "adduct": ["[m+h]+", "[m-h]-"],
        "instrument_type": ["orbitrap", "timstof"],
        "collision_energy_orig_units": ["ev", "nce"],
    }, 3, 3)


def make_encoder(schema, chunk_batch=256):
    return SpectrumTokenEncoder(
        metadata_dim=len(schema.feature_names), metadata_feature_names=schema.feature_names,
        metadata_fusion="preconv_cross", metadata_cross_gate_init=.1,
        dim=16, heads=4, dropout=0., peak_position_dim=4,
        mz_bin_width=1., mz_upper_bound=1000., peak_chunk_size=8,
        peak_chunk_batch=chunk_batch,
    ).eval()


def inputs(schema):
    peaks = torch.stack((torch.linspace(100., 300., 32).reshape(4, 8),
                         torch.linspace(.1, 1., 32).reshape(4, 8)), dim=-1)
    mask = torch.ones(4, 8, dtype=torch.bool)
    mask[0, -3:] = False
    metadata = torch.stack([
        schema.encode({"adduct": "[M+H]+", "precursor_mz": 400.,
                       "instrument_type": "timsTOF", "collision_energy_ev": [20., 40.]}, 8),
        schema.encode({"adduct": "[M-H]-", "precursor_mz": 400.,
                       "instrument_type": "Orbitrap", "collision_energy_ev": [60.]}, 8),
        schema.encode({}, 8),
    ])
    # Spectrum ownership is deliberately interleaved. Spectra 0 and 1 belong to
    # one molecule but must remain isolated until post-pooling MS self-attention.
    return (peaks, mask, torch.tensor([1, 0, 1, 2]), torch.tensor([0, 0, 1]),
            metadata, torch.tensor([400., 400., 0.]), 2)


def test_cross_precedes_two_conv_pool_and_two_self_attention_stages(schema, monkeypatch):
    import molai.models.condition as module

    encoder = make_encoder(schema)
    events = []
    handles = [encoder.metadata_cross.register_forward_pre_hook(lambda *_: events.append("cross"))]
    handles += [block.register_forward_pre_hook(lambda *_: events.append("conv"))
                for block in encoder.peak_conv_blocks]
    handles += [block.register_forward_pre_hook(lambda *_: events.append("self"))
                for block in encoder.spectrum_attention_blocks]
    pool = module.indexwise_max_pool

    def traced_pool(*args):
        events.append("pool")
        return pool(*args)

    monkeypatch.setattr(module, "indexwise_max_pool", traced_pool)
    try:
        encoder.forward_ragged_tokens(*inputs(schema))
    finally:
        for handle in handles:
            handle.remove()
    assert events == ["cross", "conv", "pool", "conv", "pool", "self", "self"]
    assert len(encoder.metadata_tokenizer.group_names) == 7
    assert not hasattr(encoder, "metadata_embedding")
    assert encoder.metadata_cross.gate.item() == pytest.approx(.1)


def test_preconv_cross_has_no_cross_spectrum_leakage(schema):
    encoder = make_encoder(schema)
    arguments = list(inputs(schema))
    before, segments, _ = encoder._encode_compressed_peaks_ragged(*arguments)
    changed = arguments[4].clone()
    changed[0] += 1.0
    arguments[4] = changed
    after, new_segments, _ = encoder._encode_compressed_peaks_ragged(*arguments)
    torch.testing.assert_close(segments, new_segments)
    torch.testing.assert_close(before[segments != 0], after[segments != 0])
    assert not torch.allclose(before[segments == 0], after[segments == 0])


def test_zero_gate_removes_all_metadata_and_precursor_effects(schema):
    encoder = make_encoder(schema)
    with torch.no_grad():
        encoder.metadata_cross.gate.zero_()
    arguments = list(inputs(schema))
    before, mask = encoder.forward_ragged_tokens(*arguments)
    arguments[4] = torch.randn_like(arguments[4]) * 10
    arguments[5] = torch.tensor([700., 200., 100.])
    after, new_mask = encoder.forward_ragged_tokens(*arguments)
    torch.testing.assert_close(mask, new_mask)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_cross_padding_is_zero_and_updates_depend_on_peak_query():
    torch.manual_seed(8)
    cross = PeakMetadataCrossAttention(16, 4, 0., .1)
    peaks = torch.randn(2, 5, 16)
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    memory = torch.randn(2, 7, 16)
    key, value = cross.project_context(memory)
    output = cross(peaks, mask, key, value)
    assert not output[~mask].any()
    update = output - peaks
    assert not torch.allclose(update[0, 0], update[0, 1])
    output.square().sum().backward()
    assert torch.isfinite(cross.gate.grad)
    assert cross.query.weight.grad.abs().sum() > 0
    assert cross.key_value.weight.grad.abs().sum() > 0


def test_chunk_batching_preserves_outputs_and_all_typed_groups_have_gradients(schema):
    encoder = make_encoder(schema)
    chunked = make_encoder(schema, 1)
    chunked.load_state_dict(encoder.state_dict(), strict=True)
    output, mask = encoder.forward_ragged_tokens(*inputs(schema))
    other, other_mask = chunked.forward_ragged_tokens(*inputs(schema))
    torch.testing.assert_close(mask, other_mask)
    torch.testing.assert_close(output, other)
    output.square().sum().backward()
    for projection in encoder.metadata_tokenizer.projections:
        assert projection[0].weight.grad is not None
        assert projection[0].weight.grad.abs().sum() > 0
    assert encoder.precursor_embedding[0].weight.grad.abs().sum() > 0
    assert encoder.metadata_cross.gate.grad.abs().sum() > 0


def test_saved_feature_binding_roundtrips_and_rejects_legacy_or_changed_order(schema):
    manifest = {"metadata_schema": schema.to_dict(), "metadata_features": schema.feature_names,
                "metadata_dim": len(schema.feature_names)}
    options = {"metadata_fusion": "preconv_cross"}
    bound = bind_spectrum_metadata(options, manifest)
    assert options["metadata_feature_names"] == schema.feature_names
    encoder = SpectrumTokenEncoder(metadata_dim=manifest["metadata_dim"], dim=16, heads=4,
                                   peak_position_dim=4, mz_bin_width=1., mz_upper_bound=1000.,
                                   dropout=0., **bound).eval()
    restored = SpectrumTokenEncoder(metadata_dim=manifest["metadata_dim"], dim=16, heads=4,
                                    peak_position_dim=4, mz_bin_width=1., mz_upper_bound=1000.,
                                    dropout=0., **copy.deepcopy(options)).eval()
    restored.load_state_dict(encoder.state_dict(), strict=True)
    torch.testing.assert_close(encoder.forward_ragged_tokens(*inputs(schema))[0],
                               restored.forward_ragged_tokens(*inputs(schema))[0])
    with pytest.raises(ValueError, match="v2"):
        bind_spectrum_metadata({"metadata_fusion": "preconv_cross"}, {"metadata_dim": 6})
    with pytest.raises(ValueError, match="feature order"):
        bind_spectrum_metadata(options, {**manifest, "metadata_features": schema.feature_names[::-1]})
    with pytest.raises(ValueError, match="unrecognized"):
        AcquisitionMetadataTokens([*schema.feature_names, "precursor_error_ppm"], 16)


def test_historical_default_keeps_additive_state_keys_and_strict_restore():
    encoder = SpectrumTokenEncoder(metadata_dim=6, dim=16, heads=4, peak_position_dim=4,
                                   mz_bin_width=1., mz_upper_bound=1000.)
    assert encoder.metadata_fusion == "add"
    assert hasattr(encoder, "metadata_embedding")
    assert not hasattr(encoder, "metadata_cross")
    restored = copy.deepcopy(encoder)
    restored.load_state_dict(encoder.state_dict(), strict=True)


def test_preconv_cross_fullgraph_forward_backward(schema):
    tokenizer = AcquisitionMetadataTokens(schema.feature_names, 16)
    cross = PeakMetadataCrossAttention(16, 4, 0., .1)
    metadata = inputs(schema)[4]
    memory = tokenizer(metadata, torch.zeros(3, 16))
    key, value = cross.project_context(memory)
    peaks, mask = torch.randn(3, 8, 16), torch.ones(3, 8, dtype=torch.bool)
    compiled = torch.compile(cross, backend="aot_eager", fullgraph=True)
    output = compiled(peaks, mask, key, value)
    torch.testing.assert_close(output, cross(peaks, mask, key, value))
    output.square().sum().backward()
    assert cross.gate.grad is not None
    assert tokenizer.type_embedding.grad is not None
