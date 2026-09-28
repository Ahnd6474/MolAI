import torch

from molai.models.bridge import VPSchedule
from molai.models.cloud import MolecularCloudModel
from molai.models.image_smiles import FieldToSmiles
from molai.models.losses import FullBandEnergyDistance
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
