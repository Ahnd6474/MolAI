import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from molai.models.attention import TokenCrossBlock2d


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("diagnose_midstart_ms_ablation")


def test_no_ms_removes_context_dependence_and_retains_image_ffn(diagnostic):
    torch.manual_seed(12)
    block = TokenCrossBlock2d(8, 8, 2).eval()
    cloud = SimpleNamespace(condition_blocks=[block], cache_condition_kv=False)
    query = torch.randn(1, 2, 2, 8)
    first, second = torch.randn(1, 3, 8), torch.randn(1, 5, 8)
    saved = {key: value.clone() for key, value in block.state_dict().items()}
    with torch.no_grad():
        original = block(query, first)
        assert not torch.allclose(original, block(query, second))
        with diagnostic.without_ms_attention(cloud):
            actual = block(query, first)
            torch.testing.assert_close(actual, block(query, second), rtol=0, atol=0)
            torch.testing.assert_close(actual, query + block.ff(query), rtol=0, atol=0)
        torch.testing.assert_close(block(query, first), original, rtol=0, atol=0)
    for key, value in block.state_dict().items():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)


def test_no_ms_restores_forward_on_failure_and_rejects_cached_execution(diagnostic):
    block = TokenCrossBlock2d(8, 8, 2)
    cloud = SimpleNamespace(condition_blocks=[block], cache_condition_kv=False)
    with pytest.raises(RuntimeError, match="intentional"), diagnostic.without_ms_attention(cloud):
        raise RuntimeError("intentional")
    assert "forward" not in block.__dict__
    cloud.cache_condition_kv = True
    with pytest.raises(ValueError, match="Disable condition caching"), diagnostic.without_ms_attention(cloud):
        pass
