import os
import subprocess

import numpy as np
import pytest

from mfq.formats.io import Float8E4M3Array
from tests.test_cuda_flash_next_runtime import mtp_fixture, write_qwen_fixture


@pytest.mark.parametrize("budget", [128 << 10, 32 << 20])
@pytest.mark.parametrize("native_context", [1024, 256])
def test_complete_qsa_offload_prefix_and_mtp(tmp_path, budget, native_context):
    binary = os.environ.get("MFQ_QSA_OFFLOAD_TEST")
    if not binary:
        pytest.skip("MFQ_QSA_OFFLOAD_TEST required")
    config, weights = mtp_fixture("qwen", layers=1)
    for name, value in list(weights.items()):
        if ".position_embedding.ngram.shard." in name:
            weights[name] = np.full(value.shape, 0x28, np.uint8).view(Float8E4M3Array)
    weights["model.block.0.position_embedding.ngram.weight_scale"] = np.array([1.0], np.float32)
    text = config["text_config"]
    text.update(max_position_embeddings=native_context, num_attention_heads=24, num_key_value_heads=2,
        head_dim=256, indexer_n_heads=4, indexer_head_dim=128, indexer_compress_ratio=4,
        indexer_budget=64, rope_parameters={"rope_theta": 1e7}, eos_token_id=31)
    rng = np.random.default_rng(3817)
    weights["model.token_embedding.weight"] = rng.uniform(.1, .4, size=(32, 16)).astype(np.float32)
    for prefix in ["model.block.1.attention", "predictor.block.0.attention"]:
        for name, shape in {"query": (24 * 2 * 256, 16), "key": (2 * 256, 16),
            "value": (2 * 256, 16), "output": (16, 24 * 256),
            "indexer.query_key": (5 * 128, 16)}.items():
            weights[f"{prefix}.{name}.weight"] = rng.normal(scale=.05, size=shape).astype(np.float32)
        weights[f"{prefix}.indexer.query_key.weight"] = rng.uniform(.01, .09, size=(5 * 128, 16)).astype(np.float32)
        for name, width in {"query_norm": 256, "key_norm": 256,
            "indexer.query_norm": 128, "indexer.key_norm": 128}.items():
            weights[f"{prefix}.{name}.weight"] = rng.normal(scale=.02, size=width).astype(np.float32)
    path = tmp_path / "qsa-offload.mfq"
    write_qwen_fixture(path, config, weights, predictor=True)
    result = subprocess.run([binary, str(path), str(budget)], text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Complete model offload, prefix restore and MTP checks passed" in result.stdout


@pytest.mark.parametrize('bits', [2.5, 4])
def test_complete_quantized_qsa_prefix_and_mtp(tmp_path, monkeypatch, bits):
    monkeypatch.setenv('MFQ_KV_TURBOQUANT_BITS', str(bits))
    test_complete_qsa_offload_prefix_and_mtp(tmp_path, 128 << 10, 1024)
