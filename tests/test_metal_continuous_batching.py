from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.test_cuda_flash_next_runtime import (
    independent_qwen_attention,
    mtp_fixture,
    write_qwen_fixture,
)


@pytest.fixture(scope="module", params=[32, 128])
def batch_model(tmp_path_factory, request):
    if not os.environ.get("MFQ_METAL_BATCHING_TEST"):
        pytest.skip("MFQ_METAL_BATCHING_TEST required")
    from mfq.formats.io import Float8E4M3Array
    from mfq.formats.mfe import MfePool, MfeTensor
    from mfq.formats.nint import NintSpec
    from mfq.quantize.nint_quant import quantize

    config, weights = mtp_fixture("qwen", layers=1)
    config["text_config"].update(
        max_position_embeddings=1024, num_attention_heads=24, num_key_value_heads=2, num_experts=7,
        head_dim=256, indexer_n_heads=4, indexer_head_dim=128,
        indexer_compress_ratio=4, indexer_budget=64,
        rope_parameters={"rope_theta": 1e7}, eos_token_id=31,
    )
    rng = np.random.default_rng(3817)
    dimension = request.param
    config["text_config"].update(linear_key_head_dim=dimension, linear_value_head_dim=dimension)
    if dimension != 32:
        prefix = "model.block.0.linear_attention"
        for name, shape in {
            "qkv": (4 * dimension, 16), "gate": (2 * dimension, 16), "output": (16, 2 * dimension),
        }.items():
            weights[f"{prefix}.{name}.weight"] = rng.normal(0, .05, shape).astype(np.float32)
        convolution = np.zeros((4 * dimension, 1, 4), np.float32)
        convolution[:, 0, -1] = .8
        convolution[:, 0, -2] = .1
        weights[f"{prefix}.conv.weight"] = convolution
        weights[f"{prefix}.norm.weight"] = np.ones(dimension, np.float32)
    weights["model.token_embedding.weight"] = rng.uniform(.1, .4, (32, 16)).astype(np.float32)
    ple_rng = np.random.default_rng(59731)
    for name, value in list(weights.items()):
        if ".position_embedding.ngram.shard." in name:
            if dimension == 128:
                weights[name] = quantize(ple_rng.uniform(.1, .4, value.shape).astype(np.float32),
                    NintSpec(bits=5, groupsize=16, sub_bits=3))
            else:
                weights[name] = ple_rng.integers(0x24, 0x2c, value.shape, dtype=np.uint8).view(Float8E4M3Array)
    weights["model.block.0.position_embedding.ngram.weight_scale"] = np.array([1.0], np.float32)
    for prefix in ("model.block.1.attention", "predictor.block.0.attention"):
        for name, shape in {
            "query": (24 * 2 * 256, 16), "key": (2 * 256, 16), "value": (2 * 256, 16),
            "output": (16, 24 * 256), "indexer.query_key": (5 * 128, 16),
        }.items():
            weights[f"{prefix}.{name}.weight"] = rng.normal(0, .05, shape).astype(np.float32)
        weights[f"{prefix}.indexer.query_key.weight"] = rng.uniform(.01, .09, (5 * 128, 16)).astype(np.float32)
        for name, width in {"query_norm": 256, "key_norm": 256,
                            "indexer.query_norm": 128, "indexer.key_norm": 128}.items():
            weights[f"{prefix}.{name}.weight"] = rng.normal(0, .02, width).astype(np.float32)
    for name, value in list(weights.items()):
        if ".experts." not in name:
            continue
        value = np.stack([value[row % value.shape[0]] * (1 + row * .05) for row in range(7)])
        matrix = quantize(value.reshape(-1, value.shape[-1]), NintSpec(bits=8, groupsize=16, sub_bits=7))
        weights[name] = MfeTensor(shape=value.shape, pools=(
            MfePool(np.arange(value.shape[0], dtype=np.int32), matrix),
        ))
    for name, value in list(weights.items()):
        if name.endswith(".mlp.router.weight"):
            weights[name] = rng.normal(0, .05, (7, value.shape[-1])).astype(np.float32)
    model = tmp_path_factory.mktemp("metal-batching") / "batch.mfq"
    write_qwen_fixture(model, config, independent_qwen_attention(config, weights), predictor=True)
    return model


@pytest.mark.parametrize("streamed,kv_bits", [(False, 0), (True, 0), (True, 4)])
def test_metal_continuous_batching(batch_model: Path, streamed: bool, kv_bits: int):
    binary = os.environ.get("MFQ_METAL_BATCHING_TEST")
    if not binary:
        pytest.skip("MFQ_METAL_BATCHING_TEST required")
    environment = os.environ.copy()
    for key in ("MFQ_QSA_KV_BUDGET_BYTES", "MFQ_QSA_KV_TARGET_CONTEXT", "MFQ_KV_TURBOQUANT_BITS"):
        environment.pop(key, None)
    arguments = [binary, str(batch_model)]
    if streamed:
        arguments.append("ssd")
        environment.update(MFQ_QSA_KV_BUDGET_BYTES="131072", MFQ_QSA_KV_TARGET_CONTEXT="1024")
    if kv_bits:
        environment["MFQ_KV_TURBOQUANT_BITS"] = str(kv_bits)
    result = subprocess.run(arguments, env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Metal continuous batching state and scheduling passed" in result.stdout
