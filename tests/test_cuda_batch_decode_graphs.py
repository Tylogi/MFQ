"""Native CUDA row-bucket graphs with recurrent and full-attention request state."""

import json
import os
import subprocess

import numpy as np
import pytest

from mfq.formats import io
from mfq.formats.assets import MODEL_CONFIG_ASSET
from tests.test_cuda_kv_quantization_native import write_qwen35_fixture
from tests.test_native_nint_ple import _models


def _recurrent_model(path):
    write_qwen35_fixture(path)
    header, weights = io.load(path)
    config = json.loads(weights[MODEL_CONFIG_ASSET])
    config.update(
        max_position_embeddings=512,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
    )
    weights[MODEL_CONFIG_ASSET] = json.dumps(config).encode()
    rng = np.random.default_rng(6318)
    root = "model.block.0.linear_attention."
    for role, shape in {
        "qkv": (128, 32), "gate": (64, 32), "alpha": (2, 32),
        "beta": (2, 32), "output": (32, 64),
    }.items():
        weights[root + role + ".weight"] = rng.normal(0, .03, shape).astype(np.float32)
    weights[root + "conv.weight"] = rng.normal(0, .03, (128, 1, 4)).astype(np.float32)
    weights[root + "norm.weight"] = np.ones(32, np.float32)
    weights[root + "a"] = -np.ones(2, np.float32)
    weights[root + "dt_bias"] = np.zeros(2, np.float32)
    io.save(path, header, weights)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("requests", [2, 32])
def test_qwen_continuous_graph_reuses_bucket_and_preserves_inactive_state(tmp_path, paged, requests):
    executable = os.environ.get("MFQ_CUDA_BATCH_GRAPH_DIAGNOSTICS")
    if not executable:
        pytest.skip("MFQ_CUDA_BATCH_GRAPH_DIAGNOSTICS must point to CUDA mfq-diagnostics")
    model = tmp_path / "recurrent.mfq"
    _recurrent_model(model)
    result = subprocess.run(
        [executable, "--model", str(model), "--ctx-size", "512", "--check-continuous-batching",
         "--continuous-batching-check-requests", str(requests)],
        env=dict(os.environ, MFQ_CONTINUOUS_PAGED_KV=str(int(paged))),
        text=True, capture_output=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines()
                if line.startswith("continuous_batching_membership_check "))
    metrics = dict(field.split("=", 1) for field in line.split()[1:])
    assert int(metrics["captures"]) == 1
    assert int(metrics["bucket_reuses"]) > 0
    assert metrics["serial"] == metrics["inactive_state_bits"] == "PASS"
    assert metrics["padding_routes"] == "PASS"
    assert "continuous_batching_check PASS" in result.stdout
    if requests == 32:
        assert "continuous_batching_growth_check requests=32 boundary=256 PASS" in result.stdout


@pytest.mark.parametrize("requests", [2, 8, 16, 32])
@pytest.mark.parametrize("kv_bits", ["0", "3.5"])
def test_qsa_independent_graph_heterogeneous_requests_match_serial(tmp_path, requests, kv_bits):
    executable = os.environ.get("MFQ_CUDA_BATCH_GRAPH_DIAGNOSTICS")
    if not executable:
        pytest.skip("MFQ_CUDA_BATCH_GRAPH_DIAGNOSTICS must point to CUDA mfq-diagnostics")
    _, model = _models(tmp_path, qsa=True, ple_layer=3, context_size=512)
    result = subprocess.run(
        [executable, "--model", str(model), "--ctx-size", "512", "--check-continuous-batching",
         "--continuous-batching-check-requests", str(requests), "--moe-gpu-cache-gb", "0"],
        env=dict(os.environ, MFQ_QSA_INDEPENDENT_BATCH="1", MFQ_KV_TURBOQUANT_BITS=kv_bits),
        text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "continuous_batching_check PASS" in result.stdout
    membership = next(line for line in result.stdout.splitlines()
                      if line.startswith("continuous_batching_membership_check "))
    metrics = dict(field.split("=", 1) for field in membership.split()[1:])
    assert int(metrics["captures"]) == 1
    assert int(metrics["bucket_reuses"]) > 0
    assert metrics["serial"] == metrics["inactive_state_bits"] == metrics["padding_routes"] == "PASS"
    if requests > 2:
        assert f"continuous_batching_growth_check requests={requests} boundary=256 PASS" in result.stdout
