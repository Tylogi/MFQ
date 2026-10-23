"""Native CUDA row-bucket graphs with recurrent and full-attention request state."""

import json
import os
import subprocess

import numpy as np
import pytest

from mfq.formats import io
from mfq.formats.assets import MODEL_CONFIG_ASSET
from tests.test_cuda_kv_quantization_native import write_qwen35_fixture


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
    line = next(line for line in result.stdout.splitlines()
                if line.startswith("continuous_batching_sampling_membership_check "))
    sampling = dict(field.split("=", 1) for field in line.split()[1:])
    assert int(sampling["captures"]) == 1
    assert int(sampling["bucket_reuses"]) > 0
    assert sampling["serial"] == sampling["inactive_state_bits"] == sampling["inactive_rng"] == "PASS"
    assert sampling["padding_routes"] == "PASS"
    assert "continuous_batching_check PASS" in result.stdout
    if requests == 32:
        assert "continuous_batching_growth_check requests=32 boundary=256 PASS" in result.stdout


@pytest.mark.parametrize("family,requests", [("qwen35", 2), ("flash_next", 2), ("flash_next", 32)])
def test_stochastic_sampling_reuses_graph_bucket_during_membership_changes(tmp_path, family, requests):
    executable = os.environ.get("MFQ_CUDA_BATCH_GRAPH_DIAGNOSTICS")
    if not executable:
        pytest.skip("MFQ_CUDA_BATCH_GRAPH_DIAGNOSTICS must point to CUDA mfq-diagnostics")
    environment = dict(os.environ)
    if family == "qwen35":
        model = tmp_path / "recurrent.mfq"
        _recurrent_model(model)
    else:
        from tests.test_native_nint_ple import _models

        _, model = _models(tmp_path, qsa=True, ple_layer=3, context_size=512, quantized_experts=requests == 2)
        if requests == 2:
            environment.update(MFQ_MOE_PIPELINE="1", MFQ_MOE_TWO_STAGE_FFN="1", MFQ_MOE_PRELOAD_ALL="1",
                               MFQ_MOE_ASSERT_RESIDENT="1", MFQ_MOE_HOST_CACHE_BYTES="8388608")
    arguments = [executable, "--model", str(model), "--ctx-size", "512", "--check-continuous-batching",
                 "--continuous-batching-check-requests", str(requests)]
    if family == "flash_next" and requests == 32:
        arguments += ["--moe-gpu-cache-gb", "0"]
    result = subprocess.run(
        arguments,
        env=environment,
        text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines()
                if line.startswith("continuous_batching_sampling_membership_check "))
    metrics = dict(field.split("=", 1) for field in line.split()[1:])
    assert int(metrics["captures"]) == 1
    assert int(metrics["bucket_reuses"]) == int(metrics["steps"]) - 1
    assert all(metrics[name] == "PASS" for name in ("serial", "inactive_state_bits", "inactive_rng", "padding_routes"))
    assert "continuous_batching_sampling_oracle logits=tied" in result.stdout
