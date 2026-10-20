"""Metal Nemotron-H checks over quantized non-gated MFE expert banks.

The fixture machine is the existing synthetic HF checkpoint from
``tests/test_quantize/test_nemotron_quantized_runtime.py``; this module only
swaps the CUDA diagnostics bridge for ``mfq-metal-nemotron-test`` and compares
against the dequantized NumPy oracle.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from tests import test_cuda_nemotron_runtime as oracle  # noqa: E402
from tests.test_quantize import test_nemotron_quantized_runtime as quantized  # noqa: E402
from tests.test_metal_nemotron_runtime import parse_tensors  # noqa: E402


def bridge_binary() -> Path:
    value = os.environ.get("MFQ_METAL_NEMOTRON_TEST")
    if not value:
        pytest.skip("MFQ_METAL_NEMOTRON_TEST required")
    return Path(value)


def run_bridge(path: Path, check: str, context: int = 32) -> dict:
    process = subprocess.run(
        [str(bridge_binary()), "--model", str(path), "--ctx-size", str(context), check],
        text=True, capture_output=True, timeout=900,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    prefix = {
        "--check-nemotron-h": "nemotron_h_check ",
        "--check-nemotron-h-mtp": "nemotron_h_mtp_check ",
    }[check]
    import json
    raw = json.loads(next(
        line.removeprefix(prefix)
        for line in process.stdout.splitlines()
        if line.startswith(prefix)
    ))
    raw.pop("architecture")
    return raw


@pytest.mark.parametrize("preset,extra,tolerance", [
    ("NINT5", ("--preset", "S5-S"), 3e-2),
    # NVQ exercises the compact MFE decoder on the routed latent bank.
    ("NVQ2J", ("--random-expert-mix", "nvq2j"), 2.5e-1),
])
def test_metal_quantized_text_graph_matches_dequantized_reference(
        tmp_path: Path, preset: str, extra: tuple[str, ...], tolerance: float) -> None:
    config, weights = oracle.model_fixture(quantized.GEOMETRY)
    root = tmp_path / "hf"
    quantized._write_hf_checkpoint(root, config, weights)
    artifact = tmp_path / f"{preset}.mfq"
    quantized._convert(root, artifact, *extra)
    dequantized, records = quantized._dequantized(artifact)
    expected_hidden, expected_logits = oracle.model_reference(config, dequantized)
    for layer in (1, 3):
        prefix = f"model.block.{layer}.mlp.experts"
        assert records[f"{prefix}.up.weight"].dtype == "MFE"
        assert records[f"{prefix}.down.weight"].dtype == "MFE"
        assert dequantized[f"{prefix}.up.weight"].shape == (
            oracle.EXPERTS, oracle.MOE_INTERMEDIATE, oracle.MOE_LATENT)
        assert dequantized[f"{prefix}.down.weight"].shape == (
            oracle.EXPERTS, oracle.MOE_LATENT, oracle.MOE_INTERMEDIATE)
    actual = parse_tensors(run_bridge(artifact, "--check-nemotron-h"))
    np.testing.assert_allclose(actual["full"][0], expected_logits[0],
                               atol=tolerance, rtol=5e-2)
    np.testing.assert_allclose(actual["full_hidden"][0], expected_hidden[0],
                               atol=tolerance, rtol=5e-2)
    np.testing.assert_allclose(actual["decode"][0], actual["full"][0][5:],
                               atol=tolerance, rtol=5e-2)
    np.testing.assert_allclose(actual["chunked"][0], actual["full"][0],
                               atol=tolerance, rtol=5e-2)
    np.testing.assert_allclose(actual["restored"][0], actual["continued"][0],
                               atol=tolerance, rtol=5e-2)


def test_metal_quantized_mtp_bank_matches_dequantized_reference(tmp_path: Path) -> None:
    config, weights = oracle.model_fixture(quantized.GEOMETRY, mtp=True)
    root = tmp_path / "hf"
    quantized._write_hf_checkpoint(root, config, weights)
    artifact = tmp_path / "mtp.mfq"
    quantized._convert(root, artifact, "--preset", "S5-S", "--quantize-mtp")
    dequantized, records = quantized._dequantized(artifact)
    bank = "predictor.block.1.mlp.experts"
    assert records[f"{bank}.up.weight"].dtype == "MFE"
    assert records[f"{bank}.down.weight"].dtype == "MFE"
    raw = run_bridge(artifact, "--check-nemotron-h-mtp", context=64)
    previous = np.asarray(raw["previous"]["data"], np.float32).reshape(
        raw["previous"]["shape"])
    ids = np.asarray(raw["ids"]["data"], np.float32).astype(np.int64).reshape(
        raw["ids"]["shape"])
    expected_hidden, expected_logits = oracle.mtp_reference(
        config, dequantized, previous, ids)
    np.testing.assert_allclose(raw["head"]["data"], expected_hidden.reshape(-1),
                               atol=3e-2, rtol=5e-2)
    np.testing.assert_allclose(raw["logits"]["data"], expected_logits.reshape(-1),
                               atol=3e-2, rtol=5e-2)
    assert raw["greedy_mtp"] == raw["greedy_plain"]
    assert raw["greedy_disabled"] == raw["greedy_plain"]
