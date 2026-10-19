"""Quantized Nemotron-H end-to-end checks against the NumPy oracle.

The synthetic HF checkpoint is the same text graph as
``tests/test_cuda_nemotron_runtime.py``, but stored under the released source
names so the normal importer, preset policy and expert-bank stacking run.  Both
the direct expert path and the cached two-stage MFE pipeline are compared with
a reference evaluated on the dequantized artifact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

import mfq.tools.quantize_hf_to_mfq as hf_to_mfq
from mfq.formats.io import (
    bfloat16_to_float32,
    is_bfloat16_array,
    load_mmap,
)
from mfq.formats.mfe import MfeTensor
from mfq.formats.nint import NintTensor
from mfq.quantize.expert_nint import dequantize_expertwise, dequantize_flat_precision

from tests import test_cuda_nemotron_runtime as oracle

GEOMETRY = ("mamba", "moe", "attention", "moe")


def _source_name(canonical: str) -> str:
    """Translate one canonical text tensor to the released checkpoint spelling."""
    name = canonical
    name = name.replace("model.token_embedding.weight",
                        "language_model.backbone.embeddings.weight")
    name = name.replace("model.output.weight", "language_model.lm_head.weight")
    name = name.replace("model.output_norm.weight", "language_model.backbone.norm_f.weight")
    name = name.replace("model.block.", "language_model.backbone.layers.")
    name = name.replace(".attention.norm.weight", ".norm.weight")
    name = name.replace(".linear_attention.dt_bias", ".mixer.dt_bias")
    name = name.replace(".linear_attention.a", ".mixer.A_log")
    name = name.replace(".linear_attention.d", ".mixer.D")
    name = name.replace(".linear_attention.in_projection.weight", ".mixer.in_proj.weight")
    name = name.replace(".linear_attention.conv.weight", ".mixer.conv1d.weight")
    name = name.replace(".linear_attention.conv.bias", ".mixer.conv1d.bias")
    name = name.replace(".linear_attention.norm.weight", ".mixer.norm.weight")
    name = name.replace(".linear_attention.output.weight", ".mixer.out_proj.weight")
    name = name.replace(".attention.query.weight", ".mixer.q_proj.weight")
    name = name.replace(".attention.key.weight", ".mixer.k_proj.weight")
    name = name.replace(".attention.value.weight", ".mixer.v_proj.weight")
    name = name.replace(".attention.output.weight", ".mixer.o_proj.weight")
    name = name.replace(".mlp.router.bias", ".mixer.gate.e_score_correction_bias")
    name = name.replace(".mlp.router.weight", ".mixer.gate.weight")
    name = name.replace(".mlp.latent.down.weight", ".mixer.fc1_latent_proj.weight")
    name = name.replace(".mlp.latent.up.weight", ".mixer.fc2_latent_proj.weight")
    name = name.replace(".mlp.shared_expert.up.weight",
                        ".mixer.shared_experts.up_proj.weight")
    name = name.replace(".mlp.shared_expert.down.weight",
                        ".mixer.shared_experts.down_proj.weight")
    return name


def _write_hf_checkpoint(root: Path, config: dict, weights: dict) -> None:
    """Expand canonical banks into the per-expert source tensors of the release."""
    root.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    for canonical, value in weights.items():
        array = np.asarray(value, np.float32)
        bank = re.match(r"^(.*\.mlp\.experts)\.(up|down)\.weight$", canonical)
        if bank is not None:
            prefix = re.sub(
                r"^model\.block\.(\d+)\.mlp\.experts$",
                r"language_model.backbone.layers.\1.mixer.experts",
                bank.group(1),
            )
            for expert in range(array.shape[0]):
                tensors[f"{prefix}.{expert}.{bank.group(2)}_proj.weight"] = (
                    torch.from_numpy(np.ascontiguousarray(array[expert]))
                )
            continue
        tensors[_source_name(canonical)] = torch.from_numpy(array)
    save_file(tensors, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _convert(root: Path, output: Path, *extra: str) -> None:
    args = hf_to_mfq.build_parser().parse_args(
        ["--input", str(root), "--output", str(output),
         "--quant-backend", "cpu", "--device", "cpu", *extra]
    )
    hf_to_mfq.convert(args)


def _dequantized(path: Path) -> tuple[dict[str, np.ndarray], object]:
    """Return canonical float32 tensors plus the raw record table."""
    _, store = load_mmap(path)
    try:
        weights: dict[str, np.ndarray] = {}
        for name in store.records:
            if not name.startswith("model."):
                continue
            value = store[name]
            if isinstance(value, NintTensor):
                array = dequantize_flat_precision(value).reshape(value.shape)
            elif isinstance(value, MfeTensor):
                array = dequantize_expertwise(value)
            elif is_bfloat16_array(value):
                array = bfloat16_to_float32(value)
            else:
                array = np.asarray(value)
            weights[name] = np.asarray(array, np.float32)
        return weights, store.records
    finally:
        store.close()


def _run_bridge(path: Path, cache: bool) -> dict[str, np.ndarray]:
    value = os.environ.get("MFQ_NEMOTRON_H_NATIVE_TEST")
    if not value:
        pytest.skip("MFQ_NEMOTRON_H_NATIVE_TEST required")
    binary = Path(value)
    if binary.name.startswith("mfq-diagnostics"):
        pass
    else:
        binary = binary.with_name("mfq-diagnostics" + binary.suffix)
    environment = dict(os.environ)
    command = [str(binary), "--model", str(path), "--ctx-size", "32",
               "--check-nemotron-h"]
    if cache:
        # Force every routed expert resident, then run the two-stage MFE
        # pipeline instead of the direct per-projection path.
        environment.update(
            MFQ_MOE_HOST_CACHE_BYTES=str(64 << 20),
            MFQ_MOE_PIPELINE="1",
            MFQ_MOE_PRELOAD_ALL="1",
            MFQ_MOE_ASSERT_RESIDENT="1",
            MFQ_MOE_TWO_STAGE_FFN="1",
            MFQ_MOE_RAM_PCIE="1",
        )
        command[command.index("--check-nemotron-h"):command.index("--check-nemotron-h")] = [
            "--moe-gpu-cache-gb", "0.05",
        ]
    process = subprocess.run(command, text=True, capture_output=True,
                             env=environment, timeout=300)
    assert process.returncode == 0, process.stdout + process.stderr
    raw = json.loads(next(
        line.removeprefix("nemotron_h_check ")
        for line in process.stdout.splitlines()
        if line.startswith("nemotron_h_check ")
    ))
    raw.pop("architecture")
    return {
        key: np.asarray(value["data"], np.float32).reshape(value["shape"])
        for key, value in raw.items()
    }


@pytest.mark.parametrize("preset,extra,tolerance", [
    ("S5-S", ("--preset", "S5-S"), 3e-2),
    ("S6", ("--preset", "S6"), 3e-2),
    ("NINT5", ("--bits", "5"), 3e-2),
    # NVQ exercises the compact MFE decoder on the cached two-stage path.
    ("NVQ2J", ("--random-expert-mix", "nvq2j"), 2.5e-1),
    # Data-free AlphaQ allocation works on the two-projection bank.
    ("DATA-FREE", ("--nint-data-free", "--bits", "5"), 3e-2),
])
def test_quantized_text_graph_matches_dequantized_reference(tmp_path: Path, preset: str,
                                                            extra: tuple[str, ...],
                                                            tolerance: float) -> None:
    config, weights = oracle.model_fixture(GEOMETRY)
    root = tmp_path / "hf"
    _write_hf_checkpoint(root, config, weights)
    artifact = tmp_path / f"{preset}.mfq"
    _convert(root, artifact, *extra)

    dequantized, records = _dequantized(artifact)
    expected_hidden, expected_logits = oracle.model_reference(config, dequantized)

    # The routed banks carry Up and Down only, at the latent geometry.
    for layer in (1, 3):
        prefix = f"model.block.{layer}.mlp.experts"
        assert records[f"{prefix}.up.weight"].dtype == "MFE"
        assert records[f"{prefix}.down.weight"].dtype == "MFE"
        assert f"{prefix}.gate.weight" not in records
        up = dequantized[f"{prefix}.up.weight"]
        down = dequantized[f"{prefix}.down.weight"]
        assert up.shape == (oracle.EXPERTS, oracle.MOE_INTERMEDIATE, oracle.MOE_LATENT)
        assert down.shape == (oracle.EXPERTS, oracle.MOE_LATENT, oracle.MOE_INTERMEDIATE)
        # The latent width is not the hidden width.
        assert up.shape[-1] != oracle.HIDDEN

    # Recurrent controls, norms and routers stay float32.
    for name in (
        "model.block.0.linear_attention.conv.weight",
        "model.block.0.linear_attention.conv.bias",
        "model.block.0.linear_attention.a",
        "model.block.0.linear_attention.d",
        "model.block.0.linear_attention.dt_bias",
        "model.block.0.linear_attention.norm.weight",
        "model.block.0.attention.norm.weight",
        "model.block.1.mlp.router.weight",
        "model.block.1.mlp.router.bias",
        "model.output_norm.weight",
    ):
        assert records[name].dtype == "F32", name

    # Mamba keeps one fused in_projection record.
    in_projection = records["model.block.0.linear_attention.in_projection.weight"]
    assert in_projection.dtype == "NINT"
    assert dequantized["model.block.0.linear_attention.in_projection.weight"].shape == (
        oracle.MAMBA_HEADS * oracle.MAMBA_HEAD_DIM
        + oracle.MAMBA_HEADS * oracle.MAMBA_HEAD_DIM
        + 2 * oracle.MAMBA_GROUPS * oracle.SSM_STATE
        + oracle.MAMBA_HEADS,
        oracle.HIDDEN,
    )

    for cache in (False, True):
        actual = _run_bridge(artifact, cache)
        np.testing.assert_allclose(actual["full"][0], expected_logits[0],
                                   atol=tolerance, rtol=5e-2)
        np.testing.assert_allclose(actual["full_hidden"][0], expected_hidden[0],
                                   atol=tolerance, rtol=5e-2)
        # Decode rows reuse the same quantized banks and must match prefill.
        np.testing.assert_allclose(actual["decode"][0], actual["full"][0][5:],
                                   atol=tolerance, rtol=5e-2)
