"""Validate the CUDA axial patch ViT against a raw NumPy reference.

Set MFQ_CUDA_DEEPSEEK_V41_VISION_TEST to the built
mfq-cuda-deepseek-v41-axial-vision-test executable. The test writes a tiny
dense-weight MFQ, runs the native tower under CUDA and compares the aligned
embedding grid with the Metal contract reimplemented in NumPy.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mfq.formats.header import FileHeader
from mfq.formats.io import save

CONFIG = {
    "hidden": 16,
    "layers": 2,
    "heads": 2,
    "intermediate": 24,
    "patch_size": 2,
    "downsample_ratio": 3,
    "output_hidden": 12,
    "theta": 10_000.0,
    "eps": 1e-6,
}
PATCH_WIDTH = 3 * CONFIG["patch_size"] ** 2
HEAD_DIM = CONFIG["hidden"] // CONFIG["heads"]


def _weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    hidden = CONFIG["hidden"]
    intermediate = CONFIG["intermediate"]
    output_hidden = CONFIG["output_hidden"]
    ratio = CONFIG["downsample_ratio"]
    result = {
        "vision.patch_embedding.weight": rng.normal(0, .1, (hidden, PATCH_WIDTH)),
        "vision.patch_embedding.bias": rng.normal(0, .05, hidden),
        "vision.output_norm.weight": np.ones(hidden),
        "vision.aligner.input.weight": rng.normal(
            0, .05, (output_hidden, hidden * ratio * ratio)
        ),
        "vision.aligner.input.bias": rng.normal(0, .05, output_hidden),
        "vision.aligner.output.weight": rng.normal(
            0, .05, (output_hidden, output_hidden)
        ),
        "vision.aligner.output.bias": rng.normal(0, .05, output_hidden),
    }
    for layer in range(CONFIG["layers"]):
        prefix = f"vision.block.{layer}"
        result.update({
            f"{prefix}.norm1.weight": np.ones(hidden),
            f"{prefix}.norm2.weight": np.ones(hidden),
            f"{prefix}.attention.qkv.weight": rng.normal(0, .1, (3 * hidden, hidden)),
            f"{prefix}.attention.qkv.bias": rng.normal(0, .05, 3 * hidden),
            f"{prefix}.attention.output.weight": rng.normal(0, .1, (hidden, hidden)),
            f"{prefix}.attention.output.bias": rng.normal(0, .05, hidden),
            f"{prefix}.mlp.gate_up.weight": rng.normal(
                0, .1, (2 * intermediate, hidden)
            ),
            f"{prefix}.mlp.down.weight": rng.normal(0, .1, (hidden, intermediate)),
        })
    return {name: value.astype(np.float32) for name, value in result.items()}


def _rms_norm(value: np.ndarray, weight: np.ndarray) -> np.ndarray:
    inverse = 1.0 / np.sqrt(np.mean(value * value, -1, keepdims=True) + CONFIG["eps"])
    return value * inverse * weight


def _linear(value: np.ndarray, weight: np.ndarray,
            bias: np.ndarray | None = None) -> np.ndarray:
    result = value @ weight.T
    return result if bias is None else result + bias


def _gelu(value: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * value * (1.0 + erf(value / np.sqrt(2.0)))


def _rope(value: np.ndarray, height: int, width: int) -> np.ndarray:
    tokens, heads, head_dim = value.shape
    rope_dim = head_dim // 2
    frequencies = rope_dim // 2
    cosine = np.zeros((tokens, rope_dim), np.float32)
    sine = np.zeros((tokens, rope_dim), np.float32)
    for row in range(height):
        for column in range(width):
            token = row * width + column
            for axis, position in enumerate((row, column)):
                for index in range(frequencies):
                    angle = position * CONFIG["theta"] ** (-2 * index / rope_dim)
                    cosine[token, axis * frequencies + index] = np.cos(angle)
                    sine[token, axis * frequencies + index] = np.sin(angle)
    first = value[..., :rope_dim]
    second = value[..., rope_dim:]
    return np.concatenate((
        first * cosine[:, None, :] - second * sine[:, None, :],
        second * cosine[:, None, :] + first * sine[:, None, :],
    ), axis=-1)


def _attention(value: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               output_weight: np.ndarray, output_bias: np.ndarray,
               height: int, width: int) -> np.ndarray:
    tokens = height * width
    qkv = _linear(value, weight, bias).reshape(
        tokens, 3, CONFIG["heads"], HEAD_DIM
    )
    query = _rope(qkv[:, 0], height, width)
    key = _rope(qkv[:, 1], height, width)
    scores = np.einsum("thd,shd->hts", query, key) / np.sqrt(HEAD_DIM)
    scores = np.exp(scores - scores.max(-1, keepdims=True))
    scores /= scores.sum(-1, keepdims=True)
    attended = np.einsum("hts,shd->thd", scores, qkv[:, 2])
    return _linear(attended.reshape(tokens, CONFIG["hidden"]), output_weight, output_bias)


def _align(value: np.ndarray, height: int, width: int,
           weights: dict[str, np.ndarray]) -> np.ndarray:
    ratio = CONFIG["downsample_ratio"]
    out_height = (height + ratio - 1) // ratio
    out_width = (width + ratio - 1) // ratio
    padding = height * width
    indices = []
    for out_row in range(out_height):
        for out_column in range(out_width):
            for patch_row in range(ratio):
                for patch_column in range(ratio):
                    row = out_row * ratio + patch_row
                    column = out_column * ratio + patch_column
                    indices.append(
                        row * width + column
                        if row < height and column < width else padding
                    )
    padded = np.concatenate(
        (value, np.zeros((1, CONFIG["hidden"]), np.float32)), axis=0
    )
    gathered = padded[np.asarray(indices)].reshape(
        out_height * out_width, ratio * ratio, CONFIG["hidden"]
    )
    gathered = gathered.transpose(0, 2, 1).reshape(
        out_height * out_width, CONFIG["hidden"] * ratio * ratio
    )
    projected = _linear(
        gathered,
        weights["vision.aligner.input.weight"],
        weights["vision.aligner.input.bias"],
    )
    return _linear(
        _gelu(projected),
        weights["vision.aligner.output.weight"],
        weights["vision.aligner.output.bias"],
    )


def _reference(patches: np.ndarray, height: int, width: int,
               weights: dict[str, np.ndarray]) -> np.ndarray:
    hidden = _linear(
        patches,
        weights["vision.patch_embedding.weight"],
        weights["vision.patch_embedding.bias"],
    ).astype(np.float32)
    for layer in range(CONFIG["layers"]):
        prefix = f"vision.block.{layer}"
        residual = hidden
        normalized = _rms_norm(hidden, weights[f"{prefix}.norm1.weight"])
        hidden = residual + _attention(
            normalized,
            weights[f"{prefix}.attention.qkv.weight"],
            weights[f"{prefix}.attention.qkv.bias"],
            weights[f"{prefix}.attention.output.weight"],
            weights[f"{prefix}.attention.output.bias"],
            height,
            width,
        )
        residual = hidden
        normalized = _rms_norm(hidden, weights[f"{prefix}.norm2.weight"])
        gate_up = _linear(normalized, weights[f"{prefix}.mlp.gate_up.weight"])
        gate, up = np.split(gate_up, 2, axis=-1)
        hidden = residual + _linear(
            gate / (1.0 + np.exp(-gate)) * up,
            weights[f"{prefix}.mlp.down.weight"],
        )
    hidden = _rms_norm(hidden, weights["vision.output_norm.weight"])
    return _align(hidden, height, width, weights)


def _run_native(executable: str, model: Path, patches: np.ndarray,
                height: int, width: int, output: Path) -> np.ndarray:
    patch_file = model.with_suffix(".patches.f32")
    patches.astype(np.float32).tofile(patch_file)
    completed = subprocess.run(
        [
            executable,
            "--model", str(model),
            "--patches", str(patch_file),
            "--output", str(output),
            "--height", str(height),
            "--width", str(width),
            "--hidden", str(CONFIG["hidden"]),
            "--layers", str(CONFIG["layers"]),
            "--heads", str(CONFIG["heads"]),
            "--intermediate", str(CONFIG["intermediate"]),
            "--patch-size", str(CONFIG["patch_size"]),
            "--downsample-ratio", str(CONFIG["downsample_ratio"]),
            "--output-hidden", str(CONFIG["output_hidden"]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 77:
        pytest.skip("CUDA device is unavailable")
    assert completed.returncode == 0, completed.stderr
    return np.fromfile(output, dtype=np.float32).reshape(
        -1, CONFIG["output_hidden"]
    )


@pytest.mark.parametrize("height,width", [(7, 5), (4, 4), (5, 8)])
def test_cuda_deepseek_v41_axial_vision_matches_numpy(
    tmp_path: Path, height: int, width: int
) -> None:
    executable = os.environ.get("MFQ_CUDA_DEEPSEEK_V41_VISION_TEST")
    if not executable:
        pytest.skip("set MFQ_CUDA_DEEPSEEK_V41_VISION_TEST to the CUDA test binary")
    rng = np.random.default_rng(7 + height * 31 + width)
    weights = _weights(rng)
    model = tmp_path / f"axial-{height}x{width}.mfq"
    save(model, FileHeader(version=2, model_arch="deepseek_v41"), weights)
    patches = rng.normal(0, .5, (height * width, PATCH_WIDTH)).astype(np.float32)
    actual = _run_native(
        executable, model, patches, height, width, tmp_path / "actual.f32"
    )
    expected = _reference(patches, height, width, weights)
    aligned = (
        (height + CONFIG["downsample_ratio"] - 1) // CONFIG["downsample_ratio"]
    ) * (
        (width + CONFIG["downsample_ratio"] - 1) // CONFIG["downsample_ratio"]
    )
    assert actual.shape == (aligned, CONFIG["output_hidden"])
    assert expected.shape == actual.shape, json.dumps(
        {"actual": actual.shape, "expected": expected.shape}
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=3e-2)
