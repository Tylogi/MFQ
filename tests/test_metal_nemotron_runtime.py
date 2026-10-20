"""Metal Nemotron-H oracle tests against the shared NumPy reference.

The native bridge ``mfq-metal-nemotron-test`` reads a synthetic ``.mfq``
fixture and prints the same JSON documents as the CUDA diagnostics, so the
graph, session, rollback and MTP contracts are compared with the reference
implementation in ``tests/test_cuda_nemotron_runtime.py``.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from tests import test_cuda_nemotron_runtime as oracle  # noqa: E402


def bridge_binary() -> Path:
    value = os.environ.get("MFQ_METAL_NEMOTRON_TEST")
    if not value:
        pytest.skip("MFQ_METAL_NEMOTRON_TEST required")
    return Path(value)


def run_bridge(path: Path, check: str, context: int = 32) -> dict:
    process = subprocess.run(
        [str(bridge_binary()), "--model", str(path), "--ctx-size", str(context), check],
        text=True, capture_output=True, timeout=600,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    prefix = {
        "--check-nemotron-h": "nemotron_h_check ",
        "--check-nemotron-h-mtp": "nemotron_h_mtp_check ",
        "--check-nemotron-h-vision": "nemotron_h_vision_check ",
    }[check]
    raw = json.loads(next(
        line.removeprefix(prefix)
        for line in process.stdout.splitlines()
        if line.startswith(prefix)
    ))
    raw.pop("architecture")
    return raw


def parse_tensors(raw: dict) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value["data"], np.float32).reshape(value["shape"])
        for key, value in raw.items()
    }


def write_fixture(tmp_path: Path, mixers=oracle.LAYER_MIXERS, mtp: bool = False):
    config, weights = oracle.model_fixture(mixers, mtp=mtp)
    components = (oracle.MTP_COMPONENT,) if mtp else ()
    path = tmp_path / ("nemotron-mtp.mfq" if mtp else "nemotron.mfq")
    oracle.write_nemotron_fixture(path, config, weights, extra_components=components)
    return config, weights, path


@pytest.mark.parametrize("mixers", [
    oracle.LAYER_MIXERS,
    ("mamba", "mamba"),
    ("attention", "moe"),
    ("moe", "mlp", "attention"),
])
def test_metal_complete_graph(tmp_path: Path, mixers) -> None:
    config, weights, path = write_fixture(tmp_path, mixers)
    expected_hidden, expected_logits = oracle.model_reference(config, weights)
    actual = parse_tensors(run_bridge(path, "--check-nemotron-h"))
    np.testing.assert_allclose(actual["full"][0], expected_logits[0], atol=5e-3, rtol=5e-3)
    np.testing.assert_allclose(
        actual["full_hidden"][0], expected_hidden[0], atol=5e-3, rtol=5e-3)
    np.testing.assert_allclose(actual["decode"][0], actual["full"][0][5:], atol=2e-3,
                               rtol=2e-3)
    np.testing.assert_allclose(actual["chunked"][0], actual["full"][0], atol=2e-3,
                               rtol=2e-3)
    np.testing.assert_allclose(actual["restored"][0], actual["continued"][0],
                               atol=5e-3, rtol=5e-3)
    for action in ("committed", "rejected"):
        np.testing.assert_allclose(actual[action][0], actual[action + "_reference"][0],
                                   atol=5e-3, rtol=5e-3)
    for row in range(actual["batch"].shape[0]):
        np.testing.assert_allclose(actual["batch"][row], actual["full"][0], atol=2e-3,
                                   rtol=2e-3)
    np.testing.assert_array_equal(actual["reset"][0], actual["full"][0])


@pytest.mark.parametrize("error", ["groups", "state", "activation", "latent",
                                   "layers", "topk", "conv"])
def test_metal_rejects_inconsistent_config(tmp_path: Path, error: str) -> None:
    config, weights = oracle.model_fixture(("mamba", "moe"))
    llm = config["llm_config"]
    if error == "groups":
        llm["n_group"] = 2
    elif error == "state":
        llm["ssm_state_size"] = 8
    elif error == "activation":
        llm["mlp_hidden_act"] = "silu"
    elif error == "latent":
        llm.pop("moe_latent_size")
    elif error == "layers":
        llm["num_hidden_layers"] = 5
    elif error == "topk":
        llm["num_experts_per_tok"] = 4
    else:
        llm["conv_kernel"] = 0
    path = tmp_path / "invalid.mfq"
    oracle.write_nemotron_fixture(path, config, weights)
    process = subprocess.run(
        [str(bridge_binary()), "--model", str(path), "--ctx-size", "32",
         "--check-nemotron-h"],
        text=True, capture_output=True, timeout=300,
    )
    assert process.returncode != 0
    assert "nemotron_h_check " not in process.stdout


def test_metal_mtp_head_matches_reference(tmp_path: Path) -> None:
    config, weights, path = write_fixture(tmp_path, mtp=True)
    actual = run_bridge(path, "--check-nemotron-h-mtp", context=64)
    previous = np.asarray(actual["previous"]["data"], np.float32).reshape(
        actual["previous"]["shape"])
    ids = np.asarray(actual["ids"]["data"], np.float32).astype(np.int64).reshape(
        actual["ids"]["shape"])
    expected_hidden, expected_logits = oracle.mtp_reference(config, weights, previous, ids)
    np.testing.assert_allclose(actual["head"]["data"], expected_hidden.reshape(-1),
                               atol=5e-3, rtol=5e-3)
    np.testing.assert_allclose(actual["logits"]["data"], expected_logits.reshape(-1),
                               atol=5e-3, rtol=5e-3)
    np.testing.assert_allclose(actual["replayed"]["data"], actual["rollback"]["data"],
                               atol=5e-3, rtol=5e-3)
    np.testing.assert_allclose(actual["committed"]["data"],
                               actual["head"]["data"][4 * oracle.HIDDEN:7 * oracle.HIDDEN],
                               atol=5e-3, rtol=5e-3)
    first, second = oracle.mtp_chain_reference(
        config, weights, previous[:, :1], [ids[:, :1], ids[:, 1:2]])
    np.testing.assert_allclose(actual["chain_first"]["data"], first[0].reshape(-1),
                               atol=5e-3, rtol=5e-3)
    np.testing.assert_allclose(actual["chain_second"]["data"], second[0].reshape(-1),
                               atol=5e-3, rtol=5e-3)
    assert actual["greedy_mtp"] == actual["greedy_plain"]
    assert actual["cycles"] > 0
    assert actual["greedy_disabled"] == actual["greedy_plain"]


# ---------------------------------------------------------------------------
# RADIO vision oracle (NumPy, mirrors tests/test_cuda_nemotron_vision.py).
# ---------------------------------------------------------------------------

VISION_HIDDEN = 64
VISION_HEADS = 4
VISION_DEPTH = 2
VISION_PATCH = 2
VISION_SIDE = 8
VISION_CLS = 3
VISION_REGISTERS = 1
VISION_PROJECTOR_INTERMEDIATE = 128
VISION_PROJECTOR_EPS = 1e-5
VISION_LAYER_NORM_EPS = 1e-6
VISION_DOWNSAMPLE = 2
VISION_IMAGE_TOKEN = 18
VISION_GRIDS = ((4, 6), (6, 2))
VISION_PATCH_WIDTH = 3 * VISION_PATCH * VISION_PATCH
VISION_SHUFFLE_WIDTH = VISION_HIDDEN * VISION_DOWNSAMPLE * VISION_DOWNSAMPLE
RADIO_COMPONENT = dict(kind="vision", tensor_root="vision", implementation="radio",
                       policy="optional", input_contract="radio_cpe.v1")


def _vision_fixture():
    config, weights = oracle.model_fixture(("mamba", "moe"))
    config.update(
        vision_config=dict(
            model_type="radio",
            hidden_size=VISION_HIDDEN,
            num_hidden_layers=VISION_DEPTH,
            num_attention_heads=VISION_HEADS,
            mlp_ratio=4.0,
            num_channels=3,
            patch_size=VISION_PATCH,
            max_img_size=VISION_PATCH * VISION_SIDE,
            num_cls_tokens=VISION_CLS,
            num_registers=VISION_REGISTERS,
            layer_norm_eps=VISION_LAYER_NORM_EPS,
            hidden_act="gelu",
            qkv_bias=True,
            use_swiglu_ffn=False,
            layerscale_value=1.0,
        ),
        vit_hidden_size=VISION_HIDDEN,
        projector_hidden_size=VISION_PROJECTOR_INTERMEDIATE,
        downsample_ratio=0.5,
        ps_version="v2",
        img_context_token_id=VISION_IMAGE_TOKEN,
    )
    rng = np.random.default_rng(20260119)

    def rand(shape, scale=0.08):
        return rng.normal(scale=scale, size=shape).astype(np.float32)

    weights["vision.patch_embedding.weight"] = rand((VISION_HIDDEN, VISION_PATCH_WIDTH))
    weights["vision.position_embedding.weight"] = rand(
        (1, VISION_SIDE * VISION_SIDE, VISION_HIDDEN), 0.05)
    weights["vision.cls_register_token.weight"] = rand(
        (VISION_CLS + VISION_REGISTERS, VISION_HIDDEN), 0.05)
    for layer in range(VISION_DEPTH):
        prefix = f"vision.block.{layer}."
        weights[prefix + "norm1.weight"] = np.linspace(0.8, 1.2, VISION_HIDDEN, dtype=np.float32)
        weights[prefix + "norm1.bias"] = rand((VISION_HIDDEN,), 0.02)
        weights[prefix + "norm2.weight"] = np.linspace(1.1, 0.9, VISION_HIDDEN, dtype=np.float32)
        weights[prefix + "norm2.bias"] = rand((VISION_HIDDEN,), 0.02)
        weights[prefix + "attention.qkv.weight"] = rand((3 * VISION_HIDDEN, VISION_HIDDEN))
        weights[prefix + "attention.qkv.bias"] = rand((3 * VISION_HIDDEN,), 0.02)
        weights[prefix + "attention.output.weight"] = rand((VISION_HIDDEN, VISION_HIDDEN))
        weights[prefix + "attention.output.bias"] = rand((VISION_HIDDEN,), 0.02)
        weights[prefix + "mlp.up.weight"] = rand((4 * VISION_HIDDEN, VISION_HIDDEN))
        weights[prefix + "mlp.up.bias"] = rand((4 * VISION_HIDDEN,), 0.02)
        weights[prefix + "mlp.down.weight"] = rand((VISION_HIDDEN, 4 * VISION_HIDDEN))
        weights[prefix + "mlp.down.bias"] = rand((VISION_HIDDEN,), 0.02)
    weights["vision.projector.norm.weight"] = np.linspace(
        0.7, 1.3, VISION_HIDDEN, dtype=np.float32)
    weights["vision.projector.norm.bias"] = rand((VISION_HIDDEN,), 0.05)
    weights["vision.projector.mlp.norm.weight"] = np.linspace(
        0.9, 1.1, VISION_SHUFFLE_WIDTH, dtype=np.float32)
    weights["vision.projector.mlp.up.weight"] = rand(
        (VISION_PROJECTOR_INTERMEDIATE, VISION_SHUFFLE_WIDTH))
    weights["vision.projector.mlp.down.weight"] = rand(
        (VISION_HIDDEN, VISION_PROJECTOR_INTERMEDIATE))
    return config, weights


def _vision_fixture_file(tmp_path: Path, component=RADIO_COMPONENT) -> Path:
    config, weights = _vision_fixture()
    path = tmp_path / "nemotron-vision.mfq"
    oracle.write_nemotron_fixture(path, config, weights, extra_components=(component,))
    return path


def _pixel_value(index: int) -> float:
    return ((index * 37 + 11) % 101) / 64.0


def _layer_norm64(x, weight, bias, eps):
    mean = x.mean(-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * weight + bias


def _cropped_positions64(table, rows, columns):
    side = VISION_SIDE
    hidden = table.shape[-1]
    positions = table.reshape(side, side, hidden)
    square = max(rows, columns)
    axis = np.arange(square, dtype=np.float64)
    source = np.maximum(0.0, (axis + 0.5) * (side / square) - 0.5)
    low = np.floor(source).astype(np.int64)
    fraction = (source - low).astype(np.float64)
    high = np.minimum(low + 1, side - 1)
    result = np.zeros((rows, columns, hidden), np.float64)
    for row in range(rows):
        for column in range(columns):
            for vertical in range(2):
                for horizontal in range(2):
                    r = low[row] if vertical == 0 else high[row]
                    c = low[column] if horizontal == 0 else high[column]
                    weight = (1 - fraction[row] if vertical == 0 else fraction[row]) * (
                        1 - fraction[column] if horizontal == 0 else fraction[column])
                    result[row, column] += weight * positions[r, c]
    return result.reshape(rows * columns, hidden)


def _vision_block64(hidden, weights, prefix):
    tokens = hidden.shape[0]
    normalized = _layer_norm64(hidden, weights[prefix + "norm1.weight"],
                               weights[prefix + "norm1.bias"], VISION_LAYER_NORM_EPS)
    qkv = normalized @ weights[prefix + "attention.qkv.weight"].T
    qkv = qkv + weights[prefix + "attention.qkv.bias"]
    head_dim = VISION_HIDDEN // VISION_HEADS
    qkv = qkv.reshape(tokens, 3, VISION_HEADS, head_dim)
    query, key, value = (qkv[:, index].transpose(1, 0, 2)
                         for index in range(3))
    scores = np.einsum("hqd,hkd->hqk", query, key) / np.sqrt(head_dim)
    scores = scores - scores.max(-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(-1, keepdims=True)
    attended = np.einsum("hqk,hkd->hqd", probabilities, value)
    attended = attended.transpose(1, 0, 2).reshape(tokens, VISION_HIDDEN)
    attended = attended @ weights[prefix + "attention.output.weight"].T
    attended = attended + weights[prefix + "attention.output.bias"]
    hidden = hidden + attended
    normalized = _layer_norm64(hidden, weights[prefix + "norm2.weight"],
                               weights[prefix + "norm2.bias"], VISION_LAYER_NORM_EPS)
    up = normalized @ weights[prefix + "mlp.up.weight"].T + weights[prefix + "mlp.up.bias"]
    branch = 0.5 * up * (1.0 + np.vectorize(math.erf)(up / np.sqrt(2.0)))
    branch = branch @ weights[prefix + "mlp.down.weight"].T + weights[prefix + "mlp.down.bias"]
    return hidden + branch


def _pixel_shuffle64(hidden, rows, columns, version="v2"):
    width = hidden.shape[-1]
    patch = hidden.reshape(rows, columns, width)
    patch = patch.reshape(rows, columns // 2, 2 * width)
    patch = patch.transpose(1, 0, 2)
    patch = patch.reshape(columns // 2, rows // 2, 4 * width)
    if version == "v1":
        return patch.reshape(-1, 4 * width)
    patch = patch.transpose(1, 0, 2)
    return patch.reshape(-1, 4 * width)


def _radio_features_reference(weights, grids):
    converted = {name: np.asarray(value, np.float64) for name, value in weights.items()}
    pixels = np.array(
        [_pixel_value(index) for index in range(
            sum(rows * columns for rows, columns in grids) * VISION_PATCH_WIDTH)],
        np.float64).reshape(-1, VISION_PATCH_WIDTH)
    table = converted["vision.position_embedding.weight"][0]
    prefix = converted["vision.cls_register_token.weight"]
    offset = 0
    features = []
    for rows, columns in grids:
        count = rows * columns
        hidden = pixels[offset:offset + count] @ converted["vision.patch_embedding.weight"].T
        hidden = hidden + _cropped_positions64(table, rows, columns)
        hidden = np.concatenate([prefix, hidden], axis=0)
        for layer in range(VISION_DEPTH):
            hidden = _vision_block64(hidden, converted, f"vision.block.{layer}.")
        offset += count
        features.append(hidden[VISION_CLS + VISION_REGISTERS:])
    return converted, np.concatenate(features, axis=0)


def _radio_reference(weights, grids, shuffle_version="v2"):
    converted, features = _radio_features_reference(weights, grids)
    normalized = _layer_norm64(
        features, converted["vision.projector.norm.weight"],
        converted["vision.projector.norm.bias"], VISION_LAYER_NORM_EPS)
    pieces = []
    start = 0
    for rows, columns in grids:
        count = rows * columns
        pieces.append(_pixel_shuffle64(normalized[start:start + count], rows, columns,
                                       shuffle_version))
        start += count
    shuffled = np.concatenate(pieces, axis=0)
    variance = (shuffled ** 2).mean(-1, keepdims=True)
    shuffled = shuffled / np.sqrt(variance + VISION_PROJECTOR_EPS) * \
        converted["vision.projector.mlp.norm.weight"]
    hidden = np.maximum(
        shuffled @ converted["vision.projector.mlp.up.weight"].T, 0.0) ** 2
    return hidden @ converted["vision.projector.mlp.down.weight"].T


def test_metal_radio_vision_matches_reference(tmp_path: Path) -> None:
    raw = run_bridge(_vision_fixture_file(tmp_path), "--check-nemotron-h-vision")
    _, weights = _vision_fixture()
    ids = raw["ids"]
    grids = [tuple(int(item) for item in grid) for grid in raw["image_grids"]]
    assert grids == list(VISION_GRIDS)
    assert raw["placeholder_mismatch_rejected"] is True
    assert raw["depth"] == VISION_DEPTH
    assert raw["summary_tokens"] == VISION_CLS + VISION_REGISTERS
    assert raw["position_side"] == VISION_SIDE
    converted = {name: np.asarray(value, np.float64) for name, value in weights.items()}
    pixels = np.array(
        [_pixel_value(index) for index in range(
            sum(rows * columns for rows, columns in grids) * VISION_PATCH_WIDTH)],
        np.float64).reshape(-1, VISION_PATCH_WIDTH)
    rows0, columns0 = grids[0]
    first_hidden = pixels[:rows0 * columns0] @ converted["vision.patch_embedding.weight"].T
    first_hidden = first_hidden + _cropped_positions64(
        converted["vision.position_embedding.weight"][0], rows0, columns0)
    first_hidden = np.concatenate(
        [converted["vision.cls_register_token.weight"], first_hidden], axis=0)
    np.testing.assert_allclose(
        np.asarray(raw["tower_input"]["data"], np.float32).reshape(
            raw["tower_input"]["shape"])[0],
        first_hidden, atol=1e-3, rtol=2e-4)
    _, expected_features = _radio_features_reference(weights, grids)
    np.testing.assert_allclose(
        np.asarray(raw["features"]["data"], np.float32).reshape(raw["features"]["shape"]),
        expected_features, atol=1e-2, rtol=5e-3)
    expected = _radio_reference(weights, grids)
    embeddings = np.asarray(raw["embeddings"]["data"], np.float32).reshape(
        raw["embeddings"]["shape"])[0]
    positions = [index for index, value in enumerate(ids) if value == VISION_IMAGE_TOKEN]
    assert len(positions) == expected.shape[0]
    assert positions[:6] == list(range(positions[0], positions[0] + 6))
    assert positions[6:] == list(range(positions[6], positions[6] + 3))
    # Metal executes the tower in float32 while the oracle is float64. MLX's
    # Metal float32 GEMM measures ~7e-4 relative error against float64 on this
    # hardware, so the projected rows are compared with 5e-3 relative and 1e-2
    # absolute tolerance (measured worst case: 2.5e-3 relative, 0.0166 absolute)
    # instead of the CUDA 2e-4, which relies on its own accumulation order.
    np.testing.assert_allclose(embeddings[positions], expected, atol=1e-2, rtol=5e-3)
    table = np.asarray(weights["model.token_embedding.weight"], np.float32)
    text_rows = [index for index in range(len(ids)) if index not in positions]
    np.testing.assert_allclose(
        embeddings[text_rows], table[[ids[index] for index in text_rows]],
        atol=2e-2, rtol=2e-2)
    np.testing.assert_array_equal(
        np.asarray(raw["positions"]["data"]).reshape(-1), np.arange(len(ids)))


def test_metal_radio_pixel_shuffle_v1_changes_the_order(tmp_path: Path) -> None:
    raw = run_bridge(_vision_fixture_file(tmp_path), "--check-nemotron-h-vision")
    _, weights = _vision_fixture()
    grids = [tuple(int(item) for item in grid) for grid in raw["image_grids"]]
    reference = _radio_reference(weights, grids)
    transposed = _radio_reference(weights, grids, shuffle_version="v1")
    assert float(np.max(np.abs(reference - transposed))) > 1e-2
    embeddings = np.asarray(raw["embeddings"]["data"], np.float32).reshape(
        raw["embeddings"]["shape"])[0]
    positions = [index for index, value in enumerate(raw["ids"])
                 if value == VISION_IMAGE_TOKEN]
    np.testing.assert_allclose(embeddings[positions], reference, atol=1e-2, rtol=5e-3)


def test_metal_rejects_a_non_radio_vision_component(tmp_path: Path) -> None:
    component = dict(kind="vision", tensor_root="vision", implementation="grid_vit",
                     policy="optional", input_contract="grid_vision.v1",
                     position_policy="grid_mrope")
    path = _vision_fixture_file(tmp_path, component=component)
    process = subprocess.run(
        [str(bridge_binary()), "--model", str(path), "--ctx-size", "32",
         "--check-nemotron-h-vision"],
        text=True, capture_output=True, timeout=300,
    )
    assert process.returncode != 0
    assert "radio/radio_cpe.v1" in process.stderr
    assert "nemotron_h_vision_check " not in process.stdout
