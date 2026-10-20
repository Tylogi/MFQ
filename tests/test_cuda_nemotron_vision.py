"""RADIO vision oracle test (torch reference + native ``.mfq`` fixture).

The reference implements the released graph without the runtime's fused ops:
patch projection, cropped-position interpolation, cls/register prefix, the
pre-LayerNorm ViT blocks with full attention, the ``features`` slice, and the
``vision_final_layernorm`` / pixel-shuffle-v2 / ``mlp1`` projection. The native
side is ``mfq-diagnostics --check-nemotron-h-vision`` reading a synthetic
``.mfq`` fixture. When ``MFQ_NEMOTRON_HF_DIR`` points at the released
checkpoint, the same reference is cross-checked against that checkpoint's
``modeling_radio.py`` (a small RADIO model, plus one real encoder layer).
"""

from __future__ import annotations

import importlib
import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_cuda_nemotron_runtime import (
    diagnostics_binary,
    model_fixture,
    write_nemotron_fixture,
)

HIDDEN = 64
HEADS = 4
DEPTH = 2
PATCH_SIZE = 2
POSITION_SIDE = 8
CLS_TOKENS = 3
REGISTERS = 1
PROJECTOR_INTERMEDIATE = 128
PROJECTOR_EPS = 1e-5
LAYER_NORM_EPS = 1e-6
DOWNSAMPLE_FACTOR = 2
IMAGE_TOKEN = 18
IMAGE_GRIDS = ((4, 6), (6, 2))
PATCH_WIDTH = 3 * PATCH_SIZE * PATCH_SIZE
SHUFFLE_WIDTH = HIDDEN * DOWNSAMPLE_FACTOR * DOWNSAMPLE_FACTOR
RADIO_COMPONENT = dict(
    kind="vision",
    tensor_root="vision",
    implementation="radio",
    policy="optional",
    input_contract="radio_cpe.v1",
)


def pixel_value(index: int) -> float:
    """Deterministic patch value shared with the native check (k/64 is exact)."""

    return ((index * 37 + 11) % 101) / 64.0


def vision_config() -> dict:
    return dict(
        model_type="radio",
        hidden_size=HIDDEN,
        num_hidden_layers=DEPTH,
        num_attention_heads=HEADS,
        mlp_ratio=4.0,
        num_channels=3,
        patch_size=PATCH_SIZE,
        max_img_size=PATCH_SIZE * POSITION_SIDE,
        num_cls_tokens=CLS_TOKENS,
        num_registers=REGISTERS,
        layer_norm_eps=LAYER_NORM_EPS,
        hidden_act="gelu",
        qkv_bias=True,
        use_swiglu_ffn=False,
        layerscale_value=1.0,
    )


def vision_fixture():
    config, weights = model_fixture(("mamba", "moe"))
    config.update(
        vision_config=vision_config(),
        vit_hidden_size=HIDDEN,
        projector_hidden_size=PROJECTOR_INTERMEDIATE,
        downsample_ratio=0.5,
        ps_version="v2",
        img_context_token_id=IMAGE_TOKEN,
    )
    rng = np.random.default_rng(20260119)

    def rand(shape, scale=0.08):
        return rng.normal(scale=scale, size=shape).astype(np.float32)

    weights["vision.patch_embedding.weight"] = rand((HIDDEN, PATCH_WIDTH))
    weights["vision.position_embedding.weight"] = rand(
        (1, POSITION_SIDE * POSITION_SIDE, HIDDEN), 0.05
    )
    weights["vision.cls_register_token.weight"] = rand((CLS_TOKENS + REGISTERS, HIDDEN), 0.05)
    for layer in range(DEPTH):
        prefix = f"vision.block.{layer}."
        weights[prefix + "norm1.weight"] = np.linspace(0.8, 1.2, HIDDEN).astype(np.float32)
        weights[prefix + "norm1.bias"] = rand((HIDDEN,), 0.02)
        weights[prefix + "norm2.weight"] = np.linspace(1.1, 0.9, HIDDEN).astype(np.float32)
        weights[prefix + "norm2.bias"] = rand((HIDDEN,), 0.02)
        weights[prefix + "attention.qkv.weight"] = rand((3 * HIDDEN, HIDDEN))
        weights[prefix + "attention.qkv.bias"] = rand((3 * HIDDEN,), 0.02)
        weights[prefix + "attention.output.weight"] = rand((HIDDEN, HIDDEN))
        weights[prefix + "attention.output.bias"] = rand((HIDDEN,), 0.02)
        weights[prefix + "mlp.up.weight"] = rand((4 * HIDDEN, HIDDEN))
        weights[prefix + "mlp.up.bias"] = rand((4 * HIDDEN,), 0.02)
        weights[prefix + "mlp.down.weight"] = rand((HIDDEN, 4 * HIDDEN))
        weights[prefix + "mlp.down.bias"] = rand((HIDDEN,), 0.02)
    weights["vision.projector.norm.weight"] = np.linspace(0.7, 1.3, HIDDEN).astype(np.float32)
    weights["vision.projector.norm.bias"] = rand((HIDDEN,), 0.05)
    weights["vision.projector.mlp.norm.weight"] = np.linspace(0.9, 1.1, SHUFFLE_WIDTH).astype(
        np.float32
    )
    weights["vision.projector.mlp.up.weight"] = rand((PROJECTOR_INTERMEDIATE, SHUFFLE_WIDTH))
    weights["vision.projector.mlp.down.weight"] = rand((HIDDEN, PROJECTOR_INTERMEDIATE))
    return config, weights


def vision_fixture_file(tmp_path: Path, component: dict = RADIO_COMPONENT) -> Path:
    config, weights = vision_fixture()
    path = tmp_path / "nemotron-vision.mfq"
    write_nemotron_fixture(path, config, weights, extra_components=(component,))
    return path


def run_vision_fixture(path: Path) -> dict:
    process = subprocess.run(
        [str(diagnostics_binary()), "--model", str(path), "--ctx-size", "32",
         "--check-nemotron-h-vision"],
        text=True, capture_output=True, timeout=600,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    raw = json.loads(
        next(
            line.removeprefix("nemotron_h_vision_check ")
            for line in process.stdout.splitlines()
            if line.startswith("nemotron_h_vision_check ")
        )
    )
    assert raw.pop("architecture") == "nemotron_h"
    return raw


def tensor(value: dict) -> torch.Tensor:
    return torch.tensor(value["data"], dtype=torch.float64).reshape(tuple(value["shape"]))


def as_float64(weights: dict) -> dict[str, torch.Tensor]:
    return {name: torch.tensor(value, dtype=torch.float64) for name, value in weights.items()}


def cropped_positions(table: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    """`_interpolate_position_embedding`: upsample the square table, then crop."""

    hidden = table.shape[-1]
    square = max(rows, columns)
    position = table.reshape(POSITION_SIDE, POSITION_SIDE, hidden).permute(2, 0, 1).unsqueeze(0)
    position = torch.nn.functional.interpolate(
        position, size=(square, square), mode="bilinear", align_corners=False
    )
    return position[..., :rows, :columns].flatten(2).transpose(1, 2)[0]


def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    mean = x.mean(-1, keepdim=True)
    variance = (x - mean).pow(2).mean(-1, keepdim=True)
    return (x - mean) / torch.sqrt(variance + eps) * weight + bias


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(variance + eps) * weight


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def vision_block(
    hidden: torch.Tensor,
    weights: dict,
    prefix: str,
    norm1: torch.Tensor | None = None,
    heads: int = HEADS,
) -> torch.Tensor:
    """Pre-LayerNorm block; `norm1` may be supplied already normalized."""

    batch, tokens, width = hidden.shape
    head_dim = width // heads
    normalized = (
        layer_norm(
            hidden,
            weights[prefix + "norm1.weight"],
            weights[prefix + "norm1.bias"],
            LAYER_NORM_EPS,
        )
        if norm1 is None
        else norm1
    )
    qkv = normalized @ weights[prefix + "attention.qkv.weight"].T
    qkv = qkv + weights[prefix + "attention.qkv.bias"]
    q, k, v = qkv.reshape(batch, tokens, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(head_dim)
    attended = torch.einsum("bhqk,bhkd->bhqd", torch.softmax(scores, dim=-1), v)
    attended = attended.transpose(1, 2).reshape(batch, tokens, width)
    attended = attended @ weights[prefix + "attention.output.weight"].T
    attended = attended + weights[prefix + "attention.output.bias"]
    hidden = hidden + attended
    normalized = layer_norm(
        hidden,
        weights[prefix + "norm2.weight"],
        weights[prefix + "norm2.bias"],
        LAYER_NORM_EPS,
    )
    branch = gelu(
        normalized @ weights[prefix + "mlp.up.weight"].T + weights[prefix + "mlp.up.bias"]
    )
    branch = branch @ weights[prefix + "mlp.down.weight"].T + weights[prefix + "mlp.down.bias"]
    return hidden + branch


def pixel_shuffle(hidden: torch.Tensor, rows: int, columns: int, version: str) -> torch.Tensor:
    """HF `pixel_shuffle`; v1 keeps the transposed grid, v2 restores the order."""

    width = hidden.shape[-1]
    patch = hidden.reshape(1, rows, columns, width)
    patch = patch.reshape(1, rows, columns // 2, 2 * width)
    patch = patch.permute(0, 2, 1, 3).contiguous()
    patch = patch.reshape(1, columns // 2, rows // 2, 4 * width)
    if version == "v1":
        return patch.reshape(-1, 4 * width)
    patch = patch.permute(0, 2, 1, 3).contiguous()
    return patch.reshape(-1, 4 * width)


def fixture_pixels(grids) -> torch.Tensor:
    patches = sum(rows * columns for rows, columns in grids)
    return torch.tensor(
        [pixel_value(index) for index in range(patches * PATCH_WIDTH)],
        dtype=torch.float64,
    ).reshape(patches, PATCH_WIDTH)


def radio_features_reference(converted: dict, grids) -> torch.Tensor:
    """RADIO tower output for every image, before the projector."""

    pixels = fixture_pixels(grids)
    table = converted["vision.position_embedding.weight"][0]
    prefix = converted["vision.cls_register_token.weight"]
    offset = 0
    features = []
    for rows, columns in grids:
        count = rows * columns
        hidden = pixels[offset:offset + count] @ converted["vision.patch_embedding.weight"].T
        hidden = hidden + cropped_positions(table, rows, columns)
        hidden = torch.cat([prefix.unsqueeze(0), hidden.unsqueeze(0)], dim=1)
        for layer in range(DEPTH):
            hidden = vision_block(hidden, converted, f"vision.block.{layer}.")
        offset += count
        features.append(hidden[0, CLS_TOKENS + REGISTERS:])
    return torch.cat(features, dim=0)


def radio_tower_reference(weights: dict, grids, shuffle_version: str = "v2") -> torch.Tensor:
    """Tower + projector output, i.e. the rows that replace `<image>` tokens."""

    converted = as_float64(weights)
    features = radio_features_reference(converted, grids)
    normalized = layer_norm(
        features,
        converted["vision.projector.norm.weight"],
        converted["vision.projector.norm.bias"],
        LAYER_NORM_EPS,
    )
    pieces = []
    start = 0
    for rows, columns in grids:
        count = rows * columns
        pieces.append(
            pixel_shuffle(normalized[start:start + count], rows, columns, shuffle_version)
        )
        start += count
    shuffled = torch.cat(pieces, dim=0)
    shuffled = rms_norm(shuffled, converted["vision.projector.mlp.norm.weight"], PROJECTOR_EPS)
    hidden = torch.relu(shuffled @ converted["vision.projector.mlp.up.weight"].T) ** 2
    return hidden @ converted["vision.projector.mlp.down.weight"].T


def test_native_radio_matches_reference(tmp_path) -> None:
    raw = run_vision_fixture(vision_fixture_file(tmp_path))
    _, weights = vision_fixture()
    ids = raw["ids"]
    grids = [tuple(int(item) for item in grid) for grid in raw["image_grids"]]
    assert grids == list(IMAGE_GRIDS)
    # The component must have refused a prompt whose `<image>` count is short.
    assert raw["placeholder_mismatch_rejected"] is True
    assert raw["depth"] == DEPTH
    assert raw["summary_tokens"] == CLS_TOKENS + REGISTERS
    assert raw["position_side"] == POSITION_SIDE

    expected = radio_tower_reference(weights, grids)
    embeddings = tensor(raw["embeddings"])[0]
    positions = [index for index, value in enumerate(ids) if value == IMAGE_TOKEN]
    assert len(positions) == expected.shape[0]
    # Each image owns one contiguous `<image>` run, in payload order.
    assert positions[:6] == list(range(positions[0], positions[0] + 6))
    assert positions[6:] == list(range(positions[6], positions[6] + 3))
    np.testing.assert_allclose(
        embeddings[positions].numpy(), expected.numpy(), atol=2e-4, rtol=2e-4
    )
    table = torch.tensor(weights["model.token_embedding.weight"], dtype=torch.float64)
    text_rows = [index for index in range(len(ids)) if index not in positions]
    np.testing.assert_allclose(
        embeddings[text_rows].numpy(), table[[ids[index] for index in text_rows]].numpy(),
        atol=2e-4, rtol=2e-4,
    )
    np.testing.assert_array_equal(
        tensor(raw["positions"]).reshape(-1).numpy(), np.arange(len(ids))
    )


def test_pixel_shuffle_v1_would_change_the_token_order(tmp_path) -> None:
    raw = run_vision_fixture(vision_fixture_file(tmp_path))
    _, weights = vision_fixture()
    grids = [tuple(int(item) for item in grid) for grid in raw["image_grids"]]
    reference = radio_tower_reference(weights, grids)
    transposed = radio_tower_reference(weights, grids, shuffle_version="v1")
    # v1 leaves the token grid transposed, so it must disagree by far more than
    # the numerical tolerance while v2 matches the runtime.
    assert float(torch.max(torch.abs(reference - transposed))) > 1e-2
    embeddings = tensor(raw["embeddings"])[0]
    positions = [index for index, value in enumerate(raw["ids"]) if value == IMAGE_TOKEN]
    np.testing.assert_allclose(
        embeddings[positions].numpy(), reference.numpy(), atol=2e-4, rtol=2e-4
    )


def test_native_rejects_a_non_radio_vision_component(tmp_path) -> None:
    component = dict(
        kind="vision",
        tensor_root="vision",
        implementation="grid_vit",
        policy="optional",
        input_contract="grid_vision.v1",
        position_policy="grid_mrope",
    )
    path = vision_fixture_file(tmp_path, component=component)
    process = subprocess.run(
        [str(diagnostics_binary()), "--model", str(path), "--ctx-size", "32",
         "--check-nemotron-h-vision"],
        text=True, capture_output=True, timeout=600,
    )
    assert process.returncode != 0
    assert "radio/radio_cpe.v1" in process.stderr
    assert "nemotron_h_vision_check " not in process.stdout


def _hf_directory() -> Path:
    value = os.environ.get("MFQ_NEMOTRON_HF_DIR")
    if not value:
        pytest.skip("MFQ_NEMOTRON_HF_DIR is not set")
    directory = Path(value)
    if not (directory / "modeling_radio.py").is_file():
        pytest.skip("MFQ_NEMOTRON_HF_DIR has no modeling_radio.py")
    return directory


def _hf_radio_module(directory: Path):
    """Import the checkpoint's ``modeling_radio.py`` as a package submodule.

    The released file uses relative imports, and the checkpoint directory name
    is not a valid module name, so the directory is registered as a synthetic
    package instead of being added to ``sys.path``.
    """

    package = "mfq_hf_nemotron_checkpoint"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(directory)]
        sys.modules[package] = module
    return importlib.import_module(f"{package}.modeling_radio")


def _hf_radio_config(module, vision: dict, summary_idxs=(0, 1)):
    return module.RadioConfig(
        hidden_size=vision["hidden_size"],
        num_hidden_layers=vision["num_hidden_layers"],
        num_attention_heads=vision["num_attention_heads"],
        mlp_ratio=vision["mlp_ratio"],
        num_channels=vision["num_channels"],
        patch_size=vision["patch_size"],
        max_img_size=vision["max_img_size"],
        num_cls_tokens=vision["num_cls_tokens"],
        num_registers=vision["num_registers"],
        layer_norm_eps=vision["layer_norm_eps"],
        hidden_act=vision["hidden_act"],
        qkv_bias=vision["qkv_bias"],
        use_swiglu_ffn=vision["use_swiglu_ffn"],
        layerscale_value=vision["layerscale_value"],
        summary_idxs=list(summary_idxs),
        drop_path_rate=0.0,
    )


def _hf_radio_state(weights: dict, depth: int) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        "embeddings.patch_projection.weight": torch.tensor(
            weights["vision.patch_embedding.weight"], dtype=torch.float64
        ),
        "embeddings.position_embedding": torch.tensor(
            weights["vision.position_embedding.weight"], dtype=torch.float64
        ),
        "embeddings.cls_register_token": torch.tensor(
            weights["vision.cls_register_token.weight"], dtype=torch.float64
        ),
    }
    width = state["embeddings.patch_projection.weight"].shape[0]
    for layer in range(depth):
        prefix = f"vision.block.{layer}."
        state[f"encoder.layer.{layer}.attention.attention.query.weight"] = torch.tensor(
            weights[prefix + "attention.qkv.weight"][:width], dtype=torch.float64
        )
        state[f"encoder.layer.{layer}.attention.attention.key.weight"] = torch.tensor(
            weights[prefix + "attention.qkv.weight"][width:2 * width], dtype=torch.float64
        )
        state[f"encoder.layer.{layer}.attention.attention.value.weight"] = torch.tensor(
            weights[prefix + "attention.qkv.weight"][2 * width:], dtype=torch.float64
        )
        state[f"encoder.layer.{layer}.attention.attention.query.bias"] = torch.tensor(
            weights[prefix + "attention.qkv.bias"][:width], dtype=torch.float64
        )
        state[f"encoder.layer.{layer}.attention.attention.key.bias"] = torch.tensor(
            weights[prefix + "attention.qkv.bias"][width:2 * width], dtype=torch.float64
        )
        state[f"encoder.layer.{layer}.attention.attention.value.bias"] = torch.tensor(
            weights[prefix + "attention.qkv.bias"][2 * width:], dtype=torch.float64
        )
        for canonical, hf in (
            ("attention.output.weight", "attention.output.dense.weight"),
            ("attention.output.bias", "attention.output.dense.bias"),
            ("norm1.weight", "norm1.weight"),
            ("norm1.bias", "norm1.bias"),
            ("norm2.weight", "norm2.weight"),
            ("norm2.bias", "norm2.bias"),
            ("mlp.up.weight", "mlp.fc1.weight"),
            ("mlp.up.bias", "mlp.fc1.bias"),
            ("mlp.down.weight", "mlp.fc2.weight"),
            ("mlp.down.bias", "mlp.fc2.bias"),
        ):
            state[f"encoder.layer.{layer}.{hf}"] = torch.tensor(
                weights[prefix + canonical], dtype=torch.float64
            )
    return state


def _hf_features(module, config, weights: dict, grids) -> torch.Tensor:
    model = module.RadioModel(config).to(torch.float64)
    model.make_preprocessor_external()
    missing, unexpected = model.load_state_dict(_hf_radio_state(weights, DEPTH), strict=False)
    # Only the (detached) input conditioner buffers and the identity LayerScale
    # parameters may stay at their defaults.
    assert all(
        "input_conditioner" in name or "layer_scale" in name or "summary_idxs" in name
        for name in missing
    ), missing
    assert not unexpected, unexpected
    model.eval()
    pixels = fixture_pixels(grids)
    offset = 0
    features = []
    with torch.no_grad():
        for rows, columns in grids:
            count = rows * columns
            image = pixels[offset:offset + count]
            image = image.reshape(rows, columns, 3, PATCH_SIZE, PATCH_SIZE)
            image = image.permute(2, 0, 3, 1, 4).reshape(
                1, 3, rows * PATCH_SIZE, columns * PATCH_SIZE
            )
            features.append(model(pixel_values=image).features[0])
            offset += count
    return torch.cat(features, dim=0)


def test_reference_matches_hf_radio_model(tmp_path) -> None:
    directory = _hf_directory()
    module = _hf_radio_module(directory)
    _, weights = vision_fixture()
    config = _hf_radio_config(module, vision_config())
    expected = _hf_features(module, config, weights, IMAGE_GRIDS)
    actual = radio_features_reference(as_float64(weights), IMAGE_GRIDS)
    assert expected.shape == actual.shape
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=2e-4, rtol=2e-4)


def test_reference_matches_one_real_encoder_layer() -> None:
    """Load only one real RADIO layer and compare the block math with HF."""

    from safetensors import safe_open

    directory = _hf_directory()
    module = _hf_radio_module(directory)
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "vision_model.radio_model.model.blocks.0."
    names = {
        "qkv.weight": prefix + "attn.qkv.weight",
        "qkv.bias": prefix + "attn.qkv.bias",
        "output.weight": prefix + "attn.proj.weight",
        "output.bias": prefix + "attn.proj.bias",
        "norm1.weight": prefix + "norm1.weight",
        "norm1.bias": prefix + "norm1.bias",
        "norm2.weight": prefix + "norm2.weight",
        "norm2.bias": prefix + "norm2.bias",
        "up.weight": prefix + "mlp.fc1.weight",
        "up.bias": prefix + "mlp.fc1.bias",
        "down.weight": prefix + "mlp.fc2.weight",
        "down.bias": prefix + "mlp.fc2.bias",
    }
    shards: dict[str, object] = {}
    tensors: dict[str, torch.Tensor] = {}
    for key, name in names.items():
        shard = index[name]
        if shard not in shards:
            shards[shard] = safe_open(str(directory / shard), framework="pt")
        tensors[key] = shards[shard].get_tensor(name).to(torch.float64)
    width = int(tensors["norm1.weight"].shape[0])
    assert width == 1280 and tensors["qkv.weight"].shape == (3 * width, width)
    tokens = 1 + 6 * 5
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(1, tokens, width, dtype=torch.float64, generator=generator)

    config = _hf_radio_config(
        module,
        dict(
            hidden_size=width,
            num_hidden_layers=1,
            num_attention_heads=16,
            mlp_ratio=4.0,
            num_channels=3,
            patch_size=16,
            max_img_size=2048,
            num_cls_tokens=3,
            num_registers=7,
            layer_norm_eps=1e-6,
            hidden_act="gelu",
            qkv_bias=True,
            use_swiglu_ffn=False,
            layerscale_value=1.0,
        ),
    )
    layer = module.RadioLayer(config).to(torch.float64)
    state = {
        "attention.attention.query.weight": tensors["qkv.weight"][:width],
        "attention.attention.key.weight": tensors["qkv.weight"][width:2 * width],
        "attention.attention.value.weight": tensors["qkv.weight"][2 * width:],
        "attention.attention.query.bias": tensors["qkv.bias"][:width],
        "attention.attention.key.bias": tensors["qkv.bias"][width:2 * width],
        "attention.attention.value.bias": tensors["qkv.bias"][2 * width:],
        "attention.output.dense.weight": tensors["output.weight"],
        "attention.output.dense.bias": tensors["output.bias"],
        "norm1.weight": tensors["norm1.weight"],
        "norm1.bias": tensors["norm1.bias"],
        "norm2.weight": tensors["norm2.weight"],
        "norm2.bias": tensors["norm2.bias"],
        "mlp.fc1.weight": tensors["up.weight"],
        "mlp.fc1.bias": tensors["up.bias"],
        "mlp.fc2.weight": tensors["down.weight"],
        "mlp.fc2.bias": tensors["down.bias"],
    }
    layer.load_state_dict(state, strict=False)
    layer.eval()
    with torch.no_grad():
        expected = layer(hidden)

    block_weights = {
        "vision.block.0.attention.qkv.weight": tensors["qkv.weight"],
        "vision.block.0.attention.qkv.bias": tensors["qkv.bias"],
        "vision.block.0.attention.output.weight": tensors["output.weight"],
        "vision.block.0.attention.output.bias": tensors["output.bias"],
        "vision.block.0.norm2.weight": tensors["norm2.weight"],
        "vision.block.0.norm2.bias": tensors["norm2.bias"],
        "vision.block.0.mlp.up.weight": tensors["up.weight"],
        "vision.block.0.mlp.up.bias": tensors["up.bias"],
        "vision.block.0.mlp.down.weight": tensors["down.weight"],
        "vision.block.0.mlp.down.bias": tensors["down.bias"],
    }
    normalized = layer_norm(
        hidden, tensors["norm1.weight"], tensors["norm1.bias"], LAYER_NORM_EPS
    )
    actual = vision_block(
        hidden, block_weights, "vision.block.0.", norm1=normalized, heads=16
    )
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=2e-4, rtol=2e-4)
