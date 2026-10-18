"""GLM CUDA vision oracle and stdio tests; no MLX dependency.

Set MFQ_CUDA_RESOURCE_TEST_RUNTIME to mfq-runtime. The numerical bridge defaults
to the sibling mfq-rms-swiglu-vision-test (override MFQ_CUDA_VISION_NATIVE_TEST).
"""
import asyncio
from contextlib import suppress
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from mfq.architectures.tensor_schema import graph_spec_for_plan
from mfq.formats.assets import MODEL_CONFIG_ASSET, MODEL_GRAPH_ASSET
from mfq.formats.header import FileHeader
from mfq.formats.io import BFloat16Array, save
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import dequantize, quantize
from mfq.server.runtime.client import BackendError, StdioRuntimeClient
from mfq.server.vision.decode import _tensor, _binary_tensors
from tests.test_cuda_flash_next_runtime import glm_model_reference, mtp_fixture
from tests.test_native_nint_ple import _worker_tokenizer


def bf16(value):
    bits = np.asarray(value, np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16).view(BFloat16Array)


def vision_fixture(storage="bf16", mtp=True):
    config, weights = mtp_fixture("glm", int(mtp))
    if not mtp:
        weights = {name: value for name, value in weights.items() if not name.startswith("predictor.")}
    config.update(image_token_id=5, video_token_id=6, vision_config=dict(
        model_type="glm5_next_vision", hidden_size=8, intermediate_size=16,
        depth=2, num_heads=2, patch_size=2, temporal_patch_size=2,
        spatial_merge_size=2, out_hidden_size=32, projection_intermediate_size=48,
        rms_norm_eps=1e-5, swiglu_limit=.15, hidden_act="silu"))
    config["text_config"]["max_position_embeddings"] = 128
    rng = np.random.default_rng(5319)
    shapes = {
        "patch_embedding": (8, 3, 2, 2, 2), "downsample": (32, 8, 2, 2),
        "merger.projection": (32, 32), "merger.mlp.gate": (48, 32),
        "merger.mlp.up": (48, 32), "merger.mlp.down": (32, 48),
    }
    for i in range(2):
        shapes.update({f"block.{i}." + name: shape for name, shape in {
            "attention.qkv": (24, 8), "attention.output": (8, 8),
            "mlp.gate": (16, 8), "mlp.up": (16, 8), "mlp.down": (8, 16),
        }.items()})
    oracle = {}
    for name, shape in shapes.items():
        value = rng.normal(0, .12, shape).astype(np.float32)
        if storage == "nint8":
            packed = quantize(value.reshape(shape[0], -1), NintSpec(8, 8, 6))
            weights[f"vision.{name}.weight"] = packed
            oracle[f"vision.{name}.weight"] = dequantize(packed).reshape(shape)
        elif storage == "bf16":
            packed = bf16(value)
            weights[f"vision.{name}.weight"] = packed
            oracle[f"vision.{name}.weight"] = (packed.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            weights[f"vision.{name}.weight"] = oracle[f"vision.{name}.weight"] = value
        if not name.startswith("merger."):
            value = rng.normal(0, .02, shape[0]).astype(np.float32)
            weights[f"vision.{name}.bias"] = oracle[f"vision.{name}.bias"] = value
    for name, width in {"output_norm": 8, "merger.norm": 32, **{
        f"block.{i}.{name}": width for i in range(2) for name, width in
        (("norm1", 8), ("norm2", 8), ("attention.query_norm", 4), ("attention.key_norm", 4))
    }}.items():
        value = rng.uniform(.8, 1.2, width).astype(np.float32)
        weights[f"vision.{name}.weight"] = oracle[f"vision.{name}.weight"] = value
    weights["vision.merger.norm.bias"] = oracle["vision.merger.norm.bias"] = rng.normal(0, .02, 32).astype(np.float32)
    return config, weights, oracle


def vision_reference(config, weights, pixels, grids):
    """NumPy port of MlxGlm5NextVision: merge-major axial RoPE and frame masks."""
    c = config["vision_config"]
    h, heads, merge = c["hidden_size"], c["num_heads"], c["spatial_merge_size"]
    d = h // heads
    def linear(x, name):
        w = weights[f"vision.{name}.weight"].reshape(-1, x.shape[-1])
        return x @ w.T + weights.get(f"vision.{name}.bias", 0)
    def norm(x, name):
        return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + c["rms_norm_eps"]) * weights[f"vision.{name}.weight"]
    def swiglu(x, name):
        gate = np.minimum(linear(x, name+".gate"), c["swiglu_limit"])
        up = np.clip(linear(x, name+".up"), -c["swiglu_limit"], c["swiglu_limit"])
        return linear(gate / (1 + np.exp(-gate)) * up, name+".down")
    positions, lengths = [], []
    for t, rows, cols in grids:
        frame = [(br*merge+ir, bc*merge+ic) for br in range(rows//merge)
                 for bc in range(cols//merge) for ir in range(merge) for ic in range(merge)]
        positions.extend(frame*t)
        lengths.extend([rows*cols]*t)
    positions = np.asarray(positions, np.float32)
    angles = positions[..., None] * (10000.0 ** (-np.arange(0, d//2, 2, dtype=np.float32)/(d//2)))
    angles = np.tile(angles.reshape(len(positions), d//2), (1, 2))[:, None, :]
    def rope(x):
        a, b = np.split(x, 2, axis=-1)
        return x*np.cos(angles) + np.concatenate((-b, a), axis=-1)*np.sin(angles)
    hidden = linear(pixels, "patch_embedding")
    for i in range(c["depth"]):
        p = f"block.{i}"
        qkv = linear(norm(hidden, p+".norm1"), p+".attention.qkv").reshape(-1, 3, heads, d)
        q = rope(norm(qkv[:, 0], p+".attention.query_norm"))
        k = rope(norm(qkv[:, 1], p+".attention.key_norm"))
        attended, offset = [], 0
        for length in lengths:
            # CUDA's shared noncausal attention uses FP16 Q/K/V.
            a, b, v = [x[offset:offset+length].astype(np.float16).astype(np.float32).transpose(1, 0, 2)
                       for x in (q, k, qkv[:, 2])]
            scores = a @ b.transpose(0, 2, 1) / math.sqrt(d)
            probabilities = np.exp(scores-scores.max(-1, keepdims=True))
            probabilities /= probabilities.sum(-1, keepdims=True)
            attended.append((probabilities @ v).astype(np.float16).astype(np.float32).transpose(1, 0, 2))
            offset += length
        hidden = hidden + linear(np.concatenate(attended).reshape(-1, h), p+".attention.output")
        hidden = hidden + swiglu(norm(hidden, p+".norm2"), p+".mlp")
    grouped = norm(hidden, "output_norm").reshape(-1, merge, merge, h).transpose(0, 3, 1, 2).reshape(-1, h*merge*merge)
    downsampled = linear(grouped, "downsample")
    projected = linear(downsampled, "merger.projection")
    centered = projected - projected.mean(-1, keepdims=True)
    projected = centered / np.sqrt((centered*centered).mean(-1, keepdims=True)+1e-5)
    projected = projected*weights["vision.merger.norm.weight"] + weights["vision.merger.norm.bias"]
    projected = .5*projected*(1+np.vectorize(math.erf)(projected/math.sqrt(2)))
    return swiglu(projected.astype(np.float32), "merger.mlp")


@pytest.fixture(scope="module")
def binaries():
    runtime = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not runtime:
        pytest.skip("set MFQ_CUDA_RESOURCE_TEST_RUNTIME to a CUDA mfq-runtime")
    runtime = Path(runtime).resolve()
    bridge = Path(os.environ.get("MFQ_CUDA_VISION_NATIVE_TEST", str(runtime.parent / "cuda" / ("mfq-rms-swiglu-vision-test" + runtime.suffix))))
    probe = subprocess.run([str(bridge), "--probe"], capture_output=True, text=True, timeout=30)
    if probe.returncode == 77:
        pytest.skip("native bridge reports no CUDA device")
    assert probe.returncode == 0, probe.stdout+probe.stderr
    return runtime, bridge


def test_numpy_vision_reference_segments_and_bounded_merger():
    config, _, w = vision_fixture("float32", False)
    pixels = np.random.default_rng(4).normal(size=(24, 24)).astype(np.float32)
    together = vision_reference(config, w, pixels, [[1, 2, 4], [2, 2, 4]])
    separate = np.concatenate([vision_reference(config, w, x, [[1, 2, 4]]) for x in np.split(pixels, 3)])
    np.testing.assert_allclose(together, separate, atol=1e-6, rtol=1e-6)
    assert together.shape == (6, 32) and np.isfinite(together).all()


@pytest.mark.parametrize("storage", ["bf16", "nint8"])
@pytest.mark.parametrize("mtp", [False, True])
def test_cuda_glm_vision_embeddings_logits_and_worker(tmp_path, binaries, storage, mtp):
    runtime, bridge = binaries
    config, weights, reference_weights = vision_fixture(storage, mtp)
    graph = graph_spec_for_plan(config, list(weights)).as_dict()
    model = tmp_path / "glm-vision.mfq"
    save(model, FileHeader(version=2, model_arch="glm5_next"), {**weights,
        MODEL_CONFIG_ASSET: json.dumps(config).encode(), MODEL_GRAPH_ASSET: json.dumps(graph).encode()})
    pixels = np.random.default_rng(53).normal(size=(16, 24)).astype(np.float32)
    grids, types = [[1, 2, 4], [2, 2, 2]], [1, 2]
    # Image/video placeholders are replaced in prompt order, including video
    # frames represented by image placeholders by the real GLM processor.
    ids = [1, 5, 5, 5, 6, 2, 3]
    merged = vision_reference(config, reference_weights, pixels, grids)
    expected = weights["model.token_embedding.weight"][np.array(ids)[None]].astype(np.float16)
    expected[0, 1:5] = merged.astype(np.float16)
    request = tmp_path / "input.json"
    request.write_text(json.dumps(dict(ids=ids, pixels=pixels.ravel().tolist(), pixel_shape=list(pixels.shape), grids=grids, types=types)))
    resident_bytes = []
    for resident in (True, False):
        env = dict(os.environ, MFQ_CUDA_VISION_RESIDENT=str(int(resident)), MFQ_RUNTIME_DISABLE_PREFIX_CACHE="1")
        proc = subprocess.run([str(bridge), str(model), str(request)], capture_output=True, text=True, env=env, timeout=120)
        assert proc.returncode == 0, proc.stdout+proc.stderr
        actual = json.loads(proc.stdout.splitlines()[-1])
        def array(name):
            return np.array(actual[name]["data"], np.float32).reshape(actual[name]["shape"])
        assert actual["vision_available"] and actual["mtp_available"] == mtp
        assert actual["positions_override"] is False
        # Existing Flash-Next native oracles use 7e-3 for mixed FP16 paths.
        # Embeddings use a tighter absolute budget (1e-3, ~0.35% of the
        # largest merged value) plus 1% relative error for BF16 projection
        # rounding through the bounded merger; RTX PRO 6000 measured 5.2e-4.
        np.testing.assert_allclose(array("embeddings"), expected, atol=1e-3, rtol=1e-2)
        # Check the injected-embedding text path on the native embeddings: the
        # tiny random model amplifies the ~5e-4 BF16 vision rounding about 30x,
        # which says nothing about the text path itself.
        native_oracle = glm_model_reference(config, weights, embeddings=array("embeddings").astype(np.float16))
        np.testing.assert_allclose(array("logits"), native_oracle, atol=7e-3, rtol=7e-3)
        np.testing.assert_array_equal(array("cached_embeddings"), array("embeddings"))
        tokenizer = _worker_tokenizer(tmp_path, cacheable_tokens=True)
        async def run_worker():
            with (tmp_path / f"worker-{resident}.log").open("wb") as log:
                process = await asyncio.create_subprocess_exec(str(runtime), "--model", str(model), "--tokenizer", str(tokenizer),
                    "--transport", "stdio", "--ctx-size", "128", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, env=env)
                client = StdioRuntimeClient(process, control_timeout_seconds=30)
                try:
                    status = await client.status()
                    assert status["vision_available"] == 1 and bool(status["mtp_available"]) == mtp
                    resident_bytes.append(status["resident_weight_bytes"])
                    caps = status["model_capabilities"]
                    assert caps["features"]["image_input"] and caps["features"]["video_input"]
                    async def generate(media, enabled, prompt_ids=ids):
                        payload = dict(messages=[dict(role="user", content="image/video")], mfq_preformatted_prompt="".join(f"token{i}" for i in prompt_ids),
                            mfq_multimodal=media, enable_vision=True, enable_mtp=enabled, max_tokens=3, temperature=0, stream=True)
                        async with client.generate(payload) as events:
                            output = [event async for event in events if event]
                        metrics = next(e["mfq_metrics"] for e in output if "mfq_metrics" in e)
                        # Models without a predictor report no MTP metrics.
                        assert bool(metrics.get("mtp_used", 0)) == enabled
                    tensors = dict(pixel_values=(pixels, "float32"), vision_grid_thw=(np.array(grids, np.int32), "int32"),
                        image_grid_thw=(np.array(grids[:1], np.int32), "int32"), video_grid_thw=(np.array(grids[1:], np.int32), "int32"),
                        vision_types=(np.array(types, np.int32), "int32"))
                    media = dict(version=3, processor="grid_vision.v1", **{k: _tensor(*v) for k, v in tensors.items()})
                    image_only = dict(version=3, processor="grid_vision.v1",
                        pixel_values=_tensor(pixels[:8], "float32"),
                        vision_grid_thw=_tensor(np.array(grids[:1], np.int32), "int32"),
                        image_grid_thw=_tensor(np.array(grids[:1], np.int32), "int32"),
                        vision_types=_tensor(np.array([1], np.int32), "int32"))
                    await generate(image_only, mtp, [1, 5, 5, 2, 3])
                    await generate(media, False)
                    if mtp:
                        await generate(media, True)
                    binary, path = _binary_tensors([(k, *v) for k, v in tensors.items()])
                    try:
                        await generate(dict(version=3, processor="grid_vision.v1", **binary), mtp)
                    finally:
                        path.unlink(missing_ok=True)
                    with pytest.raises(BackendError):
                        await generate({**media, "processor": "wrong_processor"}, False)
                    await generate(media, mtp)
                finally:
                    with suppress(Exception):
                        await client.aclose()
                    if process.returncode is None:
                        process.terminate()
                    await asyncio.wait_for(process.wait(), 10)
        asyncio.run(run_worker())
    assert resident_bytes[1] < resident_bytes[0]


@pytest.mark.parametrize("storage", ["bf16", "nint8"])
def test_glm_vision_fixture_storage_and_graph(tmp_path, storage):
    from mfq.formats.io import load, is_bfloat16_array
    config, weights, oracle = vision_fixture(storage, False)
    graph = graph_spec_for_plan(config, list(weights)).as_dict()
    assert not any(c["kind"] == "predictor" for c in graph["components"])
    component = next(c for c in graph["components"] if c["kind"] == "vision")
    assert component["implementation"] == "rms_swiglu_vit"
    assert component["position_policy"] == "sequential"
    path = tmp_path / "vision.mfq"
    save(path, FileHeader(version=2, model_arch="glm5_next"), {**weights,
        MODEL_CONFIG_ASSET: json.dumps(config).encode(), MODEL_GRAPH_ASSET: json.dumps(graph).encode()})
    restored = load(path)[1]
    value = restored["vision.patch_embedding.weight"]
    if storage == "bf16":
        assert is_bfloat16_array(value)
    else:
        assert value.spec.bits == 8
    assert oracle["vision.downsample.weight"].shape == (32, 8, 2, 2)
