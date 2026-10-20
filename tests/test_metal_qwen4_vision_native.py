import asyncio
import base64
import json
import os
from contextlib import suppress
from pathlib import Path

import numpy as np
import pytest

from mfq.formats.assets import MODEL_CONFIG_ASSET, MODEL_GRAPH_ASSET
from mfq.formats.header import FileHeader
from mfq.formats.io import load, save
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import quantize
from mfq.server.runtime.client import BackendError, StdioRuntimeClient
from mfq.server.vision.decode import _binary_tensors, _tensor
from tests.test_native_nint_ple import _models, _worker_tokenizer


@pytest.mark.parametrize("quantized", [False, True])
def test_metal_qwen4_image_generation(tmp_path, quantized):
    executable = Path(os.environ.get("MFQ_METAL_VISION_RUNTIME", "build/cpp_runtime/metal/mfq-decode-metal")).resolve()
    if not executable.is_file():
        pytest.skip("build mfq-decode-metal to exercise native Metal vision")
    _, weights = load(_models(tmp_path, mtp=True, qsa=True, wide=True, context_size=4096)[1])
    config = json.loads(weights[MODEL_CONFIG_ASSET])
    config["text_config"]["rope_parameters"] = dict(
        rope_theta=1e7, mrope_section=[32, 16, 16], mrope_interleaved=True)
    config.update(image_token_id=5, video_token_id=6, vision_config=dict(
        hidden_size=8, intermediate_size=16, depth=1, num_heads=2, patch_size=2,
        temporal_patch_size=1, spatial_merge_size=2, out_hidden_size=128,
        num_position_embeddings=16, hidden_act="gelu_pytorch_tanh"))
    rng = np.random.default_rng(91)
    for name, shape in {
        "patch_embedding": (8, 3, 1, 2, 2), "position_embedding": (16, 8),
        "block.0.attention.qkv": (24, 8), "block.0.attention.output": (8, 8),
        "block.0.mlp.up": (16, 8), "block.0.mlp.down": (8, 16),
        "merger.mlp.up": (32, 32), "merger.mlp.down": (128, 32),
    }.items():
        value = rng.normal(0, .05, shape).astype(np.float32)
        weights[f"vision.{name}.weight"] = (
            quantize(value, NintSpec(4, 8, 6))
            if quantized and len(shape) == 2 and name != "position_embedding" else value)
        if name != "position_embedding":
            weights[f"vision.{name}.bias"] = np.zeros(shape[0], np.float32)
    for name in ("block.0.norm1", "block.0.norm2", "merger.norm"):
        weights[f"vision.{name}.weight"] = np.ones(8, np.float32)
        weights[f"vision.{name}.bias"] = np.zeros(8, np.float32)
    graph = dict(schema_version=1, architecture="qwen4_exp",
        canonical_naming=dict(namespace="mfq.tensor", version=1, component_roots=["model", "predictor", "vision"]),
        topology=dict(text_layers=2, predictor_layers=1, vision_layers=1),
        graph=dict(kind="causal_lm", backbone="qwen4_exp"), capabilities=["text", "vision", "mtp"],
        components=[dict(kind="text", tensor_root="model", implementation="qwen4_exp", policy="decoder"),
            dict(kind="predictor", tensor_root="predictor", implementation="next_token_prediction", policy="optional"),
            dict(kind="vision", tensor_root="vision", implementation="grid_vit", policy="optional",
                input_contract="grid_vision.v1", position_policy="grid_mrope")])
    model = tmp_path / "vision.mfq"
    save(model, FileHeader(version=2, model_arch="qwen4_exp"), {**weights,
        MODEL_CONFIG_ASSET: json.dumps(config).encode(), MODEL_GRAPH_ASSET: json.dumps(graph).encode()})
    tokenizer = _worker_tokenizer(tmp_path, cacheable_tokens=True)
    pixels = rng.normal(0, 1, (16, 12)).astype(np.float32)
    media = dict(version=3, processor="grid_vision.v1",
        pixel_values=_tensor(pixels, "float32"),
        vision_grid_thw=_tensor(np.array([[1, 4, 4]], np.int32), "int32"),
        image_grid_thw=_tensor(np.array([[1, 4, 4]], np.int32), "int32"),
        vision_types=_tensor(np.array([1], np.int32), "int32"))

    async def run():
        with (tmp_path / "worker.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(str(executable),
                "--model", str(model), "--tokenizer", str(tokenizer), "--transport", "stdio",
                "--ctx-size", "4096", "--prefill-chunk-size", "3",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log,
                env=dict(os.environ, MFQ_RUNTIME_DISABLE_PREFIX_CACHE="1"))
            client = StdioRuntimeClient(process, control_timeout_seconds=30)
            try:
                status = await client.status()
                assert status["vision_available"] == status["mtp_available"] == 1

                async def generate(mtp, image=media, prompt=None):
                    payload = dict(messages=[dict(role="user", content="image")],
                        mfq_preformatted_prompt=prompt or "token8" + "token5" * 4 + "token9" * 12,
                        enable_mtp=mtp, max_tokens=6, temperature=0, stream=True,
                        stream_options=dict(include_usage=True))
                    if image is not None:
                        payload.update(mfq_multimodal=image, enable_vision=True)
                    async with client.generate(payload) as events:
                        output = [event async for event in events if event]
                    metrics = next(event["mfq_metrics"] for event in output if "mfq_metrics" in event)
                    assert bool(metrics["mtp_used"]) == mtp
                    assert any(event.get("usage", {}).get("completion_tokens", 0) > 0 for event in output), output
                    return "".join(c.get("delta", {}).get("content", "") for event in output for c in event.get("choices", []))

                text = await generate(False, None)
                expected = [await generate(False), await generate(True)]
                with pytest.raises(BackendError):
                    await generate(False, {**media, "vision_types": _tensor(np.array([2], np.int32), "int32")})
                binary, path = _binary_tensors([
                    (name, np.frombuffer(base64.b64decode(value["data_base64"]),
                        dtype=value["dtype"]).reshape(value["shape"]), value["dtype"])
                    for name, value in media.items() if isinstance(value, dict)])
                binary.update(version=3, processor="grid_vision.v1")
                try:
                    assert await generate(False, binary) == expected[0]
                    assert await generate(True, binary) == expected[1]
                finally:
                    path.unlink()
                assert await generate(False, None) == text
                long_prompt = "token8" + "token5" * 4 + "token9" * 2060
                await generate(False, prompt=long_prompt)
                await generate(True, prompt=long_prompt)
                assert (await client.status())["resident_weight_bytes"] == status["resident_weight_bytes"]
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)

    asyncio.run(run())
