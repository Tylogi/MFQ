"""End-to-end Nemotron-H tests through the native Metal worker.

Builds a synthetic ``.mfq`` plus a toy tokenizer, starts ``mfq-decode-metal``
over stdio and exercises text, MTP and RADIO image generation. The test skips
when the native executable or the tokenizer tooling is unavailable.
"""
from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from mfq.formats.assets import TOKENIZER_GGUF_ASSET  # noqa: E402
from mfq.formats.io import load, save  # noqa: E402

from tests import test_cuda_nemotron_runtime as oracle  # noqa: E402
from tests import test_metal_nemotron_runtime as radio  # noqa: E402


def executable() -> Path:
    value = os.environ.get("MFQ_METAL_DECODE")
    if not value:
        pytest.skip("MFQ_METAL_DECODE required")
    path = Path(value)
    if not path.is_file():
        pytest.skip(f"native Metal worker not built: {path}")
    return path


def worker_tokenizer(tmp_path: Path) -> Path:
    try:
        from tests.test_native_nint_ple import _worker_tokenizer
    except Exception as error:  # pragma: no cover - optional toolchain
        pytest.skip(f"tokenizer fixture unavailable: {error}")
    return _worker_tokenizer(tmp_path, cacheable_tokens=True)


def embed_tokenizer(path: Path, tokenizer: Path) -> None:
    header, tensors = load(path)
    tensors[TOKENIZER_GGUF_ASSET] = tokenizer.read_bytes()
    save(path, header, tensors)


def write_fixture(tmp_path: Path, mtp: bool, vision: bool) -> Path:
    if vision:
        config, weights = radio._vision_fixture()
        extras = (radio.RADIO_COMPONENT,)
    else:
        config, weights = oracle.model_fixture(oracle.LAYER_MIXERS, mtp=mtp)
        extras = (oracle.MTP_COMPONENT,) if mtp else ()
    path = tmp_path / "nemotron.mfq"
    oracle.write_nemotron_fixture(path, config, weights, extra_components=extras)
    embed_tokenizer(path, worker_tokenizer(tmp_path))
    return path


async def _generate(client, prompt: str, *, enable_mtp: bool, media=None, max_tokens: int = 5):
    payload = dict(
        messages=[dict(role="user", content="text")],
        mfq_preformatted_prompt=prompt,
        enable_mtp=enable_mtp,
        max_tokens=max_tokens,
        temperature=0,
        stream=True,
        stream_options=dict(include_usage=True),
    )
    if media is not None:
        payload.update(mfq_multimodal=media, enable_vision=True)
    async with client.generate(payload) as events:
        output = [event async for event in events if event]
    text = "".join(
        choice.get("delta", {}).get("content", "")
        for event in output for choice in event.get("choices", [])
    )
    assert any(event.get("usage", {}).get("completion_tokens", 0) > 0 for event in output), output
    return text


async def _run(path: Path, body) -> None:
    from mfq.server.runtime.client import StdioRuntimeClient

    with (path.parent / "worker.log").open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            str(executable()), "--model", str(path), "--transport", "stdio",
            "--ctx-size", "32",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log,
            env=dict(os.environ, MFQ_RUNTIME_DISABLE_PREFIX_CACHE="1"),
        )
        client = StdioRuntimeClient(process, control_timeout_seconds=30)
        try:
            await body(client)
        finally:
            with suppress(Exception):
                await client.aclose()
            if process.returncode is None:
                process.terminate()
            await asyncio.wait_for(process.wait(), 10)


def test_metal_worker_serves_nemotron_text_and_mtp(tmp_path: Path) -> None:
    path = write_fixture(tmp_path, mtp=True, vision=False)

    async def body(client) -> None:
        status = await client.status()
        assert status["mtp_available"] == 1
        plain = await _generate(client, "token8token9token10", enable_mtp=False)
        speculative = await _generate(client, "token8token9token10", enable_mtp=True)
        assert plain
        # Speculation never changes the emitted greedy tokens.
        assert speculative == plain

    asyncio.run(_run(path, body))


def test_metal_worker_serves_nemotron_radio_image(tmp_path: Path) -> None:
    from mfq.server.vision.decode import _tensor

    path = write_fixture(tmp_path, mtp=False, vision=True)
    rows, columns = radio.VISION_GRIDS[0]
    patches = rows * columns
    pixels = np.array(
        [radio._pixel_value(index) for index in range(patches * radio.VISION_PATCH_WIDTH)],
        np.float32,
    ).reshape(patches, radio.VISION_PATCH_WIDTH)
    media = dict(
        version=3,
        processor="radio_cpe.v1",
        pixel_values=_tensor(pixels, "float32"),
        vision_grid_thw=_tensor(np.array([[1, rows, columns]], np.int32), "int32"),
        image_grid_thw=_tensor(np.array([[1, rows, columns]], np.int32), "int32"),
        vision_types=_tensor(np.array([1], np.int32), "int32"),
    )
    prompt = "token8" + "token18" * (patches // 4) + "token9"

    async def body(client) -> None:
        status = await client.status()
        assert status["vision_available"] == 1
        # The toy tokenizer may decode generated ids to empty strings; the
        # completion-token assertion inside _generate proves the image prompt
        # was accepted and decoded.
        await _generate(client, prompt, enable_mtp=False, media=media)

    asyncio.run(_run(path, body))
