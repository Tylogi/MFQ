"""Managed QSA pooling must support graph preparation before the first token."""

import asyncio
import os
from contextlib import suppress

import pytest

from mfq.formats import io
from mfq.server.runtime.client import StdioRuntimeClient
from tests.test_native_nint_ple import _models, _worker_tokenizer


@pytest.mark.parametrize("prompt_repeats", [4, 20])
def test_cuda_qsa_managed_pooling_first_request(tmp_path, prompt_repeats):
    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("set MFQ_CUDA_RESOURCE_TEST_RUNTIME to a CUDA mfq-runtime")
    _, model = _models(tmp_path, qsa=True, ple_layer=3, context_size=512, quantized_experts=True)
    header, weights = io.load(model)
    # Keep generation alive across pool boundaries without fixture EOS timing.
    weights["model.output.weight"].fill(0)
    io.save(model, header, weights)
    tokenizer = _worker_tokenizer(tmp_path)
    environment = dict(os.environ, MFQ_MOE_PIPELINE="1", MFQ_MOE_TWO_STAGE_FFN="1",
        MFQ_MOE_PRELOAD_ALL="1", MFQ_MOE_ASSERT_RESIDENT="1", MFQ_MOE_HOST_CACHE_BYTES="8388608",
        MFQ_RUNTIME_DISABLE_PREFIX_CACHE="1", MFQ_RUNTIME_CUDA_GRAPH="0")
    log_path = tmp_path / "worker.log"

    async def run():
        with log_path.open("wb") as log:
            process = await asyncio.create_subprocess_exec(executable, "--model", str(model),
                "--tokenizer", str(tokenizer), "--transport", "stdio", "--ctx-size", "512",
                "--moe-gpu-cache-gb", "0.00004", stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=log, env=environment)
            client = StdioRuntimeClient(process, control_timeout_seconds=30)
            try:
                await client.status()
                prompt = "ab" * prompt_repeats
                async with client.generate({"messages": [{"role": "user", "content": prompt}],
                    "mfq_preformatted_prompt": prompt, "temperature": 0, "max_tokens": 24,
                    "enable_mtp": False, "stream": True}) as events:
                    result = [event async for event in events if event is not None]
                text = "".join(choice.get("delta", {}).get("content", "")
                    for event in result for choice in event.get("choices", []))
                assert text == "a" * 24
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await process.wait()

    asyncio.run(asyncio.wait_for(run(), 180))
    log = log_path.read_text(encoding="utf-8")
    assert "runtime_kv_memory placement=auto allocation=preallocated" in log
