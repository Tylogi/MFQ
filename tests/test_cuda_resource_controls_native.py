"""CUDA worker integration: requires MFQ_CUDA_RESOURCE_TEST_RUNTIME."""

import asyncio
import os
from contextlib import suppress
from pathlib import Path

import pytest

from mfq.server.runtime.client import BackendError, StdioRuntimeClient
from tests.test_cuda_flash_next_runtime import qwen_model_fixture, write_qwen_fixture
from tests.test_native_nint_ple import _worker_tokenizer


@pytest.mark.parametrize("kv_bits", [0, 2, 2.5, 3, 3.5, 4, 6, 8])
def test_cuda_online_budget_yarn_and_device_telemetry(tmp_path, kv_bits):
    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("set MFQ_CUDA_RESOURCE_TEST_RUNTIME to a CUDA mfq-runtime")
    config, weights = qwen_model_fixture(ple=False)
    model = tmp_path / "tiny.mfq"
    write_qwen_fixture(model, config, weights)
    tokenizer = _worker_tokenizer(tmp_path)
    environment = dict(os.environ, MFQ_SERVER_RESIDENT_BUDGET_BYTES=str(2 << 30),
        MFQ_KV_TURBOQUANT_BITS=str(kv_bits),
        MFQ_RUNTIME_DISABLE_PREFIX_CACHE="0", MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES=str(64 << 20),
        MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS="16", MFQ_RUNTIME_TRACE_SESSION_CACHE="1",
        MFQ_RUNTIME_PREFIX_CACHE_DIR=str(tmp_path / "prefix"),
        MFQ_SERVER_PREFIX_CACHE_DIR=str(tmp_path / "prefix"))

    async def run():
        with (tmp_path / "worker.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(str(Path(executable).resolve()),
                "--model", str(model), "--tokenizer", str(tokenizer), "--transport", "stdio",
                "--ctx-size", "128", "--yarn-context-extension",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, env=environment)
            client = StdioRuntimeClient(process, control_timeout_seconds=15)
            try:
                status = await client.status()
                assert status["kv_quantization_supported"] == 1
                assert status["kv_quantization_bits"] == kv_bits
                assert status["resident_memory_dynamic_budget"] == 1
                assert status["resident_weight_dynamic_budget"] == 1
                assert status["resident_memory_budget_bytes"] == 2 << 30
                assert status["gpu_0_resident_weight_bytes"] == status["resident_weight_bytes"] > 0
                assert status["gpu_0_memory_total_bytes"] > status["gpu_0_memory_available_bytes"] >= 0
                assert status["kv_memory_preallocated"] == 1
                assert status["gpu_0_kv_bytes"] == status["kv_memory_gpu_bytes"] > 0

                async def generate(tokens=50):
                    # This prompt exceeds the fixture's native 32-token context.
                    prompt = "ab" * tokens
                    async with client.generate({"messages": [{"role": "user", "content": prompt}],
                        "mfq_preformatted_prompt": prompt, "temperature": 0, "max_tokens": 4,
                        "enable_mtp": False, "stream": True}) as events:
                        result = [event async for event in events if event is not None]
                    assert result
                    return "".join(choice.get("delta", {}).get("content", "")
                        for event in result for choice in event.get("choices", []))

                await generate(48)
                expected = await generate()
                before = await client.status()
                assert before["gpu_0_kv_bytes"] > 0
                assert before["prefix_cache_hot_blocks"] > 0
                result = await client.set_resident_memory_budget(1 << 30)
                assert result["status"] == "ok"
                assert (await client.status())["resident_memory_budget_bytes"] == 1 << 30
                assert await generate() == expected
                after = await client.status()
                assert after["prefix_cache_hit_tokens"] > 0
                with pytest.raises(BackendError):
                    await client.set_resident_memory_budget(1)
                assert (await client.status())["resident_memory_budget_bytes"] == 1 << 30
                assert await generate() == expected
                await client.set_resident_memory_budget(0, weight_target_bytes=0)
                assert (await client.status())["resident_memory_budget_bytes"] == 0
                assert await generate() == expected
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)

    asyncio.run(run())
