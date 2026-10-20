"""Exercise RAM prefix reuse and online LRU eviction through the CUDA worker."""
import asyncio
import os
from contextlib import suppress
from pathlib import Path

import pytest

from mfq.server.runtime.client import StdioRuntimeClient
from tests.test_cuda_flash_next_runtime import qwen_model_fixture, write_qwen_fixture
from tests.test_native_nint_ple import _worker_tokenizer


@pytest.mark.parametrize("disk_enabled", [False, True])
def test_ram_prefix_restore_and_pressure_eviction(tmp_path, disk_enabled):
    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("set MFQ_CUDA_RESOURCE_TEST_RUNTIME to a CUDA mfq-runtime")
    config, weights = qwen_model_fixture(ple=False)
    model = tmp_path / "tiny.mfq"
    write_qwen_fixture(model, config, weights)
    tokenizer = _worker_tokenizer(tmp_path)
    environment = dict(os.environ, MFQ_RUNTIME_DISABLE_PREFIX_CACHE="0",
        MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES="0", MFQ_RUNTIME_PREFIX_CACHE_RAM_BYTES=str(64 << 20),
        MFQ_RUNTIME_PREFIX_CACHE_DISK_BYTES=str((64 << 20) if disk_enabled else 0),
        MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS="16", MFQ_KV_TURBOQUANT_BITS="4",
        MFQ_RUNTIME_PREFIX_CACHE_DIR=str(tmp_path / "prefix"))

    async def run():
        with (tmp_path / "worker.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(str(Path(executable).resolve()),
                "--model", str(model), "--tokenizer", str(tokenizer), "--transport", "stdio",
                "--ctx-size", "128", "--yarn-context-extension", stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=log, env=environment)
            client = StdioRuntimeClient(process, control_timeout_seconds=15)
            try:
                async def generate(tokens=50):
                    prompt = "ab" * tokens
                    async with client.generate({"messages": [{"role": "user", "content": prompt}],
                        "mfq_preformatted_prompt": prompt, "temperature": 0, "max_tokens": 4,
                        "enable_mtp": False, "stream": True}) as events:
                        result = [event async for event in events if event is not None]
                    return "".join(choice.get("delta", {}).get("content", "")
                        for event in result for choice in event.get("choices", []))

                await generate(48)
                expected = await generate()
                stored = await client.status()
                assert stored["prefix_cache_hot_bytes"] == 0
                assert stored["prefix_cache_ram_bytes"] > 0
                assert stored["prefix_cache_disk_bytes"] == 0
                assert await generate() == expected
                hot = await client.status()
                assert hot["prefix_cache_ram_hits"] > 0
                assert hot["prefix_cache_hit_tokens"] > 0
                released = await client.set_cache_budget(0, ram_target_bytes=0)
                assert released["released_bytes"] > 0
                evicted = await client.status()
                assert evicted["prefix_cache_ram_bytes"] == evicted["prefix_cache_pending_bytes"] == 0
                assert bool(evicted["prefix_cache_disk_bytes"]) == disk_enabled
                assert await generate() == expected
                restored = await client.status()
                assert (restored["prefix_cache_hit_tokens"] > hot["prefix_cache_hit_tokens"]) == disk_enabled
                assert bool(restored["prefix_cache_disk_hits"]) == disk_enabled
                # Disable SSD and RAM together. No subsequent prefix survives.
                await client.set_cache_budget(0, disk_target_bytes=0, ram_target_bytes=0)
                await client.clear_cache()
                await generate()
                empty = await client.status()
                assert empty["prefix_cache_ram_bytes"] == empty["prefix_cache_disk_bytes"] == 0
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)
    asyncio.run(run())
