"""Full-attention startup allocation and paged capacity obey load defaults."""
import asyncio
import os
from contextlib import suppress

import pytest

from mfq.server.runtime.client import StdioRuntimeClient
from tests.test_cuda_kv_quantization_native import write_qwen35_fixture
from tests.test_native_nint_ple import _worker_tokenizer


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('bits', [0, 4])
def test_vram_preallocation_changes_startup_capacity_without_changing_generation(tmp_path, batched, bits):
    executable = os.environ.get('MFQ_CUDA_RESOURCE_TEST_RUNTIME')
    if not executable:
        pytest.skip('CUDA runtime required')
    model = tmp_path / 'model.mfq'
    write_qwen35_fixture(model)
    tokenizer = _worker_tokenizer(tmp_path)

    async def run(preallocated):
        env = dict(os.environ, MFQ_KV_VRAM_PREALLOCATED=str(int(preallocated)), MFQ_KV_TURBOQUANT_BITS=str(bits),
            MFQ_RUNTIME_DISABLE_PREFIX_CACHE='1', MFQ_RUNTIME_PREFIX_CACHE_DIR=str(tmp_path / 'cache'))
        args = [executable, '--model', str(model), '--tokenizer', str(tokenizer), '--transport', 'stdio', '--ctx-size', '256']
        if batched:
            args += ['--continuous-batching', '2']
        with (tmp_path / f'{preallocated}.log').open('wb') as log:
            process = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=log, env=env)
            client = StdioRuntimeClient(process, control_timeout_seconds=15)
            try:
                before = await client.status()
                if batched:
                    assert bool(before['paged_kv_reserved_bytes']) == preallocated
                texts = []
                for length in (24, 48):
                    prompt = 'ab' * length
                    async with client.generate({'messages': [{'role': 'user', 'content': prompt}],
                        'mfq_preformatted_prompt': prompt, 'temperature': .7, 'top_p': .8, 'top_k': 20,
                        'seed': 1234, 'max_tokens': 4, 'enable_mtp': False, 'stream': True}) as events:
                        chunks = [event async for event in events if event]
                    texts.append(''.join(choice.get('delta', {}).get('content', '')
                        for event in chunks for choice in event.get('choices', [])))
                return before, texts
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)

    deferred, baseline = asyncio.run(run(False))
    reserved, actual = asyncio.run(run(True))
    assert reserved['cuda_allocated_bytes'] > deferred['cuda_allocated_bytes']
    assert actual == baseline
