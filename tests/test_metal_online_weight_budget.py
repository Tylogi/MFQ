import asyncio
import os
import pytest

from mfq.server.runtime.client import BackendError, StdioRuntimeClient
from tests.test_metal_continuous_batching import batch_model
from tests.test_native_nint_ple import _worker_tokenizer


def test_native_online_weight_budget_preserves_streaming_kv_and_loaded_model(batch_model, tmp_path):
    binary = os.environ.get('MFQ_METAL_BUDGET_TEST')
    if not binary:
        pytest.skip('MFQ_METAL_BUDGET_TEST required')
    tokenizer = _worker_tokenizer(tmp_path)
    environment = {**os.environ, 'MFQ_QSA_KV_BUDGET_BYTES': '131072', 'MFQ_QSA_KV_TARGET_CONTEXT': '1024',
        'MFQ_SERVER_PREFIX_CACHE_HOT_BYTES': '1048576', 'MFQ_SERVER_RESIDENT_BUDGET_BYTES': str(1 << 30)}

    async def run():
        with (tmp_path / 'worker.log').open('wb') as log:
            process = await asyncio.create_subprocess_exec(binary, '--model', str(batch_model), '--tokenizer', str(tokenizer),
                '--transport', 'stdio', '--ctx-size', '1024', stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=log, env=environment)
            client = StdioRuntimeClient(process, control_timeout_seconds=30)
            try:
                await asyncio.wait_for(client.status(), 60)
                payload = {'messages': [{'role': 'user', 'content': 'ab ab ab ab'}],
                    'temperature': 0, 'max_tokens': 4, 'enable_mtp': False, 'stream': True,
                    'mfq_session_id': 'budget-session', 'mfq_prefix_cache_enabled': True}
                async def output():
                    async with client.generate(payload) as events:
                        chunks = [event async for event in events if event is not None]
                    return ''.join(chunk['choices'][0]['delta'].get('content', '') for chunk in chunks if chunk.get('choices'))
                before_output = await output()
                before = await client.status()
                assert before['resident_weight_dynamic_budget'] == 1
                assert 0 < before['mlx_cache_limit_bytes'] <= (1 << 30) // 20
                assert before['qsa_kv_offload_enabled'] == 1
                dense = int(before['resident_weight_bytes'] - before['resident_reclaimable_weight_bytes'])
                cap = dense + max(1024, int(before['resident_reclaimable_weight_bytes']) // 2)
                result = await client.set_resident_memory_budget(1 << 30, cap)
                assert result['status'] == 'ok' and process.returncode is None
                after = await client.status()
                assert after['resident_weight_budget_bytes'] == cap
                assert after['resident_weight_bytes'] <= cap
                assert after['mlx_cache_limit_bytes'] == before['mlx_cache_limit_bytes']
                assert after['qsa_kv_budget_bytes'] == before['qsa_kv_budget_bytes'] == 131072
                assert after['kv_cache_contexts'] == before['kv_cache_contexts']
                assert after['kv_cache_bytes'] == before['kv_cache_bytes']
                assert after['max_context'] == before['max_context']
                after_output = await output()
                assert after_output == before_output
                with pytest.raises(BackendError) as rejected:
                    await client.set_resident_memory_budget(1 << 30, max(1, dense - 1))
                assert rejected.value.code == 'resident_budget_exhausted'
                assert (await client.status())['resident_weight_budget_bytes'] == cap
                assert (await client.status())['mlx_cache_limit_bytes'] == before['mlx_cache_limit_bytes']
                await client.set_resident_memory_budget(1 << 30)
                assert (await client.status())['resident_weight_budget_bytes'] == cap
                await client.set_resident_memory_budget(1 << 30, 0)
                assert (await client.status())['resident_weight_budget_bytes'] == 0
                await client.set_resident_memory_budget(0, cap)
                assert (await client.status())['resident_weight_budget_bytes'] == cap
                await client.set_resident_memory_budget(0, 0)
                unlimited = await client.status()
                assert unlimited['resident_weight_budget_bytes'] == 0
                assert unlimited['mlx_cache_limit_bytes'] > before['mlx_cache_limit_bytes']
            finally:
                await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 10)
    asyncio.run(run())
