"""Prefill may overwrite expert tails, then decode must use restored weights."""
import asyncio
import os
import re
from contextlib import suppress

import numpy as np
import pytest

from mfq.formats.mfe import MfePool, MfeTensor
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import quantize
from mfq.server.runtime.client import StdioRuntimeClient
from tests.test_cuda_flash_next_runtime import mtp_fixture, write_qwen_fixture
from tests.test_native_nint_ple import _worker_tokenizer


@pytest.mark.parametrize('ram_hot', [False, True])
@pytest.mark.parametrize('cache_only', [False, True])
def test_prefill_loan_restores_experts_and_retains_graphs(tmp_path, ram_hot, cache_only):
    executable = os.environ.get('MFQ_CUDA_RESOURCE_TEST_RUNTIME')
    if not executable:
        pytest.skip('CUDA runtime required')
    config, weights = mtp_fixture('qwen', 1)
    config['text_config']['num_experts'] = 32
    config['text_config']['max_position_embeddings'] = 2048
    for name, value in list(weights.items()):
        if '.experts.' in name or (name.endswith(('.router.weight', '.router.bias')) and value.shape[0] == 3):
            value = value[np.arange(32) % 3].copy()
            if '.experts.' in name:
                value *= (1 + np.arange(32, dtype=np.float32) * .01).reshape((-1, 1, 1))
            weights[name] = value
        if name.endswith('.experts.gate_up.weight'):
            for role, matrix in zip(('gate', 'up'), np.split(value, 2, axis=1)):
                weights[name.replace('gate_up', role)] = matrix.copy()
            del weights[name]
    for name, value in list(weights.items()):
        if '.experts.' in name:
            experts, rows, width = value.shape
            # Different layouts must coexist in the shared decode transfer arena,
            # including across target/predictor graph replays and prefill loans.
            pools = []
            for parity, bits in enumerate((4, 5)):
                ids = np.arange(parity, experts, 2, dtype=np.int32)
                packed = quantize(value[ids].reshape(len(ids) * rows, width), NintSpec(bits, 8, 6))
                pools.append(MfePool(ids, packed))
            weights[name] = MfeTensor(value.shape, tuple(pools))
        elif '.ngram.shard.' in name:
            weights[name] = quantize(value, NintSpec(4, 8, 6))
    model = tmp_path / 'loan.mfq'
    write_qwen_fixture(model, config, weights, True)
    tokenizer = _worker_tokenizer(tmp_path)

    async def run(borrow, packed=True):
        log_path = tmp_path / f'worker-{borrow}-{packed}.log'
        env = dict(os.environ, MFQ_PREFILL_BORROW=borrow, MFQ_QWEN4_REUSE_GRAPHS='1',
            MFQ_MOE_PACKED_TRANSFER_STAGE=str(int(packed)),
            MFQ_MTP_CACHE_ONLY_PRIME=str(int(cache_only)),
            MFQ_MTP_HOST_PROMPT_HISTORY='1' if borrow == 'force' else '0',
            MFQ_KV_PLACEMENT='vram', MFQ_KV_ALLOCATION='preallocated', MFQ_KV_TURBOQUANT_BITS='0',
            MFQ_MOE_PIPELINE='1', MFQ_MOE_ROUTER_LOOKAHEAD='0', MFQ_MOE_TWO_STAGE_FFN='1',
            MFQ_MOE_PRELOAD_ALL='1', MFQ_MOE_PREFILL_LAYER='1', MFQ_MOE_PREFILL_GRAPH='0',
            MFQ_MOE_HOST_CACHE_BYTES='10485760', MFQ_RUNTIME_DISABLE_PREFIX_CACHE='1')
        if ram_hot:
            env.update(MFQ_KV_PLACEMENT='auto', MFQ_MEMORY_ARCHITECTURE='discrete',
                MFQ_KV_AUTO_STREAM_MIN_CONTEXT='64', MFQ_KV_RESIDENT_CELLS='32')
        with log_path.open('wb') as log:
            process = await asyncio.create_subprocess_exec(executable, '--model', str(model),
                '--tokenizer', str(tokenizer), '--transport', 'stdio', '--ctx-size', '2048',
                '--prefill-chunk-size', '1536', '--moe-gpu-cache-gb', '0.00048' if ram_hot else '0.0009',
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, env=env)
            client = StdioRuntimeClient(process, control_timeout_seconds=30)
            try:
                await client.status()
                captures = log_path.read_text(errors='replace').count('runtime_qwen4_graph_capture')
                outputs = []
                prompts = (('ab'*1200, False), ('ba'*1100, True),
                           ('ab'*150, False), ('ba'*150, True),
                           ('ab'*15, False), ('abab', True))
                for prompt, use_mtp in prompts:
                    async with client.generate({'messages': [{'role':'user', 'content':prompt}],
                        'mfq_preformatted_prompt':prompt, 'temperature':0, 'max_tokens':12,
                        'enable_mtp':use_mtp, 'mtp_max_draft_tokens':3, 'stream':True}) as events:
                        chunks = [event async for event in events if event]
                    assert not any('error' in event for event in chunks), chunks
                    outputs.append(''.join(choice.get('delta', {}).get('content', '')
                        for event in chunks for choice in event.get('choices', [])))
                await client.status()
                trace = log_path.read_text(errors='replace')
                assert 'moe_preload_staging_release device_bytes=0 ' in trace
                assert trace.count('runtime_qwen4_graph_capture') == captures
                assert 'cleanup_error' not in trace and 'grouped_fallback' not in trace
                if borrow == 'force':
                    # Target and predictor tiles both borrow scratch. Every
                    # scope must restore its expert image before returning.
                    loans = trace.count('runtime_prefill_loan action=borrow')
                    assert loans >= 3
                    assert trace.count('runtime_prefill_loan action=return') == loans
                    assert any(int(value)>0 for value in re.findall(r'cached_arenas=(\d+)',trace)), trace
                    returns = re.findall(r'action=return restored_bytes=(\d+) restore_bytes=(\d+) selective=(\d+)', trace)
                    assert len(returns) == loans
                    assert all(0 <= int(copied) <= int(capacity) and selective == '1'
                               for capacity, copied, selective in returns), returns
                    streamed_primes = sum(use_mtp and len(prompt) > 128 for prompt, use_mtp in prompts)
                    assert trace.count('runtime_prefill_ring split=1') == (0 if cache_only else streamed_primes)
                else:
                    assert 'runtime_prefill_loan action=borrow' not in trace
                return outputs
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)
    expected = asyncio.run(run('0', packed=False))
    assert asyncio.run(run('0')) == expected
    assert asyncio.run(run('force')) == expected
