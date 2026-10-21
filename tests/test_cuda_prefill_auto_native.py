"""Exercise Auto through the production stdio worker, for ordinary and MTP PP."""
import asyncio
from contextlib import suppress
import os
import re

import pytest

from mfq.server.runtime.client import StdioRuntimeClient
from tests.test_native_nint_ple import _models, _worker_tokenizer


@pytest.mark.parametrize("limit", ["auto", "2048"])
def test_native_auto_prefill_dispatches_bounded_compute_chunks(tmp_path, limit):
    executable=os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("CUDA runtime required")
    model=_models(tmp_path,mtp=True,qsa=True,context_size=32768,quantized_experts=True)[1]
    tokenizer=_worker_tokenizer(tmp_path)
    log_path=tmp_path/"worker.log"

    async def run():
        env=dict(os.environ,MFQ_KV_PLACEMENT="vram",MFQ_KV_ALLOCATION="preallocated",
            MFQ_KV_TURBOQUANT_BITS="0",MFQ_MOE_PIPELINE="1",MFQ_MOE_TWO_STAGE_FFN="1",
            MFQ_MOE_PRELOAD_ALL="1",MFQ_MOE_PREFILL_LAYER="1",MFQ_MOE_PREFILL_LAYER_ASYNC="1",
            MFQ_MOE_RAM_PCIE="1",MFQ_MOE_RAM_PCIE_FRACTION="1",MFQ_MOE_DECODE_HOST_PIN="1",
            MFQ_MOE_MAPPED_COPY="0",MFQ_MOE_TRANSFER_CACHE="1",
            MFQ_MOE_HOST_CACHE_BYTES="10485760",MFQ_RUNTIME_DISABLE_PREFIX_CACHE="1")
        with log_path.open("wb") as log:
            process=await asyncio.create_subprocess_exec(executable,"--model",str(model),
                "--tokenizer",str(tokenizer),"--transport","stdio","--ctx-size","32768",
                "--prefill-chunk-size",limit,"--moe-gpu-cache-gb","0.0009",
                stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=log,env=env)
            client=StdioRuntimeClient(process,control_timeout_seconds=60)
            try:
                await client.status()
                cases=[(4096,False),(8192,True),(16384,False)] if limit=="auto" else [(4096,True)]
                for tokens,mtp in cases:
                    payload={"messages":[{"role":"user","content":"abab"}],
                        "mfq_preformatted_prompt":"abab","mfq_benchmark_prompt_tokens":tokens,
                        "mfq_prefix_cache_enabled":False,"temperature":0,"max_tokens":4,
                        "enable_mtp":mtp,"mtp_max_draft_tokens":3,"stream":True,
                        "stream_options":{"include_usage":True}}
                    async with client.generate(payload) as events:
                        result=[event async for event in events if event]
                    assert not any(event.get("error") for event in result),result
                    assert any(event.get("usage",{}).get("prompt_tokens")==tokens for event in result),result
                await client.status()
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(),5)
        trace=log_path.read_text(errors="replace")
        plans=re.findall(r"runtime_prefill_plan mode=(\w+) tokens=(\d+) limit=(\d+) compute_chunk=(\d+)",trace)
        assert plans,trace
        expected=[("auto","4096","8192","4096"),("auto","8192","8192","8192"),
                  ("auto","16384","8192","8192")] if limit=="auto" else [("limited","4096","2048","2048")]
        assert plans==expected,trace
        assert "cleanup_error" not in trace and "grouped_fallback" not in trace,trace
    asyncio.run(run())
