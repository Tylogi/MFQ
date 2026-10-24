"""Small synthetic CUDA workers under an explicit KV admission budget."""

import asyncio
import json
import os
from contextlib import suppress

import numpy as np
import pytest

from mfq.formats import io
from mfq.formats.assets import MODEL_CONFIG_ASSET
from mfq.server.runtime.client import StdioRuntimeClient
from tests.test_native_nint_ple import _fresh_idle_status, _models, _worker_tokenizer


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("stochastic", [False, True])
def test_cuda_kv_pressure_queues_retracts_and_preserves_serial_output(tmp_path, graph, stochastic):
    executable = os.environ.get("MFQ_ADMISSION_CUDA_RUNTIME")
    if not executable:
        pytest.skip("MFQ_ADMISSION_CUDA_RUNTIME must point to CUDA mfq-runtime")
    _, model = _models(tmp_path, qsa=True, wide=True, ple_layer=3, context_size=512, quantized_experts=True)
    header, weights = io.load(model)
    config = json.loads(weights[MODEL_CONFIG_ASSET])
    config["text_config"].update(
        num_attention_heads=2, num_key_value_heads=1, head_dim=4096, partial_rotary_factor=128 / 4096,
    )
    weights[MODEL_CONFIG_ASSET] = json.dumps(config).encode()
    rng = np.random.default_rng(8711)
    # Large KV rows cross a VMM mapping boundary within a short generation.
    for role, shape in {
        "query": (2 * 2 * 4096, 128), "key": (4096, 128),
        "value": (4096, 128), "output": (128, 2 * 4096),
    }.items():
        weights[f"model.block.1.attention.{role}.weight"] = rng.normal(0, .03, shape).astype(np.float16)
    for role in ("query_norm", "key_norm"):
        weights[f"model.block.1.attention.{role}.weight"] = np.zeros(4096, np.float16)
    weights["model.output.weight"].fill(0)
    if not stochastic:
        direction = rng.normal(0, .03, 128).astype(np.float16)
        weights["model.output.weight"][0] = direction
        weights["model.output.weight"][1] = -direction
    io.save(model, header, weights)
    tokenizer = _worker_tokenizer(tmp_path)
    environment = dict(
        os.environ,
        MFQ_INIT_NEW_TOKEN_RATIO="0.01",
        MFQ_CONTINUOUS_BATCH_KV_BUDGET_BYTES="18000000",
        MFQ_KV_ALLOCATION="dynamic",
        MFQ_CONTINUOUS_BATCH_CUDA_GRAPH=str(int(graph)),
        MFQ_RUNTIME_DISABLE_PREFIX_CACHE="1",
        MFQ_MOE_PIPELINE="1",
        MFQ_MOE_TWO_STAGE_FFN="1",
        MFQ_MOE_PRELOAD_ALL="1",
        MFQ_MOE_ASSERT_RESIDENT="1",
        MFQ_MOE_HOST_CACHE_BYTES="8388608",
    )

    async def run():
        with (tmp_path / "worker.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                executable, "--model", str(model), "--tokenizer", str(tokenizer),
                "--transport", "stdio", "--ctx-size", "512", "--continuous-batching", "3",
                "--prefill-chunk-size", "8", "--moe-gpu-cache-gb", "0.00004",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, env=environment,
            )
            client = StdioRuntimeClient(process, control_timeout_seconds=15)
            try:
                await client.status()

                async def generate(row):
                    async with client.generate({
                        "messages": [{"role": "user", "content": "ab" * (4 + row)}],
                        "temperature": .8 if stochastic else 0,
                        "top_k": 2 if stochastic else 1,
                        "seed": 1901 + row,
                        "frequency_penalty": -.01 if stochastic else 0,
                        "max_tokens": 384,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    }) as events:
                        result = [event async for event in events if event is not None]
                    assert not any("error" in event for event in result), result
                    usage = [event["usage"] for event in result if event.get("usage")]
                    assert usage and usage[-1]["completion_tokens"] == 384, result
                    return "".join(choice.get("delta", {}).get("content", "")
                                   for event in result for choice in event.get("choices", []))

                expected = [await generate(row) for row in range(3)]
                before = await _fresh_idle_status(client)
                observed = []

                async def monitor():
                    while True:
                        status = await client.status()
                        observed.append({key: status.get(key) for key in (
                            "continuous_batching_active", "continuous_batching_request_kv_bytes",
                            "kv_memory_gpu_bytes",
                            "continuous_batching_admission_deferred", "continuous_batching_retractions",
                            "continuous_batching_new_token_ratio")})
                        await asyncio.sleep(.02)

                monitoring = asyncio.create_task(monitor())
                try:
                    actual = await asyncio.gather(*(generate(row) for row in range(3)))
                finally:
                    monitoring.cancel()
                    with suppress(asyncio.CancelledError):
                        await monitoring
                after = await _fresh_idle_status(client)
                (tmp_path / "pressure.json").write_text(json.dumps({
                    "before": before, "after": after, "observed": observed,
                    "expected": expected, "actual": actual,
                }, indent=2))
                assert actual == expected
                key = "continuous_batching_admission_deferred"
                assert after[key] > before[key]
                assert after["continuous_batching_retractions"] > before["continuous_batching_retractions"]
                assert after["continuous_batching_admissions"] == 6
                assert after["continuous_batching_request_kv_bytes"] == 0
                assert after["continuous_batching_request_contexts"] == 0
                assert after["kv_memory_gpu_bytes"] == before["kv_memory_gpu_bytes"]
                assert max(row["continuous_batching_new_token_ratio"] for row in observed) > .01
                assert after["continuous_batching_new_token_ratio"] == pytest.approx(.01)
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await process.wait()

    asyncio.run(asyncio.wait_for(run(), 90))
