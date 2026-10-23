"""Shared CUDA sampler: request RNG, grammar retries and model adapters."""

import os
from pathlib import Path
import subprocess

import pytest

from tests.test_native_nint_ple import _worker_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def test_cuda_batch_sampling_is_shared_by_both_adapters():
    batching = (ROOT / "cpp_runtime/backends/cuda/engine/cuda_batching.cpp").read_text()
    decode = (ROOT / "cpp_runtime/backends/cuda/engine/cuda_batch_decode.cpp").read_text()
    runtime = ROOT / "cpp_runtime/backends/cuda/runtime/batch_sampling.cpp"
    assert (batching + decode).count("sampling_.sample(") == 2
    for adapter in ("QwenBatchOperations", "FlashNextBatchOperations"):
        sample = batching.split(f"{adapter}::sample(", 1)[1].split(f"void {adapter}::accept", 1)[0]
        assert ".item<" not in sample
        assert "sample_logits(" not in sample
    source = runtime.read_text()
    assert source.count("cudaMemcpyDeviceToHost") == 1
    assert source.count("cudaMemcpyHostToDevice") == 1
    assert "next_uniform_float()" in source


def test_cuda_batched_sampling_with_real_grammar(tmp_path):
    executable = os.environ.get("MFQ_CUDA_BATCHED_SAMPLING_TEST")
    if not executable:
        pytest.skip("set MFQ_CUDA_BATCHED_SAMPLING_TEST to mfq-cuda-runtime-state-test")
    tokenizer = _worker_tokenizer(tmp_path)
    result = subprocess.run(
        [executable, "--batched-sampling-grammar-check", str(tokenizer)],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "grammar=1" in result.stdout


@pytest.mark.parametrize("family", ["qwen35", "flash_next"])
def test_cuda_stochastic_batch_adapters_preserve_seeded_eager_generation(tmp_path, family):
    import asyncio
    from contextlib import suppress

    from mfq.formats import io
    from mfq.server.runtime.client import StdioRuntimeClient
    from tests.test_cuda_kv_quantization_native import write_qwen35_fixture
    from tests.test_native_nint_ple import _models, _fresh_idle_status

    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("set MFQ_CUDA_RESOURCE_TEST_RUNTIME to mfq-runtime")
    if family == "qwen35":
        model = tmp_path / "model.mfq"
        write_qwen35_fixture(model, context_size=512)
    else:
        _, model = _models(tmp_path, qsa=True, ple_layer=3, context_size=512, quantized_experts=True)
    header, weights = io.load(model)
    # Ties isolate sampling semantics from floating point differences in GEMMs.
    weights["model.output.weight"].fill(0)
    io.save(model, header, weights)
    tokenizer = _worker_tokenizer(tmp_path)
    environment = dict(os.environ, MFQ_MOE_PIPELINE="1", MFQ_MOE_TWO_STAGE_FFN="1",
        MFQ_MOE_PRELOAD_ALL="1", MFQ_MOE_ASSERT_RESIDENT="1", MFQ_MOE_HOST_CACHE_BYTES="8388608",
        MFQ_RUNTIME_DISABLE_PREFIX_CACHE="1",
        # Model execution is eager here; sampling graph replay is checked by
        # the native sampler test, independently of model graph allocation.
        MFQ_RUNTIME_CUDA_GRAPH="0")

    async def run():
        async def worker(batch):
            arguments = [executable, "--model", str(model), "--tokenizer", str(tokenizer),
                "--transport", "stdio", "--ctx-size", "512"]
            if family == "flash_next":
                arguments += ["--moe-gpu-cache-gb", "0.00004"]
            if batch:
                arguments += ["--continuous-batching", "8"]
            with (tmp_path / f"worker-{batch}.log").open("wb") as log:
                process = await asyncio.create_subprocess_exec(*arguments, stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=log, env=environment)
                client = StdioRuntimeClient(process, control_timeout_seconds=30)
                try:
                    await client.status()

                    async def generate(row):
                        prompt = "ab" * (4 + row)
                        async with client.generate({"messages": [{"role": "user", "content": prompt}],
                            "mfq_preformatted_prompt": prompt, "temperature": .7 + row * .1,
                            "top_k": [0, 1, 2, 8, 20, 4, 16, 3][row], "top_p": .8 if row % 2 else 1.,
                            "presence_penalty": .2 if row % 3 else 0., "frequency_penalty": -.01,
                            "repetition_penalty": 1.2, "seed": 73 + row * 173, "max_tokens": 24,
                            "enable_mtp": False, "stream": True}) as events:
                            result = [event async for event in events if event is not None]
                        text = "".join(choice.get("delta", {}).get("content", "")
                            for event in result for choice in event.get("choices", []))
                        return text

                    if batch:
                        outputs = await asyncio.gather(*(generate(row) for row in range(8)))
                        status = await _fresh_idle_status(client)
                        assert (status["continuous_batching_decode_tokens"] >
                                status["continuous_batching_decode_batches"])
                        return outputs
                    return [await generate(row) for row in range(8)]
                finally:
                    with suppress(Exception):
                        await client.aclose()
                    if process.returncode is None:
                        process.terminate()
                    await process.wait()

        reference = await worker(False)
        actual = await worker(True)
        assert actual == reference

    asyncio.run(asyncio.wait_for(run(), 180))
