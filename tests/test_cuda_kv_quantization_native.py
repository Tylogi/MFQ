"""Native CUDA TurboQuant: persisted prefixes, MTP and concurrent generation."""
import asyncio
import json
import os
from contextlib import suppress
from pathlib import Path

import numpy as np
import pytest

from mfq.server.runtime.client import StdioRuntimeClient
from tests.test_cuda_flash_next_runtime import mtp_fixture, write_qwen_fixture
from tests.test_native_nint_ple import _worker_tokenizer


def write_qwen35_fixture(path, *, vocabulary=32, context_size=256):
    from mfq.formats.io import save
    from mfq.formats.header import FileHeader
    from mfq.formats.assets import MODEL_CONFIG_ASSET, MODEL_GRAPH_ASSET

    rng = np.random.default_rng(6184)
    config = dict(model_type="qwen3_5", hidden_size=32, vocab_size=vocabulary, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        max_position_embeddings=context_size, layer_types=["full_attention"] * 2,
        rope_theta=10000., rms_norm_eps=1e-6, eos_token_id=31, tie_word_embeddings=False)
    weights = {}
    def matrix(name, shape):
        weights[name] = rng.normal(0, .08, shape).astype(np.float32)
    matrix("model.token_embedding.weight", (vocabulary, 32))
    matrix("model.output.weight", (vocabulary, 32))
    weights["model.output_norm.weight"] = np.zeros(32, np.float32)
    for i in range(2):
        prefix = f"model.block.{i}."
        for name, shape in {"attention.query": (256, 32), "attention.key": (128, 32),
            "attention.value": (128, 32), "attention.output": (32, 256),
            "mlp.gate": (64, 32), "mlp.up": (64, 32), "mlp.down": (32, 64)}.items():
            matrix(prefix + name + ".weight", shape)
        for name, size in {"attention.norm": 32, "mlp.norm": 32,
            "attention.query_norm": 64, "attention.key_norm": 64}.items():
            weights[prefix + name + ".weight"] = np.zeros(size, np.float32)
    graph = dict(schema_version=1, architecture="qwen3_5",
        canonical_naming=dict(namespace="mfq.tensor", version=1, component_roots=["model"]),
        topology=dict(text_layers=2), graph=dict(kind="causal_lm", backbone="qwen3_5"),
        components=[dict(kind="text", tensor_root="model", implementation="qwen3_5", policy="decoder")],
        capabilities=["text"])
    save(path, FileHeader(version=2, model_arch="qwen3_5"), {**weights,
        MODEL_CONFIG_ASSET: json.dumps(config).encode(), MODEL_GRAPH_ASSET: json.dumps(graph).encode()})


@pytest.mark.parametrize("family,kv_bits,combined,streamed", [("qwen4", 0, False, False), ("qwen4", 2.5, False, False), ("qwen4", 4, False, False), ("qwen4", 4, True, False), ("qwen35", 3.5, False, False), ("qwen35", 8, False, False), ("qwen4", 4, True, True)])
def test_cuda_quantized_prefix_mtp_and_batch(tmp_path, family, kv_bits, combined, streamed):
    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("set MFQ_CUDA_RESOURCE_TEST_RUNTIME to a CUDA mfq-runtime")
    model = tmp_path / "model.mfq"
    if family == "qwen4":
        config, weights = mtp_fixture("qwen", 1)
        config["text_config"]["max_position_embeddings"] = 256
        from mfq.formats.nint import NintSpec
        from mfq.formats.mfe import MfePool, MfeTensor
        from mfq.quantize.nint_quant import quantize
        for name, value in list(weights.items()):
            if name.endswith(".experts.gate_up.weight") and (not combined or name.startswith("model.")):
                for role, matrix in zip(("gate", "up"), np.split(value, 2, axis=1)):
                    weights[name.replace("gate_up", role)] = matrix.copy()
                del weights[name]
        for name, value in list(weights.items()):
            if ".experts." in name:
                experts, output, width = value.shape
                packed = quantize(value.reshape(experts * output, width), NintSpec(4, 8, 6))
                weights[name] = MfeTensor(value.shape, (MfePool(np.arange(experts, dtype=np.int32), packed),))
            elif ".mhc." in name and name.endswith((".down.weight", ".up.weight", ".inject.weight")):
                weights[name] = quantize(value, NintSpec(8, 8, 6))
            elif ".ngram.shard." in name or (".shared_expert." in name and not name.endswith(".router.weight")):
                weights[name] = quantize(value, NintSpec(4, 8, 6))
        write_qwen_fixture(model, config, weights, True)
    else:
        write_qwen35_fixture(model)
    tokenizer = _worker_tokenizer(tmp_path)
    environment = dict(os.environ, MFQ_KV_TURBOQUANT_BITS=str(kv_bits),
        MFQ_MOE_PIPELINE="1", MFQ_MOE_TWO_STAGE_FFN="1", MFQ_MOE_PRELOAD_ALL="1",
        MFQ_MOE_HOST_CACHE_BYTES="1048576", MFQ_MTP_PREDICTOR_GRAPH="1",
        MFQ_MOE_RAM_PCIE="1", MFQ_MOE_RAM_PCIE_FRACTION="1", MFQ_MOE_ROUTER_LOOKAHEAD="0",
        MFQ_MOE_PREFILL_LAYER="1" if combined else "0", MFQ_MOE_PREFILL_GRAPH="0",
        MFQ_MOE_DECODE_HOST_PIN="1", MFQ_MOE_TRANSFER_CACHE="1", MFQ_MOE_ASSERT_RESIDENT="1",
        MFQ_EMBEDDING_MAPPED="0",
        MFQ_SERVER_RESIDENT_BUDGET_BYTES="0", MFQ_RUNTIME_DISABLE_PREFIX_CACHE="0",
        MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES=str(64 << 20), MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS="16",
        # This matrix explicitly exercises SSD restore across forced process exits.
        MFQ_RUNTIME_PREFIX_CACHE_RAM_BYTES="0",
        MFQ_RUNTIME_PREFIX_CACHE_DIR=str(tmp_path / "prefix"),
        MFQ_SERVER_PREFIX_CACHE_DIR=str(tmp_path / "prefix"))
    if streamed:
        environment.update(MFQ_QSA_KV_BUDGET_BYTES="16384", MFQ_QSA_KV_RAM_BUDGET_BYTES="256")

    async def run():
        async def worker(batch, ordinal, previous=None):
            with (tmp_path / f"worker-{ordinal}.log").open("wb") as log:
                arguments = [str(Path(executable).resolve()), "--model", str(model), "--tokenizer", str(tokenizer),
                    "--transport", "stdio", "--ctx-size", "256"]
                if batch:
                    arguments += ["--continuous-batching", "2"]
                if family == "qwen4":
                    arguments += ["--moe-gpu-cache-gb", "0.000008" if combined else "0.01"]
                process = await asyncio.create_subprocess_exec(*arguments, stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=log,
                    env={**environment, "MFQ_RUNTIME_DISABLE_PREFIX_CACHE": "1" if batch else "0"})
                client = StdioRuntimeClient(process, control_timeout_seconds=15)
                try:
                    async def observed(key):
                        # Transport status is a scheduler snapshot refreshed every 250 ms.
                        for _ in range(75):
                            status = await client.status()
                            if status.get(key, 0) > 0:
                                return status[key]
                            await asyncio.sleep(.02)
                        raise AssertionError(f"{key} was never reported: {status}")

                    status = await client.status()
                    assert status["kv_quantization_supported"] == 1
                    assert status["kv_quantization_bits"] == kv_bits

                    async def generate(tokens, mtp=False):
                        if streamed:
                            tokens += 128
                        prompt = "ab" * tokens
                        async with client.generate({"messages": [{"role": "user", "content": prompt}],
                            "mfq_preformatted_prompt": prompt, "temperature": 0, "max_tokens": 12,
                            "enable_mtp": mtp, "stream": True}) as events:
                            result = [event async for event in events if event is not None]
                        assert result
                        return "".join(choice.get("delta", {}).get("content", "")
                            for event in result for choice in event.get("choices", []))

                    if batch:
                        expected = [await generate(n) for n in (50, 59)]
                        await client.clear_cache()
                        outputs = await asyncio.gather(generate(50), generate(59))
                        assert outputs == expected
                        assert await observed("continuous_batching_admissions") >= 2
                        return
                    if previous is None:
                        await generate(48)
                    expected = await generate(50)
                    if previous is not None:
                        assert expected == previous
                    status = await client.status()
                    if not streamed:
                        assert await observed("prefix_cache_hit_tokens") > 0
                    if family == "qwen4":
                        assert status["mtp_available"] == 1
                        await generate(50, True)
                        if not streamed:
                            assert await observed("mtp_predictor_graph_replays") > 0
                        else:
                            assert await observed("qsa_kv_ssd_reads") > 0
                    return expected
                finally:
                    with suppress(Exception):
                        await client.aclose()
                    if process.returncode is None:
                        process.terminate()
                    await asyncio.wait_for(process.wait(), 5)
        expected = await worker(False, 0)
        # A new process must decode the compressed SSD prefix, not reuse GPU snapshots.
        await worker(False, 1, expected)
        await worker(True, 2)
    asyncio.run(run())
