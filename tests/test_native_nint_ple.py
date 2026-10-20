"""Matched native model-graph gate; requires the locally built Metal test."""

import asyncio
import json
import os
import subprocess
from contextlib import suppress
from pathlib import Path

import numpy as np
import pytest

from mfq.formats import io
from mfq.formats.assets import model_config_asset
from mfq.formats.header import FileHeader
from mfq.formats.nint import NintSpec, NintTensor


def _models(tmp_path: Path, *, mtp=False, qsa=False, wide=False, ple_layer=1,
            context_size=64, quantized_experts=False, vocab_size=32) -> tuple[Path, Path]:
    rng = np.random.default_rng(20261001)
    hidden, streams, experts, intermediate = 128, 2, 2, 32
    qsa = qsa or ple_layer == 3
    config = dict(
        model_type="qwen4_exp_text", vocab_size=vocab_size, hidden_size=hidden,
        num_hidden_layers=3 if ple_layer == 3 else (2 if qsa else 1), max_position_embeddings=context_size,
        num_attention_heads=24 if wide else 2, num_key_value_heads=2 if wide else 1,
        head_dim=256 if wide else 64,
        full_attention_interval=2,
        layer_types=(["linear_attention", "full_attention", "linear_attention"] if ple_layer == 3
            else (["linear_attention", "full_attention"] if qsa else ["linear_attention"])),
        hc_count=streams, hc_lowrank=4, partial_rotary_factor=0.5,
        linear_num_key_heads=1, linear_num_value_heads=2,
        linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, num_experts=experts, num_experts_per_tok=1,
        moe_intermediate_size=intermediate, shared_expert_intermediate_size=intermediate,
        indexer_n_heads=1, indexer_head_dim=128 if wide else 64, indexer_compress_ratio=2,
        indexer_budget=8 if wide else 32, indexer_kv_heads=1, ple_conv_kernel_size=4,
        ple_embed_dim=hidden, ple_layer_ids=[ple_layer], ngram_size=3,
        ngram_vocab_size_base=8, heads_per_ngram=4, split_ngram_parts=2,
        hidden_act="silu", output_gate_type="silu", rms_norm_eps=1e-6,
        mtp_num_hidden_layers=int(mtp), eos_token_id=7,
    )
    tensors = {}

    def weight(name, shape, scale=0.03):
        tensors[name] = rng.normal(scale=scale, size=shape).astype(np.float16)

    def zero(name, shape):
        tensors[name] = np.zeros(shape, dtype=np.float16)

    def mixer(root, injection):
        zero(root + ".pre.norm.weight", (streams * hidden,))
        weight(root + ".pre.down.weight", (4, streams * hidden))
        weight(root + ".pre.up.weight", (streams * hidden, 4))
        if injection:
            weight(root + ".post.inject.weight", (streams, streams * hidden))

    weight("model.token_embedding.weight", (vocab_size, hidden))
    weight("model.output.weight", (vocab_size, hidden))
    mixer("model.mhc", False)
    block = "model.block.0"
    mixer(block + ".attention.mhc", True)
    mixer(block + ".mlp.mhc", True)
    gdn = block + ".linear_attention"
    for suffix, shape in {
        "qkv": (512, hidden), "gate": (256, hidden),
        "alpha": (2, hidden), "beta": (2, hidden), "output": (hidden, 256),
    }.items():
        weight(gdn + "." + suffix + ".weight", shape)
    zero(gdn + ".dt_bias", (2,))
    zero(gdn + ".a", (2,))
    tensors[gdn + ".norm.weight"] = np.ones(128, dtype=np.float16)
    weight(gdn + ".conv.weight", (512, 1, 4))
    mlp = block + ".mlp"
    weight(mlp + ".router.weight", (experts, hidden))
    weight(mlp + ".experts.gate_up.weight", (experts, 2 * intermediate, hidden))
    weight(mlp + ".experts.down.weight", (experts, hidden, intermediate))
    weight(mlp + ".shared_expert.router.weight", (1, hidden))
    for suffix, shape in {"gate": (intermediate, hidden), "up": (intermediate, hidden),
                          "down": (hidden, intermediate)}.items():
        weight(mlp + ".shared_expert." + suffix + ".weight", shape)

    def attention(root):
        heads, kv_heads, head_dim = (config[name] for name in
            ("num_attention_heads", "num_key_value_heads", "head_dim"))
        index_dim = config["indexer_head_dim"]
        for suffix, shape in {
            "query": (2 * heads * head_dim, hidden), "key": (kv_heads * head_dim, hidden),
            "value": (kv_heads * head_dim, hidden), "output": (hidden, heads * head_dim),
            "indexer.query_key": (2 * index_dim, hidden),
        }.items():
            weight(root + "." + suffix + ".weight", shape)
        for suffix in ("query_norm", "key_norm", "indexer.query_norm", "indexer.key_norm"):
            zero(root + "." + suffix + ".weight", (index_dim if suffix.startswith("indexer") else head_dim,))

    if qsa:
        for name, value in list(tensors.items()):
            if name.startswith(mlp + ".") or name.startswith(block + ".attention.mhc."):
                tensors[name.replace(block, "model.block.1", 1)] = value.copy()
        attention("model.block.1.attention")
    if ple_layer == 3:
        for name, value in list(tensors.items()):
            if name.startswith(block + "."):
                tensors[name.replace(block, "model.block.2", 1)] = value.copy()
    if mtp:
        zero("predictor.embedding_norm.weight", (hidden,))
        zero("predictor.hidden_norm.weight", (streams * hidden,))
        weight("predictor.fusion.embedding.weight", (hidden, hidden))
        weight("predictor.fusion.hidden.weight", (hidden, hidden))
        mixer("predictor.mhc", False)
        mixer("predictor.block.0.attention.mhc", True)
        mixer("predictor.block.0.mlp.mhc", True)
        attention("predictor.block.0.attention")
        for name, value in list(tensors.items()):
            if name.startswith(mlp + ".") and ".mhc." not in name:
                tensors[name.replace(mlp, "predictor.block.0.mlp", 1)] = value.copy()
    ple = f"model.block.{ple_layer - 1}.position_embedding"
    weight(ple + ".key.weight", (streams * hidden, hidden))
    weight(ple + ".value.weight", (hidden, hidden))
    for suffix in ("key_norm", "query_norm", "conv_norm"):
        zero(ple + "." + suffix + ".weight", (streams * hidden,))
    weight(ple + ".conv.weight", (streams * hidden, 1, 4))
    ngram = ple + ".ngram"
    # Keep uint64 hash semantics, signed modulo and EOS routing in the graph.
    tensors[ngram + ".layer_multipliers"] = np.array(
        [6364136223846793005, 1442695040888963407, 3202034522624059733], dtype=np.int64)
    tensors[ngram + ".head_offsets"] = np.arange(8, dtype=np.int64) * 8
    tensors[ngram + ".head_vocab_sizes"] = np.full(8, 8, dtype=np.int64)
    fp8, nint = dict(tensors), dict(tensors)
    fp8[ngram + ".weight_scale"] = np.array([2.0], dtype=np.float16)
    for shard in range(2):
        rows, width, gs = 32, 16, (5 if shard == 0 else 7)
        groups = (width + gs - 1) // gs
        qbits = np.arange(rows, dtype=np.uint8) % 8 + 1
        kbits = np.arange(rows, dtype=np.uint8) % 4 + 5
        codes = np.zeros((rows, groups, gs), dtype=np.uint8)
        for row in range(rows):
            codes[row].reshape(-1)[:] = (np.arange(groups * gs) + row + shard) % min(4, 1 << int(qbits[row]))
        table = NintTensor(
            spec=NintSpec(4, gs, 6), shape=(rows, width), axis=0, q=codes,
            neuron_scale=np.ones(rows, dtype=np.float32),
            neuron_min=np.zeros(rows, dtype=np.float32),
            sub_scale=np.ones((rows, groups), dtype=np.uint8),
            sub_min=np.zeros((rows, groups), dtype=np.uint8), neuron_len=width,
            row_q_bits=qbits, row_sub_bits=kbits,
        )
        name = ngram + f".shard.{shard}.weight"
        nint[name] = table
        # These small integers are exactly representable, including the
        # separate FP8 scale. Thus model logits must match bit-for-bit.
        fp8[name] = np.array([0x00, 0x30, 0x38, 0x3C], dtype=np.uint8)[
            codes.reshape(rows, -1)[:, :width]].view(io.Float8E4M3Array)
    asset = model_config_asset({"model_type": "qwen4_exp", "text_config": config})
    if quantized_experts:
        from mfq.formats.mfe import MfePool, MfeTensor
        from mfq.quantize.nint_quant import quantize
        for values in (fp8, nint):
            for name, weight in list(values.items()):
                if ".mlp.experts.gate_up.weight" in name:
                    gate, up = np.split(weight, 2, axis=1)
                    values[name.replace("gate_up.weight", "gate.weight")] = gate
                    values[name.replace("gate_up.weight", "up.weight")] = up
                    del values[name]
            for name, weight in list(values.items()):
                if ".mlp.experts." not in name:
                    continue
                experts, output, width = weight.shape
                if quantized_experts == "nvq":
                    from mfq.formats.nvq import NvqSpec
                    from mfq.quantize.nvq_quant import quantize as quantize_nvq
                    values[name] = MfeTensor(shape=weight.shape, pools=tuple(
                        MfePool(expert_ids=np.array([expert], dtype=np.int32), tensor=quantize_nvq(
                            weight[expert], NvqSpec("e8_256" if expert == 0 else "d4_256", groupsize=24),
                            search_steps=3, group_chunk=128)) for expert in range(experts)))
                else:
                    packed = quantize(weight.reshape(experts * output, width), NintSpec(4, 8, 6))
                    values[name] = MfeTensor(shape=weight.shape, pools=(
                        MfePool(expert_ids=np.arange(experts, dtype=np.int32), tensor=packed),))
    paths = (tmp_path / "fp8.mfq", tmp_path / "nintv2.mfq")
    for path, values in zip(paths, (fp8, nint), strict=True):
        values[asset.name] = asset.data
        io.save(path, FileHeader(model_arch="qwen4_exp", num_tensors=len(values)), values)
    return paths


@pytest.mark.parametrize("quantized_experts", [True, "nvq"])
def test_native_cuda_flash_mtp_graph_position_buffer(tmp_path, quantized_experts):
    executable = Path(__file__).resolve().parents[1] / "build/cuda-parity/mfq-diagnostics.exe"
    if not executable.is_file():
        pytest.skip("build mfq-diagnostics to check CUDA MTP graphs")
    model = _models(tmp_path, mtp=True, qsa=True, ple_layer=3, context_size=512,
                    quantized_experts=quantized_experts)[1]
    env = dict(os.environ, MFQ_MOE_PIPELINE="1", MFQ_MOE_TWO_STAGE_FFN="1",
               MFQ_MOE_PRELOAD_ALL="1", MFQ_MOE_ASSERT_RESIDENT="1",
               MFQ_MOE_HOST_CACHE_BYTES="8388608", MFQ_TRACE_LAYER_TIMINGS="0")
    outputs = []
    for buffered in ("0", "1"):
        env["MFQ_MTP_POSITION_BUFFER"] = buffered
        result = subprocess.run([str(executable), "--model", str(model), "--ids", "1,2,3,4,5,6,7",
            "--ctx-size", "512", "--moe-gpu-cache-gb", "0.25", "--check-flash-next-mtp"],
            env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr[-3000:] + result.stdout[-1000:]
        record = next(line for line in result.stdout.splitlines() if line.startswith("flash_next_mtp_check "))
        outputs.append(json.loads(record.removeprefix("flash_next_mtp_check ")))
    assert outputs[0]["greedy"] == outputs[1]["greedy"]
    assert outputs[1]["predictor_graph_replays"] > 0


@pytest.mark.parametrize("ple_layer", [1, 3])
def test_native_qwen_ple_mixed_qk_matches_scaled_fp8_graph(tmp_path, ple_layer):
    executable = Path(__file__).resolve().parents[1] / "build/csrc/metal/mfq-metal-nint-rows-test"
    if not executable.is_file():
        pytest.skip("build mfq-metal-nint-rows-test to exercise the native model graph")
    fp8, nint = _models(tmp_path, ple_layer=ple_layer)
    result = subprocess.run([str(executable), "--qwen-ple", str(fp8), str(nint)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "batch logits passed" in result.stdout


@pytest.mark.parametrize("mtp,qsa,wide", [(False, False, False), (True, False, False),
    (False, True, False), (True, True, False), (True, True, True)])
def test_native_flash_prefix_checkpoints(tmp_path, mtp, qsa, wide):
    executable = Path(__file__).resolve().parents[1] / "build/cpp_runtime/metal/mfq-metal-nint-rows-test"
    if not executable.is_file():
        pytest.skip("build mfq-metal-nint-rows-test to exercise native prefix snapshots")
    fp8, nint = _models(tmp_path, mtp=mtp, qsa=qsa, wide=wide)
    result = subprocess.run([str(executable), "--qwen-prefix", str(fp8), str(nint)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "prefix checkpoints passed" in result.stdout


def _worker_tokenizer(tmp_path, *, cacheable_tokens=False):
    from mfq.server.runtime.hf_tokenizer import ensure_hf_tokenizer_gguf
    from tests.test_hf_native_models import _hf_fixture
    tokenizer_source = tmp_path / "tokenizer-source"
    _hf_fixture(tokenizer_source)
    config = json.loads((tokenizer_source / "config.json").read_text())
    config["text_config"]["vocab_size"] = 32
    (tokenizer_source / "config.json").write_text(json.dumps(config))
    vocabulary = json.loads((tokenizer_source / "tokenizer.json").read_text())
    vocabulary["model"]["vocab"].update({f"token{index}": index for index in range(5, 32)})
    if cacheable_tokens:
        vocabulary["added_tokens"].extend({"id": index, "content": f"token{index}", "special": False,
            "single_word": False, "lstrip": False, "rstrip": False, "normalized": False}
            for index in range(5, 32))
    (tokenizer_source / "tokenizer.json").write_text(json.dumps(vocabulary))
    return ensure_hf_tokenizer_gguf(tokenizer_source, tmp_path / "tokenizer-cache")


@pytest.mark.parametrize("quantized", [False, True])
def test_native_worker_reports_resource_breakdown_over_stdio(tmp_path, quantized):
    from mfq.server.runtime.client import StdioRuntimeClient

    executable = Path(__file__).resolve().parents[1] / "build/cpp_runtime/metal/mfq-decode-metal"
    if not executable.is_file():
        pytest.skip("build mfq-decode-metal to exercise real worker telemetry")
    models = _models(tmp_path)
    model = models[int(quantized)]
    tokenizer = _worker_tokenizer(tmp_path)

    async def run():
        with (tmp_path / "worker.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(str(executable), "--model", str(model),
                "--tokenizer", str(tokenizer), "--transport", "stdio", "--ctx-size", "64",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log)
            client = StdioRuntimeClient(process, control_timeout_seconds=10)
            try:
                status = await client.status()
                assert status["resident_weight_bytes"] > 0
                assert status["metal_wired_available"] == 1
                assert status["metal_wired_bytes"] >= status["resident_weight_bytes"]
                assert status["metal_wired_bytes"] <= status["metal_wired_limit_bytes"]
                assert status["kv_cache_bytes"] == 0
                assert status["kv_cache_contexts"] == 0
                assert status["ssd_expert_enabled"] == 0
                assert status["ssd_expert_payload_bytes"] == 0
                assert status["ssd_ple_enabled"] == 1
                with io.open_mmap(model) as store:
                    expected = sum(record.nbytes for record in store.records.values()
                        if ".position_embedding.ngram.shard." in record.name)
                assert status["ssd_ple_payload_bytes"] == expected
                repeated = await client.status()
                assert repeated["resident_weight_bytes"] == status["resident_weight_bytes"]
                assert repeated["kv_cache_bytes"] == 0
                before = repeated["ple_source_bytes_read"]
                async with client.generate({"messages": [{"role": "user", "content": "ab" * 4}],
                        "temperature": 0, "max_tokens": 2, "stream": True}) as events:
                    chunks = [event async for event in events if event is not None]
                assert chunks
                after = await client.status()
                assert after["metal_wired_bytes"] >= status["resident_weight_bytes"]
                assert after["metal_wired_limit_bytes"] == status["metal_wired_limit_bytes"]
                assert after["ple_source_bytes_read"] > before
                assert (await client.status())["ple_source_bytes_read"] == after["ple_source_bytes_read"]
                await client.reload(32)
                reloaded = await client.status()
                assert reloaded["metal_wired_available"] == 1
                assert reloaded["metal_wired_bytes"] >= reloaded["resident_weight_bytes"]
                assert reloaded["metal_wired_limit_bytes"] == status["metal_wired_limit_bytes"]
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5)

    asyncio.run(run())


@pytest.mark.parametrize("mtp,toggle,output", [(False, False, False), (True, False, False),
    (True, True, False), (True, False, True)])
def test_native_flash_short_prefix_survives_worker_restart(tmp_path, mtp, toggle, output):
    from mfq.server.runtime.client import StdioRuntimeClient

    executable = Path(__file__).resolve().parents[1] / "build/cpp_runtime/metal/mfq-decode-metal"
    if not executable.is_file():
        pytest.skip("build mfq-decode-metal to exercise persisted native prefix reuse")
    _, model = _models(tmp_path, mtp=mtp, qsa=True)
    tokenizer = _worker_tokenizer(tmp_path, cacheable_tokens=output)
    environment = dict(os.environ, MFQ_SERVER_PREFIX_CACHE_DIR=str(tmp_path / "prefix-cache"),
        MFQ_SERVER_PREFIX_CACHE_BLOCK_TOKENS="8192", MFQ_SERVER_PREFIX_CACHE_HOT_BYTES="0",
        MFQ_SERVER_TRACE_SESSION_CACHE="1")

    async def run_worker(restart, previous_output=""):
        with (tmp_path / f"worker-{restart}.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(str(executable), "--model", str(model),
                "--tokenizer", str(tokenizer), "--transport", "stdio", "--ctx-size", "64",
                env=environment, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log)
            client = StdioRuntimeClient(process, control_timeout_seconds=15)
            try:
                prompt = "ab" * (6 + 2 * restart) if not output else "ab" * 6 + previous_output + "ab" * (2 * restart)
                request = {"messages": [{"role": "user", "content": prompt}],
                    "temperature": 0, "max_tokens": 8, "enable_mtp": mtp and (not toggle or restart > 0),
                    "stream": True,
                    "stream_options": {"include_usage": True}}
                if not output:
                    request["mfq_preformatted_prompt"] = prompt
                async with client.generate(request) as events:
                    chunks = [event async for event in events if event is not None]
                status = await client.status()
                assert chunks
                assert status["prefix_cache_snapshots"] > 0, status
                assert status["prefix_cache_hot_blocks"] == 0, status
                if restart:
                    if toggle and restart == 1:
                        assert status["prefix_cache_hit_tokens"] == 0, status
                    else:
                        assert status["prefix_cache_hit_tokens"] >= 4 + 2 * restart, status
                    if output:
                        assert status["prefix_cache_hit_tokens"] > 6, status
                trace = (tmp_path / f"worker-{restart}.log").read_text()
                if not restart:
                    assert "tokens=6 blocks=1" in trace, trace
                assert "paged_codec_invalidate" not in trace
                return "".join(choice.get("delta", {}).get("content", "")
                    for event in chunks for choice in event.get("choices", []))
            finally:
                with suppress(Exception):
                    await client.aclose()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5)

    async def run():
        previous_output = await run_worker(0)
        await run_worker(1, previous_output)
        if toggle:
            await run_worker(2)

    asyncio.run(run())


def test_native_qwen_ple_rejects_mixed_fp8_nint_shards(tmp_path):
    executable = Path(__file__).resolve().parents[1] / "build/csrc/metal/mfq-metal-nint-rows-test"
    if not executable.is_file():
        pytest.skip("build mfq-metal-nint-rows-test to exercise the native model graph")
    fp8, nint = _models(tmp_path)
    header, tensors = io.load(nint)
    _, original = io.load(fp8)
    name = "model.block.0.position_embedding.ngram.shard.1.weight"
    tensors[name] = original[name]
    mixed = tmp_path / "mixed.mfq"
    io.save(mixed, header, tensors)
    result = subprocess.run([str(executable), "--qwen-ple", str(fp8), str(mixed)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "cannot mix FP8 and NINT shards" in result.stderr


def _mhc_models(tmp_path, *, adaptive, projections):
    _, source = _models(tmp_path)
    header, tensors = io.load(source)
    reference, packed = dict(tensors), dict(tensors)
    names = [name for name in tensors if ".mhc." in name and
             name.endswith((".pre.down.weight", ".pre.up.weight", ".post.inject.weight"))]
    if projections == "down":
        # Match EWQ's first failure while retaining dense companion matrices.
        names = ["model.block.0.attention.mhc.pre.down.weight"]
    for name in names:
        rows, width = tensors[name].shape
        gs = 7
        groups = (width + gs - 1) // gs
        qbits = np.arange(rows, dtype=np.uint8) % 8 + 1 if adaptive else np.full(rows, 4, dtype=np.uint8)
        kbits = np.arange(rows, dtype=np.uint8) % 4 + 5 if adaptive else np.full(rows, 6, dtype=np.uint8)
        codes = np.empty((rows, groups, gs), dtype=np.uint8)
        for row in range(rows):
            codes[row].reshape(-1)[:] = (np.arange(groups * gs) * 3 + row) % min(8, 1 << int(qbits[row]))
        table = NintTensor(
            spec=NintSpec(4, gs, 6), shape=(rows, width), axis=0, q=codes,
            neuron_scale=np.full(rows, 1 / 8192, dtype=np.float32),
            neuron_min=np.full(rows, 1 / 4096, dtype=np.float32),
            sub_scale=np.broadcast_to(np.arange(groups, dtype=np.uint8) % 7 + 1, (rows, groups)).copy(),
            sub_min=np.ones((rows, groups), dtype=np.uint8),
            neuron_len=width,
            row_q_bits=qbits if adaptive else None,
            row_sub_bits=kbits if adaptive else None,
        )
        scales = table.neuron_scale[:, None, None] * table.sub_scale[:, :, None]
        minima = table.neuron_min[:, None, None] * table.sub_min[:, :, None]
        reference[name] = ((scales * codes - minima).reshape(rows, -1)[:, :width]).astype(np.float16)
        packed[name] = table
    paths = (tmp_path / "dense_mhc.mfq", tmp_path / "nint_mhc.mfq")
    for path, values in zip(paths, (reference, packed), strict=True):
        io.save(path, header, values)
    return paths


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("projections", ["down", "all"])
def test_native_qwen_nint_mhc_matches_dense_graph(tmp_path, adaptive, projections):
    executable = Path(__file__).resolve().parents[1] / "build/csrc/metal/mfq-metal-nint-rows-test"
    if not executable.is_file():
        pytest.skip("build mfq-metal-nint-rows-test to exercise the native model graph")
    reference, packed = _mhc_models(tmp_path, adaptive=adaptive, projections=projections)
    result = subprocess.run([str(executable), "--qwen-ple", str(reference), str(packed)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "batch logits passed" in result.stdout


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("group_sizes,decode_only", [
    ((48, 24, 7), False), ((28, 7, 24), False),
    ((24, 28, 48), True), ((12, 12, 12), True), ((48, 48, 28), True),
])
def test_native_qwen_packed_mhc_real_geometry(tmp_path, adaptive, group_sizes, decode_only):
    executable = Path(__file__).resolve().parents[1] / "build/cpp_runtime/metal/mfq-metal-qwen-gated-hc-test"
    if not executable.is_file():
        pytest.skip("build mfq-metal-qwen-gated-hc-test to exercise packed MHC kernels")
    tensors = {"test.mhc.pre.norm.weight": np.zeros(10240, dtype=np.float32)}
    for suffix, rows, width, gs in [("pre.down", 320, 10240, group_sizes[0]),
                                    ("pre.up", 10240, 320, group_sizes[1]),
                                    ("post.inject", 4, 10240, group_sizes[2])]:
        groups = (width + gs - 1) // gs
        qbits = np.arange(rows, dtype=np.uint8) % 8 + 1 if adaptive else np.full(rows, 8, dtype=np.uint8)
        kbits = np.arange(rows, dtype=np.uint8) % 3 + 6 if adaptive else np.full(rows, 7, dtype=np.uint8)
        codes = np.empty((rows, groups, gs), dtype=np.uint8)
        for row in range(rows):
            levels = 1 << int(qbits[row])
            codes[row].reshape(-1)[:] = (np.arange(groups * gs) * 3 + row) % (levels if decode_only else min(8, levels))
        tensors[f"test.mhc.{suffix}.weight"] = NintTensor(
            spec=NintSpec(8, gs, 7), shape=(rows, width), axis=0, q=codes,
            neuron_scale=np.full(rows, 1 / 8192, dtype=np.float32),
            neuron_min=np.full(rows, 1 / 4096, dtype=np.float32),
            sub_scale=np.broadcast_to(np.arange(groups, dtype=np.uint8) % 7 + 1, (rows, groups)).copy(),
            sub_min=np.ones((rows, groups), dtype=np.uint8), neuron_len=width,
            row_q_bits=qbits if adaptive else None, row_sub_bits=kbits if adaptive else None,
        )
    path = tmp_path / "packed_mhc.mfq"
    io.save(path, FileHeader(model_arch="qwen4_exp", num_tensors=len(tensors)), tensors)
    mode = "--packed-mhc-decode" if decode_only else "--packed-mhc"
    result = subprocess.run([str(executable), mode, str(path)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("final mixer passed" if decode_only else "4096-row prefill passed") in result.stdout


@pytest.mark.parametrize("name", [
    "model.block.0.position_embedding.ngram.head_offsets",
    "model.block.0.attention.mhc.pre.norm.weight",
])
def test_native_qwen_nint_mhc_does_not_relax_metadata_loading(tmp_path, name):
    executable = Path(__file__).resolve().parents[1] / "build/csrc/metal/mfq-metal-nint-rows-test"
    if not executable.is_file():
        pytest.skip("build mfq-metal-nint-rows-test to exercise the native model graph")
    reference, packed = _mhc_models(tmp_path, adaptive=True, projections="down")
    header, tensors = io.load(packed)
    tensors[name] = tensors["model.block.0.attention.mhc.pre.down.weight"]
    invalid = tmp_path / "invalid_metadata.mfq"
    io.save(invalid, header, tensors)
    result = subprocess.run([str(executable), "--qwen-ple", str(reference), str(invalid)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert f"requires a dense tensor: {name} (got NINT)" in result.stderr


def _cuda_models(tmp_path):
    fp8, nint = _models(tmp_path)
    header, tensors = io.load(fp8)
    prefix = "model.block.0.position_embedding.ngram"
    scale = float(tensors.pop(prefix + ".weight_scale")[0])
    # Use dense F16 as the CUDA oracle: same exact values, no FP8 scale
    # binding dependence and no use of the NINT loader/kernel under test.
    for shard in range(2):
        name = prefix + f".shard.{shard}.weight"
        raw = np.asarray(tensors[name], dtype=np.uint8)
        assert set(np.unique(raw)).issubset({0x00, 0x30, 0x38, 0x3C})
        lut = np.zeros(256, dtype=np.float32)
        lut[[0x00, 0x30, 0x38, 0x3C]] = [0.0, 0.5, 1.0, 1.5]
        tensors[name] = (lut[raw] * scale).astype(np.float16)
    baseline = tmp_path / "dense.mfq"
    io.save(baseline, header, tensors)
    return baseline, nint


@pytest.mark.parametrize("requests", [2, 4, 6, 8, 9, 16, 17, 32])
@pytest.mark.parametrize("quantized_experts", [False, True, "nvq"])
def test_cuda_flash_next_continuous_batching_preserves_request_state(tmp_path, requests, quantized_experts):
    executable = os.environ.get("MFQ_NINT_PLE_CUDA_DECODE")
    if not executable:
        pytest.skip("MFQ_NINT_PLE_CUDA_DECODE must point to CUDA mfq-diagnostics")
    _, model = _models(tmp_path, qsa=True, ple_layer=3, context_size=512,
                      quantized_experts=quantized_experts)
    environment = dict(os.environ, MFQ_MOE_PIPELINE="1", MFQ_MOE_TWO_STAGE_FFN="1",
                       MFQ_MOE_PRELOAD_ALL="1", MFQ_MOE_ASSERT_RESIDENT="1",
                       MFQ_MOE_HOST_CACHE_BYTES="8388608")
    trace_gdn = requests == 32 and quantized_experts == "nvq"
    if trace_gdn:
        environment["MFQ_TRACE_GDN_BATCH"] = "1"
    if requests == 17 and quantized_experts == "nvq":
        environment["MFQ_TRACE_LAYER_TIMINGS"] = "3"
        environment["MFQ_CONTINUOUS_BATCH_KV_INITIAL_CAPACITY"] = "64"
    process = subprocess.run([
        executable, "--model", str(model), "--ctx-size", "512",
        "--check-continuous-batching", "--continuous-batching-check-requests", str(requests),
        "--moe-gpu-cache-gb", "0.00004" if quantized_experts == "nvq" else ("0.1" if quantized_experts else "0"),
    ], capture_output=True, text=True, timeout=90, env=environment)
    assert process.returncode == 0, process.stdout + process.stderr
    assert "continuous_batching_check PASS" in process.stdout
    if trace_gdn:
        assert "gdn_batch_state eligible=1" in process.stderr
        assert "gdn_batch_core rows=32" in process.stderr
    if requests == 17 and quantized_experts == "nvq":
        phases = [dict(field.split("=", 1) for field in line.split()[1:])
                  for line in process.stderr.splitlines() if line.startswith("native_layer_gpu ")]
        tile_rows = int(environment.get("MFQ_CONTINUOUS_FFN_TILE", "8"))
        expected_tiles = {str(tile) for tile in range((requests + tile_rows - 1) // tile_rows)}
        assert {phase["tile"] for phase in phases if phase["from"] == "hot_begin"
                and phase["to"] == "hot_end"} >= expected_tiles
    if requests == 32:
        assert "continuous_batching_growth_check requests=32 boundary=256 PASS" in process.stdout
    state = next(line for line in process.stdout.splitlines()
                 if line.startswith("continuous_batching_state_check"))
    metrics = dict(field.split("=", 1) for field in state.split()[1:])
    assert int(metrics["captures"]) > 0
    assert int(metrics["replays"]) > int(metrics["captures"])
    if os.environ.get("MFQ_BATCH_PACKED_PREFILL") != "0":
        assert int(metrics["packed_prefills"]) > 0


async def _fresh_idle_status(client):
    # Status uses the scheduler's nonblocking 250 ms snapshot. Require a
    # snapshot captured after this call before asserting final counters.
    loop = asyncio.get_running_loop()
    started = loop.time()
    while loop.time() - started < 5:
        requested = loop.time()
        status = await client.status()
        if (status["runtime_metrics_age_ms"] <= (requested - started) * 1000
                and status.get("active_requests", 0) == 0
                and status.get("continuous_batching_active", 0) == 0):
            return status
        await asyncio.sleep(0.02)
    raise AssertionError("runtime metrics did not publish an idle snapshot")


@pytest.mark.parametrize("requests", [2, 32])
def test_cuda_flash_next_stdio_batches_normal_chat_requests(tmp_path, requests):
    from mfq.server.runtime.client import StdioRuntimeClient

    executable = os.environ.get("MFQ_FLASH_NEXT_CUDA_RUNTIME")
    if not executable:
        pytest.skip("MFQ_FLASH_NEXT_CUDA_RUNTIME must point to CUDA mfq-runtime")
    _, model = _models(tmp_path, qsa=True, ple_layer=3, context_size=512,
                      quantized_experts=True)
    # Token 0 is the fixture tokenizer's ordinary "a" token. Keep the worker
    # producing long enough for both requests to overlap, without EOS timing.
    header, weights = io.load(model)
    weights["model.output.weight"].fill(0)
    io.save(model,header,weights)
    tokenizer = _worker_tokenizer(tmp_path)
    environment = dict(os.environ, MFQ_MOE_PIPELINE="1", MFQ_MOE_TWO_STAGE_FFN="1",
                       MFQ_MOE_PRELOAD_ALL="1", MFQ_MOE_ASSERT_RESIDENT="1",
                       MFQ_MOE_HOST_CACHE_BYTES="8388608")

    async def run():
        with (tmp_path / "worker.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(executable,"--model",str(model),
                "--tokenizer",str(tokenizer),"--transport","stdio","--ctx-size","512",
                "--continuous-batching","32","--moe-gpu-cache-gb","0.00004",
                stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=log,env=environment)
            client = StdioRuntimeClient(process,control_timeout_seconds=10)
            try:
                await client.status()
                async def generate(content, penalty=False):
                    async with client.generate({"messages":[{"role":"user","content":content}],
                            "temperature":0,"top_k":1,"max_tokens":32,"stream":True,
                            "presence_penalty":-0.01 if penalty else 0}) as events:
                        return [event async for event in events if event is not None]
                prompts = ["ab" * (4 + row % 5 * 8) for row in range(requests)]
                reference = [await generate(prompt,row == 0) for row,prompt in enumerate(prompts)]
                parallel = await asyncio.gather(*(generate(prompt,row == 0) for row,prompt in enumerate(prompts)))
                def text(events):
                    return "".join(choice.get("delta",{}).get("content","")
                        for event in events for choice in event.get("choices",[]))
                assert list(map(text,parallel)) == list(map(text,reference))
                assert list(map(text,parallel)) == ["ab"*32] + ["a"*32] * (requests-1)
                status = await _fresh_idle_status(client)
                (tmp_path / "worker-status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
                assert status["continuous_batching_admissions"] == 2 * requests
                assert status["continuous_batching_cuda_graph_replays"] > 0
                assert status["continuous_batching_batched_greedy_batches"] > 0
                if requests == 32 and os.environ.get("MFQ_BATCH_PACKED_PREFILL") != "0":
                    assert status["continuous_batching_packed_prefill_batches"] > 0
                assert status["continuous_batching_decode_tokens"] > status["continuous_batching_decode_batches"]
                assert status["continuous_batching_active"] == 0
                assert status["kv_cache_bytes"] == 0
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await process.wait()

    asyncio.run(asyncio.wait_for(run(),30))


def test_cuda_flash_next_stdio_mtp_avoids_ordinary_batch_path(tmp_path):
    from mfq.server.runtime.client import StdioRuntimeClient

    executable=os.environ.get("MFQ_FLASH_NEXT_CUDA_RUNTIME")
    if not executable:
        pytest.skip("MFQ_FLASH_NEXT_CUDA_RUNTIME must point to CUDA mfq-runtime")
    _,model=_models(tmp_path,mtp=True,qsa=True,ple_layer=3,context_size=512,
        quantized_experts=True)
    header,weights=io.load(model);weights["model.output.weight"].fill(0);io.save(model,header,weights)
    tokenizer=_worker_tokenizer(tmp_path)
    environment=dict(os.environ,MFQ_MOE_PIPELINE="1",MFQ_MOE_TWO_STAGE_FFN="1",
        MFQ_MOE_PRELOAD_ALL="1",MFQ_MOE_ASSERT_RESIDENT="1",MFQ_MOE_HOST_CACHE_BYTES="8388608")

    async def run():
        with (tmp_path/"worker.log").open("wb") as log:
            process=await asyncio.create_subprocess_exec(executable,"--model",str(model),
                "--tokenizer",str(tokenizer),"--transport","stdio","--ctx-size","512",
                "--continuous-batching","2","--moe-gpu-cache-gb","0.00004",
                stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=log,env=environment)
            client=StdioRuntimeClient(process,control_timeout_seconds=15)
            try:
                await client.status()
                for use_mtp in [False,True,False,True]:
                    before=await client.status()
                    async with client.generate({"messages":[{"role":"user","content":"ababab"}],
                        "temperature":0,"top_k":1,"max_tokens":16,"enable_mtp":use_mtp,
                        "stream":True,"stream_options":{"include_usage":True},
                        "mfq_prefix_cache_enabled":False}) as events:
                        chunks=[event async for event in events if event is not None]
                    after=await _fresh_idle_status(client)
                    text="".join(choice.get("delta",{}).get("content","")
                        for event in chunks for choice in event.get("choices",[]))
                    assert text=="a"*16
                    metrics={key:value for event in chunks for key,value in event.get("mfq_metrics",{}).items()}
                    assert metrics["mtp_used"]==int(use_mtp),metrics
                    if use_mtp:
                        assert metrics["mtp_cycles"]>0,metrics
                        assert after["continuous_batching_admissions"]==before["continuous_batching_admissions"]
                        assert after["mtp_predictor_graph_replays"]>0,after
                    else:
                        assert after["continuous_batching_admissions"]==before["continuous_batching_admissions"]+1
            finally:
                with suppress(Exception):await client.aclose()
                if process.returncode is None:process.terminate()
                await process.wait()
    asyncio.run(asyncio.wait_for(run(),30))


@pytest.mark.parametrize("requests",[9,16,17,32])
def test_cuda_flash_next_large_nint_head_matches_serial_batches(tmp_path,requests):
    from mfq.quantize.nint_quant import quantize

    executable=os.environ.get("MFQ_NINT_PLE_CUDA_DECODE")
    if not executable:
        pytest.skip("MFQ_NINT_PLE_CUDA_DECODE must point to CUDA mfq-diagnostics")
    _,model=_models(tmp_path,qsa=True,ple_layer=3,context_size=512,
        quantized_experts=True,vocab_size=65536)
    header,weights=io.load(model)
    output=weights["model.output.weight"].copy()
    output[-32:]*=8
    weights["model.output.weight"]=quantize(output,NintSpec(4,8,6))
    io.save(model,header,weights)
    environment=dict(os.environ,MFQ_MOE_PIPELINE="1",MFQ_MOE_TWO_STAGE_FFN="1",
        MFQ_MOE_PRELOAD_ALL="1",MFQ_MOE_ASSERT_RESIDENT="1",MFQ_MOE_HOST_CACHE_BYTES="8388608")
    process=subprocess.run([executable,"--model",str(model),"--ctx-size","512",
        "--check-continuous-batching","--continuous-batching-check-requests",str(requests),
        "--moe-gpu-cache-gb","0.1"],capture_output=True,text=True,env=environment,timeout=30)
    assert process.returncode==0,process.stdout+process.stderr
    assert "continuous_batching_check PASS" in process.stdout
    assert f"continuous_batching_growth_check requests={requests} boundary=256 PASS" in process.stdout


@pytest.mark.parametrize("cancel",[False,True])
def test_cuda_flash_next_stdio_eof_cancels_concurrent_admissions(tmp_path,cancel):
    from mfq.server.runtime.client import _runtime_generate_params

    executable=os.environ.get("MFQ_FLASH_NEXT_CUDA_RUNTIME")
    if not executable:
        pytest.skip("MFQ_FLASH_NEXT_CUDA_RUNTIME must point to CUDA mfq-runtime")
    _,model=_models(tmp_path,qsa=True,ple_layer=3,context_size=512,quantized_experts=True)
    tokenizer=_worker_tokenizer(tmp_path)
    params=_runtime_generate_params({"messages":[{"role":"user","content":"aa"}],
        "temperature":0,"top_k":1,"max_tokens":128,"stream":True})
    commands=[{"v":1,"id":f"eof-{row}","op":"generate","params":params} for row in range(32)]
    if cancel:
        commands.extend({"v":1,"id":f"cancel-{row}","op":"request.cancel",
            "params":{"target_id":f"eof-{row}"}} for row in range(32))
    environment=dict(os.environ,MFQ_MOE_PIPELINE="1",MFQ_MOE_TWO_STAGE_FFN="1",
        MFQ_MOE_PRELOAD_ALL="1",MFQ_MOE_ASSERT_RESIDENT="1",MFQ_MOE_HOST_CACHE_BYTES="8388608")
    result=subprocess.run([executable,"--model",str(model),"--tokenizer",str(tokenizer),
        "--transport","stdio","--ctx-size","512","--continuous-batching","32",
        "--moe-gpu-cache-gb","0.00004"],
        input="".join(json.dumps(command)+"\n" for command in commands),
        capture_output=True,text=True,env=environment,timeout=15)
    assert result.returncode==0,result.stderr
    frames=[json.loads(line) for line in result.stdout.splitlines()]
    terminals=[frame for frame in frames if frame["type"] in {"done","error"}]
    assert sorted(frame["id"] for frame in terminals)==sorted(f"eof-{row}" for row in range(32))
    assert not any(frame["type"]=="error" for frame in frames)


def test_cuda_dense_fixture_matches_canonical_nint_rows(tmp_path):
    baseline, nint = _cuda_models(tmp_path)
    _, dense = io.load(baseline)
    _, packed = io.load(nint)
    for shard in range(2):
        name = f"model.block.0.position_embedding.ngram.shard.{shard}.weight"
        table = packed[name]
        scales = table.neuron_scale[:, None, None] * table.sub_scale[:, :, None]
        minima = table.neuron_min[:, None, None] * table.sub_min[:, :, None]
        reconstructed = (scales * table.q - minima).reshape(32, -1)[:, :16]
        np.testing.assert_array_equal(dense[name], reconstructed)


def test_cuda_qwen_nint_ple_matches_dense_native_graph(tmp_path):
    executable = os.environ.get("MFQ_NINT_PLE_CUDA_DECODE")
    if not executable:
        pytest.skip("MFQ_NINT_PLE_CUDA_DECODE must point to the built CUDA mfq-decode")
    baseline, nint = _cuda_models(tmp_path)
    outputs = []
    for model in (baseline, nint):
        process = subprocess.run(
            [executable, "--model", str(model), "--ctx-size", "64", "--check-flash-next"],
            capture_output=True, text=True, timeout=180)
        assert process.returncode == 0, process.stdout + process.stderr
        payload = next(line.removeprefix("flash_next_check ")
                       for line in process.stdout.splitlines()
                       if line.startswith("flash_next_check "))
        outputs.append(json.loads(payload))
    assert outputs[0].keys() == outputs[1].keys()
    for name, reference in outputs[0].items():
        if name == "architecture":
            assert reference == outputs[1][name] == "qwen4_exp"
            continue
        assert reference["shape"] == outputs[1][name]["shape"]
        np.testing.assert_array_equal(reference["data"], outputs[1][name]["data"])
