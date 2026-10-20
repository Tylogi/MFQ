"""Bounded RAM and actual checkpoint reads through eager and retained FFN graphs."""
import os
import subprocess
import asyncio
from contextlib import suppress

import numpy as np
import pytest

from mfq.formats.header import FileHeader
from mfq.formats.io import save
from mfq.formats.mfe import MfePool, MfeTensor
from mfq.formats.mx import MxTensor
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import quantize
from mfq.formats.nvq import D4_256, NVQ3_D4, NvqJscTensor


@pytest.mark.parametrize("wire_fallback", ["0", "1"])
def test_clipped_generic_ssd_ffn(tmp_path, wire_fallback):
    executable = os.environ.get("MFQ_CUDA_SSD_EXPERT_TEST")
    if not executable:
        pytest.skip("MFQ_CUDA_SSD_EXPERT_TEST required")
    rng = np.random.default_rng(2038)
    def packed(family, rows, width):
        if family == "mxfp4":
            return MxTensor("MXFP4", (rows, width),
                rng.integers(0, 256, (rows, width // 2), dtype=np.uint8),
                rng.integers(122, 128, (rows, width // 32), dtype=np.uint8))
        if family == "nint":
            return quantize(rng.normal(0, .03, (rows, width)).astype(np.float32), NintSpec(4, 24, 6))
        books = np.stack((D4_256, np.roll(D4_256, 1, axis=0))).astype(np.int16) * 8
        return NvqJscTensor((rows, width), 0, width,
            rng.uniform(.005, .05, rows).astype(np.float32),
            np.arange(16, dtype=np.float32), np.arange(16, dtype=np.uint8) & 1,
            rng.integers(1, 16, (rows, width // 24), dtype=np.uint8),
            rng.integers(0, 256, (rows, width // 4), dtype=np.uint8),
            rng.integers(0, 128, (rows, width // 8), dtype=np.uint8),
            books.astype(np.int8), base_spec=NVQ3_D4)

    for family in ("nint", "nvq3j", "mxfp4", "mixed"):
        experts, width = (12, 384) if family == "mixed" else (3, 128) if family == "mxfp4" else (64, 96)
        families = ("nint", "nvq3j", "mxfp4") if family == "mixed" else (family,)
        for projection in range(3):
            pools = []
            for cohort, kind in enumerate(families):
                ids = (np.array([2, 0, 1], dtype=np.int32) if family == "mxfp4" else
                       np.arange(cohort, experts, len(families), dtype=np.int32))
                pools.append(MfePool(ids, packed(kind, len(ids) * width, width)))
            save(tmp_path / f"{family}-{projection}.mfq", FileHeader(version=2, model_arch="deepseek_v41"),
                 {"linear.weight": MfeTensor((experts, width, width), tuple(pools))})
    env = dict(os.environ, MFQ_MOE_SSD_EXPERTS="1", MFQ_MOE_NVQ_DENSE="0",
               MFQ_MOE_PREFILL_LAYER="0", MFQ_MOE_PREFILL_GROUPS="1",
               MFQ_MOE_TRANSFER_CACHE="1", MFQ_MOE_RAM_PCIE_FRACTION="1",
               MFQ_MOE_WIRE_FALLBACK=wire_fallback,
               MFQ_MOE_NATIVE_DMA_BATCH="1", MFQ_MOE_KV_HOST_PACKET="1")
    result = subprocess.run([executable, str(tmp_path), "--clipped-check"],
                            env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "generic_clipped_ffn_cases=352" in result.stdout


@pytest.mark.parametrize("pcie_fraction,read_mode,native_dma,ram_pin_bytes", [
    (1.0,"buffered","1","0"),(0.5,"buffered","1","0"),(1.0,"direct","1","0"),
    (1.0,"buffered","0","0"),(1.0,"buffered","1","16384")])
def test_ssd_expert_streaming(tmp_path, pcie_fraction,read_mode,native_dma,ram_pin_bytes):
    from tests.mixed_family_fixtures import make_flat_family
    executable = os.environ.get("MFQ_CUDA_SSD_EXPERT_TEST")
    if not executable:
        pytest.skip("MFQ_CUDA_SSD_EXPERT_TEST required")
    rng = np.random.default_rng(2038)
    for family in ("nint", "nvq2j", "nvq3j", "nvq2j-l", "nvq2j-xl", "nvq3j-l"):
        for projection in range(4 if family == "nint" else 3):
            width = 128 if family == "nvq2j-xl" else 96
            output = 192 if projection == 3 else width
            values = (quantize(rng.normal(0, .03, (64 * output, 96)).astype(np.float32), NintSpec(4, 24, 6))
                      if family == "nint" else make_flat_family(family.upper(), rows=64 * output, neuron_len=width, seed=2038+projection))
            save(tmp_path / f"{family}-{projection}.mfq", FileHeader(version=2, model_arch="qwen4_exp"),
                 {"linear.weight": MfeTensor((64, output, width), (MfePool(np.arange(64, dtype=np.int32), values),))})
    # Complete projections must exceed the GPU loan's 64 KiB admission floor
    # to exercise GU uploads preceding the completion of Down's SSD reads.
    for projection in range(3):
        values=quantize(rng.normal(0,.03,(64*384,384)).astype(np.float32),NintSpec(4,24,6))
        save(tmp_path / f"phase-wide-{projection}.mfq",FileHeader(version=2,model_arch="qwen4_exp"),
             {"linear.weight":MfeTensor((64,384,384),(MfePool(np.arange(64,dtype=np.int32),values),))})
    families = ("nint", "nvq2j", "nvq3j", "nvq2j-l", "nvq2j-xl", "nvq3j-l")
    for projection in range(3):
        pools = []
        for index, family in enumerate(families):
            ids = np.arange(index, 128, len(families), dtype=np.int32)
            values = (quantize(rng.normal(0, .03, (len(ids) * 128, 128)).astype(np.float32), NintSpec(4, 24, 6))
                      if family == "nint" else make_flat_family(family.upper(), rows=len(ids) * 128,
                          neuron_len=128, seed=4096+projection*len(families)+index))
            pools.append(MfePool(ids, values))
        save(tmp_path / f"mixed-{projection}.mfq", FileHeader(version=2, model_arch="qwen4_exp"),
             {"linear.weight": MfeTensor((128, 128, 128), tuple(pools))})
    env = dict(os.environ, MFQ_MODEL_FILE_IO=read_mode, MFQ_MOE_SSD_EXPERTS="1", MFQ_MOE_NVQ_DENSE="0", MFQ_MOE_PRELOAD_ALL="0", MFQ_MOE_ASSERT_RESIDENT="0",
        MFQ_MOE_TRANSFER_CACHE="1", MFQ_MOE_TWO_STAGE_FFN="1", MFQ_MOE_PREFILL_LAYER="0",
        MFQ_MOE_PREFILL_GROUPS="1", MFQ_MOE_RESIDENCY_PROMOTE_TRANSFER="1", MFQ_MOE_RAM_PCIE_FRACTION="1",
        MFQ_MOE_PREFILL_GROUP_SIZE="64" if pcie_fraction < 1 else "16",
        MFQ_MOE_RESIDENCY_NATIVE_DMA=native_dma,
        MFQ_MOE_RAM_PIN_GIB=str(int(ram_pin_bytes)/(1<<30)),
        MFQ_SSD_EXPERT_TEST_PCIE_FRACTION=str(pcie_fraction),
        MFQ_CUDA_WORKSPACE_RESERVE_GIB="0.01", CUDA_MODULE_LOADING="LAZY")
    result = subprocess.run([executable, str(tmp_path)], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SSD expert streaming cases=243 PASS" in result.stdout
    if native_dma == "1" and pcie_fraction == 1 and ram_pin_bytes == "0":
        assert "moe_ssd_gu_io_overlap scope=first_upload enabled=1" in result.stderr
        assert "moe_ssd_gu_compute_overlap scope=first_ready enabled=1" in result.stderr
        assert "moe_ssd_down_io_overlap scope=first_upload enabled=1" in result.stderr
    if os.name == "nt" and read_mode == "direct":
        assert "model_direct_staging_batch enabled=1" in result.stderr


@pytest.mark.parametrize("wide,grouped,router_prefetch", [
    (False, True, False), (True, True, False), (True, False, False),
    pytest.param(True, True, True, id="router-prefetch"),
    pytest.param(True, True, None, id="router-auto"),
])
@pytest.mark.parametrize("read_mode,preload_requested", [("buffered",False),("direct",False),("direct",True)])
def test_ssd_experts_across_requests_and_mtp(tmp_path, wide, grouped,router_prefetch,read_mode,preload_requested):
    from mfq.server.runtime.client import StdioRuntimeClient
    from tests.test_cuda_flash_next_runtime import mtp_fixture, write_qwen_fixture
    from tests.test_native_nint_ple import _worker_tokenizer

    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("MFQ_CUDA_RESOURCE_TEST_RUNTIME required")
    # Consecutive backbone layers exercise shared pinned metadata retirement.
    config, weights = mtp_fixture("qwen", 1, hidden=256 if wide else 16)
    assert config["text_config"]["num_hidden_layers"] >= 2
    config["text_config"]["max_position_embeddings"] = 512
    if wide:
        config["text_config"].update(moe_intermediate_size=1024, shared_expert_intermediate_size=1024)
    for name, value in list(weights.items()):
        if name.endswith(".experts.gate_up.weight"):
            for role, matrix in zip(("gate", "up"), np.split(value, 2, axis=1)):
                weights[name.replace("gate_up", role)] = np.tile(matrix, (1, 64, 1)) if wide else matrix.copy()
            del weights[name]
        elif wide and name.endswith(".experts.down.weight"):
            weights[name] = np.tile(value, (1, 1, 64)) / 64
        elif wide and name.endswith((".shared_expert.gate.weight", ".shared_expert.up.weight")):
            weights[name] = np.tile(value, (64, 1))
        elif wide and name.endswith(".shared_expert.down.weight"):
            weights[name] = np.tile(value, (1, 64)) / 64
    for name, value in list(weights.items()):
        if ".experts." in name:
            experts, output, width = value.shape
            packed = quantize(value.reshape(experts * output, width), NintSpec(4, 8, 6))
            weights[name] = MfeTensor(value.shape, (MfePool(np.arange(experts, dtype=np.int32), packed),))
        elif ".ngram.shard." in name:
            weights[name] = quantize(value, NintSpec(4, 8, 6))
    model = tmp_path / "streamed.mfq"
    write_qwen_fixture(model, config, weights, True)
    tokenizer = _worker_tokenizer(tmp_path)

    async def run(ssd):
        workspace_cache=ssd and wide and grouped and read_mode=="direct" and not preload_requested
        env = dict(os.environ, MFQ_MODEL_FILE_IO=read_mode, MFQ_PLE_FILE_MAP="0", MFQ_PROFILE_SSD_EXPERTS="1",
            MFQ_MOE_PIPELINE="1", MFQ_MOE_TWO_STAGE_FFN="1", MFQ_MOE_SSD_EXPERTS="1",
            MFQ_MOE_PRELOAD_ALL=str(int(not ssd or preload_requested)), MFQ_MOE_ASSERT_RESIDENT=str(int(not ssd)), MFQ_MOE_NVQ_DENSE="0",
            MFQ_MOE_HOST_CACHE_BYTES=str((1 << 19 if wide else 4096) if ssd else 1 << 28), MFQ_MOE_RAM_PCIE="1", MFQ_MOE_RAM_PCIE_FRACTION="1",
            MFQ_MOE_RESIDENCY_WARM="1", MFQ_MOE_RESIDENCY_PROJECTION_HEAT="1", MFQ_MOE_RESIDENCY_PROMOTE_TRANSFER="1",
            MFQ_MOE_ROUTER_LOOKAHEAD=str(int(router_prefetch is True and ssd)),
            MFQ_MOE_PREFILL_LAYER="0", MFQ_RUNTIME_DISABLE_PREFIX_CACHE="1",
            MFQ_KV_TURBOQUANT_BITS="0", MFQ_CUDA_WORKSPACE_RESERVE_GIB="0.01")
        if workspace_cache:env["MFQ_CUDA_WORKSPACE_RESERVE_GIB"]="1.5"
        if router_prefetch is None and ssd:env.pop("MFQ_MOE_ROUTER_LOOKAHEAD")
        env.pop("MFQ_MOE_PREFILL_GROUPS", None)
        if not grouped:
            env["MFQ_MOE_PREFILL_GROUPS"] = "0"
        with (tmp_path / f"worker-{ssd}.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(executable, "--model", str(model), "--tokenizer", str(tokenizer),
                "--transport", "stdio", "--ctx-size", "512", "--moe-gpu-cache-gb", "0.001" if wide else "0.000005", "--yarn-context-extension",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, env=env)
            client = StdioRuntimeClient(process, control_timeout_seconds=30)
            try:
                status = await client.status()
                assert bool(status["ssd_expert_enabled"]) == ssd
                assert bool(status["ssd_reuse_policy_enabled"]) == ssd
                assert bool(status["ssd_ram_reuse_policy_enabled"]) == ssd
                if read_mode == "direct":
                    assert status["model_io_direct"] == 1
                    assert status["model_io_physical_bytes"] >= status["model_io_logical_bytes"] > 0
                    assert status["model_io_cached_staging_bytes"] > 0
                if ssd:
                    assert status["ssd_ram_reuse_promotions"] == 0, "startup learned synthetic RAM cache priority"
                if workspace_cache:
                    assert status["prefill_workspace_expert_capacity_bytes"] > 0
                startup_captures = (tmp_path / f"worker-{ssd}.log").read_text(errors="replace").count(
                    "runtime_qwen4_graph_capture role=")
                outputs = []
                for mtp,repetitions in ((False,160),(True,240),(False,144)):
                    prompt = "ab" * repetitions
                    async with client.generate(dict(messages=[dict(role="user", content=prompt)], mfq_preformatted_prompt=prompt,
                        temperature=0, seed=1234, max_tokens=16, enable_mtp=mtp, mtp_max_draft_tokens=3,
                        mfq_prefix_cache_enabled=False, stream=True)) as events:
                        chunks = [event async for event in events if event]
                    assert not any("error" in event for event in chunks), chunks
                    status = await client.status()
                    assert status["last_request"]["prompt_tokens"] > 128
                    outputs.append("".join(choice.get("delta", {}).get("content", "")
                        for event in chunks for choice in event.get("choices", [])))
                    assert outputs[-1]
                status = await client.status()
                if ssd:
                    assert status["ssd_expert_bytes_read"] > 0 and status["ram_expert_cache_evictions"] > 0
                    assert bool(status.get("ssd_mapped_read_calls", 0)) == (read_mode == "buffered")
                    if router_prefetch or router_prefetch is None:
                        assert status["ssd_router_prefetch_jobs"] > 0
                        assert status["ssd_router_prefetch_bytes"] > 0
                        assert status["ssd_router_prefetch_used_jobs"] > 0
                text = (tmp_path / f"worker-{ssd}.log").read_text(errors="replace")
                captures = [line for line in text.splitlines() if "runtime_qwen4_graph_capture role=" in line]
                assert captures
                assert len(captures) == startup_captures, "requests captured additional decode graphs"
                assert "runtime_mtp_depth_plan" not in text, "requests rebuilt startup graphs"
                if ssd and grouped:
                    assert "moe_ssd_prefill_dispatch path=grouped" in text
                if ssd:
                    assert f"moe_ssd_source_dispatch mapped={int(read_mode == 'buffered')} " in text
                    assert "batch_fields=1" in text
                if workspace_cache:
                    assert "runtime_prefill_workspace_cache action=lend" in text
                    await client.set_resident_memory_budget(int(status["gpu_0_resident_bytes"]))
                    for _ in range(20):
                        status=await client.status()
                        if status["prefill_workspace_expert_capacity_bytes"]==0:break
                        await asyncio.sleep(.05)
                    assert status["prefill_workspace_expert_capacity_bytes"]==0,"memory budget retained the revocable prefill cache"
                return outputs
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)

    expected = asyncio.run(run(False))
    assert asyncio.run(run(True)) == expected
