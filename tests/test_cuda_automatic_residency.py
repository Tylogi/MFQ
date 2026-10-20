from datetime import datetime, timezone
from dataclasses import replace

import numpy as np
import pytest

from mfq.formats.header import FileHeader
from mfq.formats.io import save
from mfq.formats.mfe import MfePool, MfeTensor
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import quantize
from mfq.server.protocol.models import ModelArtifactResource, ModelLoadRequest
from mfq.server.runtime.native import supports_cuda_ram_expert_pipeline
from mfq.server.runtime.runtime_pool import RuntimePool
from mfq.server.services.hardware import GpuMemory, HardwareIdentity
from mfq.server.state.catalog import DiscoveredModel, ModelCatalog
from mfq.server.services.jobs import JobExecutionError
from mfq.formats.nvq import E8_256, NVQ2_E8, NvqJscTensor


def make_nvq_fixture():
    return NvqJscTensor((6, 96), 0, 96, np.ones(6, dtype=np.float32),
        np.arange(16, dtype=np.float32), np.zeros(16, dtype=np.uint8),
        np.ones((6, 4), dtype=np.uint8), np.zeros((6, 12), dtype=np.uint8),
        np.zeros((6, 12), dtype=np.uint8), (E8_256[None].astype(np.int16) * 8).astype(np.int8),
        base_spec=NVQ2_E8)

GIB = 1 << 30


@pytest.fixture(autouse=True)
def _no_host_commit_accounting(monkeypatch):
    monkeypatch.setattr(
        "mfq.server.runtime.runtime_pool.host_commit_capacity", lambda: (None, None)
    )
    monkeypatch.setattr("mfq.server.runtime.runtime_pool._wddm_host", lambda: False)
    monkeypatch.delenv("MFQ_CUDA_VRAM_LIMIT_GIB", raising=False)


def artifact(tmp_path):
    path = tmp_path / "model.mfq"
    experts = MfeTensor((2, 3, 96), (MfePool(np.arange(2, dtype=np.int32), make_nvq_fixture()),))
    save(path, FileHeader(version=2, model_arch="qwen4_exp"),
         {"model.block.0.mlp.experts.gate.weight": experts})
    resource = ModelArtifactResource(id="1" * 32, name="model", architecture="qwen4_exp",
        total_bytes=48 * GIB, shard_count=1, tensor_count=1, record_count=1,
        complete=True, loadable=True, modified_at=datetime.now(timezone.utc))
    return DiscoveredModel(resource=resource, path=path, routed_expert_bytes=44 * GIB)


@pytest.mark.parametrize("unified,device", [(False, "0"), (False, "1"), (True, "0")])
def test_cuda_auto_expert_budget_uses_selected_physical_device(tmp_path, monkeypatch, unified, device):
    hardware = HardwareIdentity("cpu", 8, gpu_memory=(
        GpuMemory("NVIDIA GB10" if unified else "RTX GPU", 24 * GIB, unified=unified),
        GpuMemory("RTX GPU", 16 * GIB),
    )) if not unified else HardwareIdentity("cpu", 8, gpu_memory=(GpuMemory("NVIDIA GB10", 128 * GIB, unified=True),))
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.hardware_identity", lambda: hardware)
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda")
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", device_ids=[device])
    selected = pool._apply_automatic_expert_residency(model, request)
    if unified:
        assert selected.moe_gpu_cache_gb is None
    else:
        capacity = hardware.gpu_memory[int(device)].capacity_bytes
        assert selected.moe_gpu_cache_gb is not None and selected.moe_gpu_cache_gb > 0
        # Native admission prices actual allocations; no extra third for PP.
        assert pool._estimated_load_bytes(model, selected) == capacity - max(GIB, capacity // 16)
    explicit = request.model_copy(update={"moe_gpu_cache_gb": 7})
    assert pool._apply_automatic_expert_residency(model, explicit).moe_gpu_cache_gb == 7


@pytest.mark.parametrize("wddm,capacity,configured,limit", [
    (False, 96 * GIB, None, 90 * GIB),
    # The desktop shares a WDDM device; a fifth of it stays free.
    (True, 96 * GIB, None, 96 * GIB - 96 * GIB // 5),
    (True, 16 * GIB, None, 12 * GIB),
    # An explicit limit wins on every platform.
    (True, 96 * GIB, "40", 40 * GIB),
])
def test_cuda_vram_limit_leaves_headroom_for_the_windows_desktop(tmp_path, monkeypatch, wddm,
                                                                 capacity, configured, limit):
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.hardware_identity", lambda: HardwareIdentity(
        "cpu", 8, gpu_memory=(GpuMemory("RTX GPU", capacity),)))
    monkeypatch.setattr("mfq.server.runtime.runtime_pool._wddm_host", lambda: wddm)
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda")
    if configured:
        pool.runtime_environment["MFQ_CUDA_VRAM_LIMIT_GIB"] = configured
    request = ModelLoadRequest(model="model", device_ids=["0"])
    assert pool._cuda_device_vram_limit(request) == limit
    selected = pool._apply_automatic_expert_residency(artifact(tmp_path), request)
    # Automatic residency fills the worker's limit, never the whole device.
    assert pool._estimated_load_bytes(artifact(tmp_path), selected) <= limit


@pytest.mark.parametrize("available,enabled", [(52 * GIB, True), (8 * GIB, False)])
def test_cuda_launch_selects_ram_or_ssd_expert_cache(tmp_path, monkeypatch, available, enabled):
    monkeypatch.delenv("MFQ_MODEL_FILE_IO", raising=False)
    monkeypatch.delenv("MFQ_PLE_FILE_MAP", raising=False)
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.memory_topology", lambda: {"memory_architecture": "discrete"})
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.host_memory_capacity", lambda: (64 * GIB, available))
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.hardware_identity", lambda: HardwareIdentity(
        "cpu", 8, gpu_memory=(GpuMemory("RTX GPU", 24 * GIB),)))
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda", automatic_memory_budget=False)
    pool.memory_policy.prefix_directory = str(tmp_path / "prefix")
    model = artifact(tmp_path)
    assert supports_cuda_ram_expert_pipeline(model.path)
    # Select the mocked GPU explicitly, independent of the test runner's
    # CUDA_VISIBLE_DEVICES setting.
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16, device_ids=["0"])
    command, environment = pool._launch_configuration(model, request, port=1234)
    assert "--moe-gpu-cache-gb" in command
    assert (environment.get("MFQ_MOE_PRELOAD_ALL") == "1") == enabled
    assert environment["MFQ_MOE_SSD_EXPERTS"] == "1"
    assert environment["MFQ_MOE_ASSERT_RESIDENT"] == str(int(enabled))
    if not enabled:
        assert int(environment["MFQ_MOE_HOST_CACHE_BYTES"]) == 6 * GIB
        assert environment["MFQ_MOE_NVQ_DENSE"] == "0"
        assert environment["MFQ_MODEL_FILE_IO"] == "direct"
        assert environment["MFQ_PLE_FILE_MAP"] == "1"
    if enabled:
        assert environment["MFQ_MOE_PIPELINE"] == "1"
        assert environment["MFQ_MOE_TWO_STAGE_FFN"] == "1"
        assert environment["MFQ_MOE_RAM_PCIE"] == "1"
        assert environment["MFQ_MOE_RAM_PCIE_FRACTION"] == "1"
        assert environment["MFQ_MOE_DECODE_HOST_PIN"] == "1"
        assert environment["MFQ_MOE_PREFILL_LAYER"] == "1"
        assert environment["MFQ_MOE_PREFILL_LAYER_ASYNC"] == "1"
        assert environment["MFQ_MOE_RESIDENCY_INTERVAL"] == "4"
        assert float(environment["MFQ_CUDA_VRAM_LIMIT_GIB"]) == 22.5
        assert int(environment["MFQ_MOE_HOST_CACHE_BYTES"]) >= 28 * GIB


def test_cuda_ram_pipeline_rejects_dense_expert_pools(tmp_path):
    path = tmp_path / "dense.mfq"
    experts = MfeTensor((2, 3, 96), (MfePool(np.arange(2, dtype=np.int32), np.zeros((6, 96), dtype=np.float16)),))
    save(path, FileHeader(version=2, model_arch="qwen4_exp"),
         {"model.block.0.mlp.experts.gate.weight": experts})
    assert not supports_cuda_ram_expert_pipeline(path)


def test_cuda_ram_pipeline_ignores_resident_dense_banks(tmp_path):
    # A model whose MTP head keeps dense BF16 experts still streams its
    # quantized backbone experts; the dense head runs on the device.
    path = tmp_path / "mixed.mfq"
    experts = MfeTensor((2, 3, 96), (MfePool(np.arange(2, dtype=np.int32), make_nvq_fixture()),))
    dense = np.zeros((2, 3, 96), dtype=np.float16)
    save(path, FileHeader(version=2, model_arch="qwen4_exp"), {
        "model.block.0.mlp.experts.gate.weight": experts,
        "predictor.block.1.mlp.experts.up.weight": dense,
    })
    assert supports_cuda_ram_expert_pipeline(path)


def test_cuda_ram_pipeline_needs_a_quantized_expert_bank(tmp_path):
    path = tmp_path / "dense-only.mfq"
    save(path, FileHeader(version=2, model_arch="qwen4_exp"),
         {"model.block.0.mlp.experts.gate.weight": np.zeros((2, 3, 96), dtype=np.float16)})
    assert not supports_cuda_ram_expert_pipeline(path)


def test_cuda_ssd_experts_respect_explicit_ram_cache_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.memory_topology", lambda: {"memory_architecture": "discrete"})
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.host_memory_capacity", lambda: (64 * GIB, 52 * GIB))
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.hardware_identity", lambda: HardwareIdentity(
        "cpu", 8, gpu_memory=(GpuMemory("RTX GPU", 24 * GIB),)))
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda", automatic_memory_budget=False)
    pool.runtime_environment["MFQ_MOE_HOST_CACHE_BYTES"] = str(2 * GIB)
    pool.runtime_environment.update(MFQ_MODEL_FILE_IO="buffered", MFQ_PLE_FILE_MAP="0")
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16, device_ids=["0"])
    _, environment = pool._launch_configuration(model, request, port=1234)
    assert environment["MFQ_MOE_HOST_CACHE_BYTES"] == str(2 * GIB)
    assert environment["MFQ_MOE_PRELOAD_ALL"] == "0"
    assert environment["MFQ_MODEL_FILE_IO"] == "buffered"
    assert environment["MFQ_PLE_FILE_MAP"] == "0"
    pool.runtime_environment["MFQ_MOE_SSD_EXPERTS"] = "0"
    with pytest.raises(JobExecutionError) as raised:
        pool._launch_configuration(model, request, port=1234)
    assert raised.value.detail.code == "runtime_ram_expert_budget_exceeded"


def _commit_budget_pool(
    tmp_path, monkeypatch, *, architecture, commit_available, available=100 * GIB
):
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.memory_topology",
                        lambda: {"memory_architecture": architecture})
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.host_memory_capacity",
                        lambda: (128 * GIB, available))
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.host_commit_capacity",
                        lambda: (224 * GIB, commit_available))
    monkeypatch.setattr(
        "mfq.server.runtime.runtime_pool.hardware_identity",
        lambda: HardwareIdentity("cpu", 8, gpu_memory=(GpuMemory("RTX GPU", 96 * GIB),)),
    )
    return RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda",
                       automatic_memory_budget=False)


def test_windows_discrete_expert_budget_is_limited_by_commit(tmp_path, monkeypatch):
    pool = _commit_budget_pool(tmp_path, monkeypatch, architecture="discrete",
                               commit_available=120 * GIB)
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16)
    plan = pool._cuda_expert_host_budget_plan(model, request)
    # 20 GiB device weights (dense plus cached experts) and 4 GiB workspace
    # are committed from 120 GiB headroom before the reserve, an eighth of
    # the 224 GiB commit limit (28 GiB).
    assert plan.commit_footprint_bytes == 24 * GIB
    assert plan.commit_bytes == 68 * GIB
    assert plan.physical_bytes == 98 * GIB
    assert plan.commit_limited
    assert pool._cuda_expert_host_budget(model, request) == 68 * GIB


def test_windows_commit_footprint_covers_the_worker_vram_limit(tmp_path, monkeypatch):
    pool = _commit_budget_pool(tmp_path, monkeypatch, architecture="discrete",
                               commit_available=120 * GIB)
    monkeypatch.setattr(pool, "_cuda_discrete_device_capacity", lambda _request: 96 * GIB)
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16)
    plan = pool._cuda_expert_host_budget_plan(model, request)
    # The worker may grow to its VRAM limit (96 GiB minus 1/16), which WDDM
    # commits even though the load estimate is only 24 GiB.
    assert plan.commit_footprint_bytes == 90 * GIB
    assert plan.commit_bytes == 2 * GIB
    assert pool._cuda_expert_host_budget(model, request) == 2 * GIB


def test_expert_budget_ignores_commit_without_windows_accounting(tmp_path, monkeypatch):
    pool = _commit_budget_pool(tmp_path, monkeypatch, architecture="discrete",
                               commit_available=None)
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16)
    plan = pool._cuda_expert_host_budget_plan(model, request)
    assert plan.commit_bytes is None and not plan.commit_limited
    assert pool._cuda_expert_host_budget(model, request) == 98 * GIB


def test_unified_expert_budget_keeps_physical_accounting(tmp_path, monkeypatch):
    pool = _commit_budget_pool(tmp_path, monkeypatch, architecture="unified",
                               commit_available=4 * GIB)
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16)
    # Unified memory already subtracts the estimated load from RAM; commit
    # headroom must not shrink the shared-memory budget a second time.
    assert pool._cuda_expert_host_budget(model, request) == 78 * GIB


def test_windows_commit_pressure_streams_experts_from_ssd(tmp_path, monkeypatch, caplog):
    import logging

    monkeypatch.delenv("MFQ_MODEL_FILE_IO", raising=False)
    monkeypatch.delenv("MFQ_PLE_FILE_MAP", raising=False)
    pool = _commit_budget_pool(tmp_path, monkeypatch, architecture="discrete",
                               commit_available=40 * GIB)
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16, device_ids=["0"])
    assert pool._cuda_expert_host_budget(model, request) == 0
    with caplog.at_level(logging.INFO, logger="mfq.server.runtime.runtime_pool"):
        _, environment = pool._launch_configuration(model, request, port=1234)
    assert "limited_by=commit" in caplog.text
    assert environment["MFQ_MOE_PRELOAD_ALL"] == "0"
    assert environment["MFQ_MOE_HOST_CACHE_BYTES"] == "0"
    assert environment["MFQ_MOE_SSD_EXPERTS"] == "1"


def test_preload_requires_room_for_every_routed_expert(tmp_path, monkeypatch):
    monkeypatch.delenv("MFQ_MODEL_FILE_IO", raising=False)
    monkeypatch.delenv("MFQ_PLE_FILE_MAP", raising=False)
    pool = _commit_budget_pool(tmp_path, monkeypatch, architecture="discrete",
                               commit_available=None, available=36 * GIB)
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16, device_ids=["0"])
    # Routed experts need 44 GiB and the cold subset only 28 GiB. The 34 GiB
    # host budget fits the cold subset but not a complete preload.
    assert pool._cuda_expert_host_budget(model, request) == 34 * GIB
    _, environment = pool._launch_configuration(model, request, port=1234)
    assert environment["MFQ_MOE_PRELOAD_ALL"] == "0"
    assert environment["MFQ_MOE_HOST_CACHE_BYTES"] == str(34 * GIB)
    assert environment["MFQ_MOE_SSD_EXPERTS"] == "1"


def test_preload_kv_ram_limit_charges_the_complete_expert_set(tmp_path, monkeypatch):
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.memory_topology",
                        lambda: {"memory_architecture": "discrete"})
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.host_memory_capacity",
                        lambda: (64 * GIB, 52 * GIB))
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda",
                       automatic_memory_budget=False)
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16)
    assert pool._cuda_expert_host_budget(model, request) == 50 * GIB
    # Preload holds all 44 GiB of routed experts, not only the 28 GiB that
    # the GPU cache cannot mirror, so KV RAM gets 52 - 44 - 2 GiB.
    assert pool._kv_ram_limit(model, request) == 6 * GIB


def test_cuda_expert_cache_leaves_ram_for_preallocated_kv(tmp_path, monkeypatch):
    from mfq.server.runtime.kv_budget import KvTierBudget
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.memory_topology", lambda: {"memory_architecture": "discrete"})
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.host_memory_capacity", lambda: (64 * GIB, 12 * GIB))
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.checkpoint_kv_budget", lambda *args: KvTierBudget(0, 6 * GIB, 0, 6 * GIB, 0.1))
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda", automatic_memory_budget=False)
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16)
    assert pool._cuda_expert_host_budget(model, request) == 4 * GIB
    assert pool._kv_ram_limit(model, request) == 6 * GIB


def test_cuda_ram_pipeline_accepts_packed_nint_predictor_experts(tmp_path):
    model = artifact(tmp_path)
    packed = replace(quantize(np.ones((6, 96), dtype=np.float32), NintSpec(4, 24, 6), axis=0), shape=(2, 3, 96))
    save(model.path, FileHeader(version=2, model_arch="qwen4_exp"),
         {"predictor.block.0.mlp.experts.gate_up.weight": packed})
    assert supports_cuda_ram_expert_pipeline(model.path)


def test_cuda_ram_pipeline_accepts_mxfp4_predictor_with_quantized_target(tmp_path):
    from mfq.formats.mx import MxTensor

    model = artifact(tmp_path)
    target = MfeTensor((2, 3, 96), (MfePool(np.arange(2, dtype=np.int32), make_nvq_fixture()),))
    mx = MxTensor("MXFP4", (6, 96), np.zeros((6, 48), dtype=np.uint8),
                  np.full((6, 3), 127, dtype=np.uint8))
    predictor = MfeTensor((2, 3, 96), (MfePool(np.arange(2, dtype=np.int32), mx),))
    save(model.path, FileHeader(version=2, model_arch="deepseek_v41"), {
        "model.block.0.mlp.experts.gate.weight": target,
        "predictor.stage.0.mlp.experts.down.weight": predictor,
    })
    assert supports_cuda_ram_expert_pipeline(model.path)
