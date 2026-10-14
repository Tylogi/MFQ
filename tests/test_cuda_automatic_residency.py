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
from tests.mixed_family_fixtures import make_flat_family

GIB = 1 << 30


def artifact(tmp_path):
    path = tmp_path / "model.mfq"
    experts = MfeTensor((2, 3, 96), (MfePool(np.arange(2, dtype=np.int32), make_flat_family("NVQ2J")),))
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


@pytest.mark.parametrize("available,enabled", [(52 * GIB, True), (8 * GIB, False)])
def test_cuda_launch_enables_ram_pipeline_when_experts_fit(tmp_path, monkeypatch, available, enabled):
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
    if not enabled:
        with pytest.raises(JobExecutionError) as raised:
            pool._launch_configuration(model, request, port=1234)
        assert raised.value.detail.code == "runtime_ram_expert_budget_exceeded"
        return
    command, environment = pool._launch_configuration(model, request, port=1234)
    assert "--moe-gpu-cache-gb" in command
    assert (environment.get("MFQ_MOE_PRELOAD_ALL") == "1") == enabled
    if enabled:
        assert environment["MFQ_MOE_PIPELINE"] == "1"
        assert environment["MFQ_MOE_TWO_STAGE_FFN"] == "1"
        assert environment["MFQ_MOE_RAM_PCIE"] == "1"
        assert environment["MFQ_MOE_RAM_PCIE_FRACTION"] == "1"
        assert environment["MFQ_MOE_DECODE_HOST_PIN"] == "1"
        assert environment["MFQ_MOE_PREFILL_LAYER"] == "1"
        assert environment["MFQ_MOE_PREFILL_LAYER_ASYNC"] == "0"
        assert environment["MFQ_MOE_RESIDENCY_INTERVAL"] == "4"
        assert float(environment["MFQ_CUDA_VRAM_LIMIT_GIB"]) == 22.5
        assert int(environment["MFQ_MOE_HOST_CACHE_BYTES"]) >= 28 * GIB


def test_cuda_ram_pipeline_rejects_dense_expert_pools(tmp_path):
    path = tmp_path / "dense.mfq"
    experts = MfeTensor((2, 3, 96), (MfePool(np.arange(2, dtype=np.int32), np.zeros((6, 96), dtype=np.float16)),))
    save(path, FileHeader(version=2, model_arch="qwen4_exp"),
         {"model.block.0.mlp.experts.gate.weight": experts})
    assert not supports_cuda_ram_expert_pipeline(path)


def test_cuda_ram_pipeline_accepts_packed_nint_predictor_experts(tmp_path):
    model = artifact(tmp_path)
    packed = replace(quantize(np.ones((6, 96), dtype=np.float32), NintSpec(4, 24, 6), axis=0), shape=(2, 3, 96))
    save(model.path, FileHeader(version=2, model_arch="qwen4_exp"),
         {"predictor.block.0.mlp.experts.gate_up.weight": packed})
    assert supports_cuda_ram_expert_pipeline(model.path)
