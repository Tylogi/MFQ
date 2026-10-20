import asyncio

import pytest

from mfq.server.protocol.models import ModelLoadRequest
from mfq.server.runtime.runtime_pool import RuntimePool
from mfq.server.services.hardware import GpuMemory, HardwareIdentity
from mfq.server.state.catalog import ModelCatalog
from tests.test_qsa_kv_offload_policy import artifact


@pytest.mark.parametrize("ids,visible", [([], ""), (["3"], "3"), (["3", "1"], "3,1")])
def test_cuda_worker_device_mapping_and_yarn(tmp_path, monkeypatch, ids, visible):
    model = artifact(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.native_tokenizer_arguments", lambda *_: [])
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda", max_requests_per_instance=2,
        automatic_memory_budget=False)
    request = ModelLoadRequest(model=model.resource.name, device_ids=ids, context_size=524288, yarn_enabled=True)
    args, environment = pool._launch_configuration(model, request, port=8105)
    assert environment.get("CUDA_VISIBLE_DEVICES", "") == visible
    assert args[args.index("--ctx-size") + 1] == "524288"
    assert "--yarn-context-extension" in args
    assert "--layer-parallel" not in args
    assert args[args.index("--continuous-batching") + 1] == "2"
    assert pool._cuda_device_ids(request) == (ids[:1] or ["0"])
    asyncio.run(pool.aclose())


def test_capacity_tracks_selected_physical_gpus_and_runtime_environment(tmp_path, monkeypatch):
    GiB = 1 << 30
    hardware = HardwareIdentity("CPU", 8, gpu_memory=(
        GpuMemory("A", 24 * GiB, device_id="1"), GpuMemory("B", 48 * GiB, device_id="3")))
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.hardware_identity", lambda: hardware)
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda", automatic_memory_budget=False,
        runtime_environment={"CUDA_VISIBLE_DEVICES": "3,1"})
    request = ModelLoadRequest(model="test")
    assert pool._cuda_device_ids(request) == ["3"]
    assert pool._cuda_discrete_device_capacity(request) == 48 * GiB
    assert pool._cuda_discrete_device_capacity(request.model_copy(update={"device_ids": ["3", "1"]})) == 48 * GiB
    assert pool._cuda_discrete_device_capacity(request.model_copy(update={"device_ids": ["2"]})) is None
    asyncio.run(pool.aclose())
