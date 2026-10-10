"""Resource telemetry keeps residency, file payloads and unknown data distinct."""
import math
import asyncio
from types import SimpleNamespace
from uuid import uuid4

from mfq.server.runtime.runtime_pool import RuntimePool
from mfq.server.runtime.runtime_pool import _Runtime
from mfq.server.protocol.models import RuntimeInstanceState
from mfq.server.state.catalog import ModelCatalog
from tests.test_server_models import IdleBackend, _model


def test_memory_resources_use_worker_measurements_not_allocator_or_io_totals():
    memory = RuntimePool._memory_resources({
        "resident_weight_bytes": 1024, "mlx_active_bytes": 999999,
        "mlx_cache_bytes": 99999, "kv_cache_bytes": 100, "kv_cache_contexts": 1,
        "prefix_cache_hot_bytes": 50, "prefix_cache_sessions": 2,
        "prefix_cache_max_bytes": 2048,
        "prefix_cache_resident_sessions": 2,
        "prefix_cache_hot_blocks": 3, "prefix_cache_disk_blocks": 10,
        "ssd_expert_enabled": 1, "ssd_expert_payload_bytes": 5000,
        "ssd_expert_bytes_read": 900000, "ssd_ple_enabled": 0,
        "ssd_ple_payload_bytes": 0,
    })
    assert memory.resident_weight_bytes == 1024
    assert memory.kv_bytes == 150
    assert memory.prefix_cache_bytes == 50
    assert memory.prefix_cache_limit_bytes == 2048
    assert memory.context_count == 3
    assert memory.prefix_cache_blocks == 3
    assert memory.ssd_experts is True
    assert memory.ssd_expert_bytes == 5000
    assert memory.ssd_ple is False


def test_busy_worker_retains_last_breakdown_instead_of_resetting_to_zero():
    previous = RuntimePool._memory_resources({"resident_weight_bytes": 1024, "ssd_ple_enabled": 1})
    assert RuntimePool._memory_resources({"mlx_active_bytes": 2048}, previous) == previous


def test_older_worker_does_not_fabricate_weight_bytes_or_ssd_modes():
    memory = RuntimePool._memory_resources({"mlx_active_bytes": 1024, "prefix_cache_bytes": 128})
    assert memory.resident_weight_bytes is None
    assert memory.kv_bytes is None
    assert memory.context_count is None
    assert memory.ssd_experts is None
    assert memory.ssd_ple is None
    assert memory.ssd_kv is None
    assert memory.ssd_kv_bytes is None


def test_streamed_kv_reports_storage_and_traffic_separately_from_resident_kv():
    memory = RuntimePool._memory_resources({"resident_weight_bytes": 1024,
        "kv_cache_bytes": 200, "prefix_cache_hot_bytes": 50, "qsa_kv_offload_enabled": 1,
        "qsa_kv_ssd_bytes": 4096, "qsa_kv_resident_bytes": 150, "qsa_kv_budget_bytes": 2048,
        "qsa_kv_pending_bytes": 20, "qsa_kv_ssd_read_bytes": 8192,
        "qsa_kv_ssd_written_bytes": 12288, "qsa_kv_ssd_reads": 2, "qsa_kv_ram_hits": 6})
    assert memory.ssd_kv is True and memory.ssd_kv_bytes == 4096
    assert memory.streaming_kv_resident_bytes == 150 and memory.streaming_kv_budget_bytes == 2048
    assert memory.streaming_kv_pending_bytes == 20
    assert memory.ssd_kv_read_bytes == 8192 and memory.ssd_kv_written_bytes == 12288
    assert memory.ssd_kv_reads == 2 and memory.ssd_kv_hits == 6
    assert memory.kv_bytes == 250 and memory.resident_weight_bytes == 1024
    assert RuntimePool._memory_resources({}, memory) == memory
    zero = RuntimePool._memory_resources({"qsa_kv_offload_enabled": 0, "qsa_kv_ssd_bytes": 0})
    assert zero.ssd_kv is False and zero.ssd_kv_bytes == 0
    invalid = RuntimePool._memory_resources({"qsa_kv_ssd_bytes": math.nan,
        "qsa_kv_ssd_read_bytes": -1, "qsa_kv_ssd_reads": True, "qsa_kv_offload_enabled": 2})
    assert invalid.ssd_kv is None and invalid.ssd_kv_bytes is None and invalid.ssd_kv_reads is None
    assert invalid.ssd_kv_read_bytes is None


def test_disk_only_prefix_contexts_are_not_counted_as_resident():
    memory = RuntimePool._memory_resources({"kv_cache_bytes": 100, "kv_cache_contexts": 1,
        "prefix_cache_hot_bytes": 0, "prefix_cache_sessions": 20, "prefix_cache_hot_blocks": 0})
    assert memory.context_count == 1
    assert memory.prefix_cache_blocks == 0
    partial = RuntimePool._memory_resources({"kv_cache_bytes": 100, "kv_cache_contexts": 1,
        "prefix_cache_hot_bytes": 50, "prefix_cache_sessions": 20, "prefix_cache_hot_blocks": 1})
    assert partial.context_count is None


def test_invalid_measurements_are_unknown_but_explicit_zero_is_valid():
    memory = RuntimePool._memory_resources({"resident_weight_bytes": math.inf,
        "kv_cache_bytes": -1, "ssd_ple_payload_bytes": math.nan, "ssd_expert_enabled": 2})
    assert memory.resident_weight_bytes is None
    assert memory.kv_bytes is None
    assert memory.ssd_ple_bytes is None
    assert memory.ssd_experts is None
    zero = RuntimePool._memory_resources({"resident_weight_bytes": 0, "ssd_ple_enabled": 0})
    assert zero.resident_weight_bytes == 0
    assert zero.ssd_ple is False


def test_instance_list_reports_per_model_resources_and_retains_busy_telemetry(tmp_path):
    class Backend(IdleBackend):
        def __init__(self, weight_bytes):
            self.status = {"resident_weight_bytes": weight_bytes, "kv_cache_bytes": 64,
                "kv_cache_contexts": 1, "prefix_cache_bytes": 0, "prefix_cache_sessions": 0,
                "prefix_cache_snapshots": 0, "ssd_expert_enabled": 0, "ssd_expert_payload_bytes": 0,
                "ssd_ple_enabled": 1, "ssd_ple_payload_bytes": 2048}

        async def runtime_status(self):
            return self.status

    async def run():
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        pool = RuntimePool(catalog, tmp_path / "runtime", metric_interval_seconds=0)
        for name, weight_bytes in (("first", 1024), ("second", 3072)):
            path = tmp_path / f"{name}.mfq"
            _model(path)
            instance = _Runtime(id=uuid4(), artifact=await catalog.resolve_path(path),
                process=SimpleNamespace(returncode=None), backend=Backend(weight_bytes), port=0,
                context_size=4096, state=RuntimeInstanceState.READY)
            pool._instances[instance.id] = instance
        result = await pool.instances()
        assert [item.memory.resident_weight_bytes for item in result.data] == [1024, 3072]
        assert all(item.memory.context_count == 1 for item in result.data)
        assert all(item.memory.ssd_ple_bytes == 2048 for item in result.data)
        for instance in pool._instances.values():
            instance.backend.status = {"mlx_active_bytes": 9999}
            instance.usage_refreshed_at = 0
        assert [item.memory for item in (await pool.instances()).data] == [item.memory for item in result.data]
        assert result.model_dump(mode="json")["data"][1]["memory"]["ssd_experts"] is False

    asyncio.run(run())
