from __future__ import annotations

import asyncio
import ctypes
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from mfq.server.runtime import host_memory
from mfq.server.runtime.host_memory import HostMemorySnapshot
from mfq.server.runtime.runtime_pool import RuntimePool, _Runtime
from mfq.server.protocol.models import RuntimeInstanceState
from mfq.server.state.catalog import ModelCatalog
from tests.test_server_models import IdleBackend, _model


def test_host_vm_counter_layout_and_short_snapshots(monkeypatch):
    class Host:
        def host_statistics64(self, _host, _flavor, stats, count):
            value = host_memory._VmStatistics64.from_buffer(stats)
            value.free, value.active, value.inactive, value.wired = 2, 3, 4, 5
            value.compressions = (1 << 33) + 7
            value.swapouts = (1 << 34) + 9
            value.external_pages = 11
            count._obj.value = ctypes.sizeof(value) // 4
            return 0
    monkeypatch.setattr(host_memory, "_MACH_HOST", (Host(), 1, 16384))
    monkeypatch.setattr(host_memory, "total_physical_memory", lambda: 128 << 30)
    result = host_memory.host_memory_snapshot()
    assert result.free == 2 * 16384
    assert result.wired == 5 * 16384
    assert result.compression_bytes == ((1 << 33) + 7) * 16384
    assert result.swapout_bytes == ((1 << 34) + 9) * 16384
    assert result.external == 11 * 16384
    class LegacyHost(Host):
        def host_statistics64(self, *args):
            super().host_statistics64(*args)
            args[-1]._obj.value = (host_memory._VmStatistics64.swapouts.offset + 8) // 4
            return 0
    monkeypatch.setattr(host_memory, "_MACH_HOST", (LegacyHost(), 1, 16384))
    legacy = host_memory.host_memory_snapshot()
    assert legacy.compression_bytes == result.compression_bytes
    assert legacy.external == 0
    class ShortHost(Host):
        def host_statistics64(self, *args):
            super().host_statistics64(*args)
            args[-1]._obj.value = 4
            return 0
    monkeypatch.setattr(host_memory, "_MACH_HOST", (ShortHost(), 1, 16384))
    short = host_memory.host_memory_snapshot()
    assert short.free == result.free
    assert short.swapout_bytes is None
    assert short.compression_bytes is None
    assert short.external == 0


def test_balanced_reclaimable_memory_counts_file_cache_once():
    snapshot = HostMemorySnapshot(128 << 30, 2 << 30, 20 << 30, 4 << 30, 0, external=10 << 30)
    assert snapshot.reclaimable(active_ratio=0) == 12 << 30
    assert snapshot.reclaimable(active_ratio=0.5) == 19 << 30
    assert host_memory.automatic_memory_reserve(128 << 30) == 8 << 30
    assert host_memory.automatic_memory_reserve(32 << 30) == 3 << 30


def test_kernel_process_footprint_is_used_instead_of_rss(monkeypatch):
    def query(pid, flavor, pointer):
        assert pid == 123 and flavor == 0
        pointer._obj.resident_size = 20 << 30
        pointer._obj.phys_footprint = 25 << 30
        return 0
    monkeypatch.setattr(host_memory, "_PROCESS_USAGE", query)
    assert host_memory.process_physical_footprint(123) == 25 << 30
    monkeypatch.setattr(host_memory, "_PROCESS_USAGE", lambda *_: -1)
    assert host_memory.process_physical_footprint(123) is None


def test_explicit_metal_kernel_limit_overrides_recommendation(monkeypatch):
    class Host:
        def sysctlbyname(self, name, value, *_):
            assert name == b'iogpu.wired_limit_mb'
            value._obj.value = 110 << 10
            return 0
    monkeypatch.setattr(host_memory, '_MACH_HOST', (Host(), 1, 16384))
    monkeypatch.setattr(host_memory, 'metal_recommended_working_set_size', lambda: 96 << 30)
    assert host_memory.metal_allocation_limit() == 110 << 30
    monkeypatch.setattr(host_memory, '_MACH_HOST', None)
    assert host_memory.metal_allocation_limit() == 96 << 30


def test_live_budget_includes_own_usage_but_not_pending_capacity(tmp_path, monkeypatch):
    gib = 1 << 30
    snapshot = HostMemorySnapshot(128 * gib, 5 * gib, 30 * gib, 10 * gib, 80 * gib, external=12 * gib)
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.host_memory_snapshot", lambda: snapshot)
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.process_physical_footprint", lambda _: 80 * gib)
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.metal_allocation_limit", lambda: 108 * gib)
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", max_runtime_memory_bytes=120 * gib)
    memory = RuntimePool._memory_resources({"metal_wired_bytes": 80 * gib})
    item = SimpleNamespace(state=RuntimeInstanceState.READY, process=SimpleNamespace(pid=123), memory=memory)
    pool._instances[uuid4()] = item
    pool._load_bytes['not_yet_allocated'] = 80 * gib
    assert pool._effective_runtime_memory_budget_locked() == 89 * gib
    assert pool.memory_policy_status()['effective_total_limit_bytes'] == 89 * gib
    snapshot = replace(snapshot, free=1 * gib, external=10 * gib)
    assert pool._effective_runtime_memory_budget_locked() == 83 * gib


def test_metal_limit_only_caps_gpu_memory_not_cpu_cache(tmp_path, monkeypatch):
    gib = 1 << 30
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.host_memory_snapshot",
        lambda: HostMemorySnapshot(128 * gib, 40 * gib, 0, 0, 80 * gib))
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.process_physical_footprint", lambda _: 86 * gib)
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.metal_allocation_limit", lambda: 80 * gib)
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", max_runtime_memory_bytes=120 * gib)
    memory = RuntimePool._memory_resources({"metal_wired_bytes": 80 * gib})
    pool._instances[uuid4()] = SimpleNamespace(state=RuntimeInstanceState.READY, process=SimpleNamespace(pid=123), memory=memory)
    assert pool._effective_runtime_memory_budget_locked() == 86 * gib


@pytest.mark.parametrize("counter", ["compression_bytes", "swapout_bytes"])
def test_only_new_host_paging_under_low_headroom_triggers_pressure(monkeypatch, counter):
    snapshot = HostMemorySnapshot(128 << 30, 8 << 30, 0, 0, 100 << 30,
        compression_bytes=1 << 40, swapout_bytes=1 << 40)
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.host_memory_snapshot", lambda: snapshot)
    now = 100.0
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.time.monotonic", lambda: now)
    pool = RuntimePool(ModelCatalog([]), "runtime", max_runtime_memory_bytes=100 << 30)
    pool._observe_host_memory_pressure_locked()
    assert pool._host_vm_pressure_until == 0
    snapshot = replace(snapshot, **{counter: getattr(snapshot, counter) + (128 << 20)})
    pool._observe_host_memory_pressure_locked()
    assert pool._runtime_memory_pressure_locked()[0] == "hard"
    now += 6
    assert pool._runtime_memory_pressure_locked()[0] == "normal"
    snapshot = replace(snapshot, **{counter: 0})
    pool._observe_host_memory_pressure_locked()
    assert pool._host_vm_pressure_until == 0
    snapshot = replace(snapshot, free=32 << 30, **{counter: 1 << 40})
    pool._observe_host_memory_pressure_locked()
    assert pool._host_vm_pressure_until == 0


def test_host_paging_reaper_retires_idle_lru_not_busy_queued_or_pinned(tmp_path, monkeypatch):
    async def run():
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        pool = RuntimePool(catalog, "runtime", max_instances=4,
            max_runtime_memory_bytes=100 << 30, metric_interval_seconds=0.25)
        snapshot = HostMemorySnapshot(128 << 30, 11 << 30, 0, 0, 100 << 30,
            compression_bytes=0, swapout_bytes=0)
        monkeypatch.setattr("mfq.server.runtime.runtime_pool.host_memory_snapshot", lambda: snapshot)
        instances = []
        for index, name in enumerate(("old", "queued", "busy", "pinned")):
            _model(tmp_path / f"{name}.mfq")
            instance = _Runtime(id=uuid4(), artifact=await catalog.resolve(name),
                process=SimpleNamespace(returncode=None), backend=IdleBackend(), port=0,
                context_size=4096, state=RuntimeInstanceState.READY,
                resident_bytes=10 << 30, reserved_bytes=10 << 30,
                last_used_at=datetime(2026, 1, index + 1, tzinfo=timezone.utc))
            instances.append(instance)
            pool._instances[instance.id] = instance
        instances[1].queued_requests = 1
        instances[2].state = RuntimeInstanceState.BUSY
        instances[2].active_requests = 1
        instances[3].pinned = True
        stopped = asyncio.Event()
        async def stop(victim):
            assert victim is instances[0]
            stopped.set()
        async def refresh(_instance):
            return None
        pool._stop_process, pool._refresh_instance_usage = stop, refresh
        pool._observe_host_memory_pressure_locked()
        assert pool._runtime_memory_pressure_locked()[0] != "hard"
        snapshot = replace(snapshot, swapout_bytes=128 << 20)
        pool._idle_reaper_task = asyncio.create_task(pool._idle_reaper())
        pool._idle_reaper_wakeup.set()
        await asyncio.wait_for(stopped.wait(), timeout=2)
        assert instances[0].id not in pool._instances
        assert list(pool._instances) == [item.id for item in instances[1:]]
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


def test_wired_telemetry_survives_busy_status_and_never_means_weight_bytes():
    memory = RuntimePool._memory_resources({"resident_weight_bytes": 100,
        "metal_wired_bytes": 130, "metal_wired_limit_bytes": 200, "metal_wired_available": 1})
    assert memory.resident_weight_bytes == 100
    assert memory.wired_bytes == 130
    assert memory.wired_limit_bytes == 200
    assert memory.wired_available is True
    assert RuntimePool._memory_resources({}, memory) == memory
    assert RuntimePool._memory_resources({}).wired_available is None
    assert RuntimePool._memory_resources({"metal_wired_available": 0}).wired_available is False
