from __future__ import annotations

import asyncio
from collections import namedtuple
from types import SimpleNamespace

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.protocol.models import RuntimeResourceSnapshot
from mfq.server.services import resource_monitor as module
from mfq.server.services.resource_monitor import ResourceMonitor
from mfq.server.services.service import ServerService
from mfq.server.state.storage import SessionStore
from tests.test_server_service import FakeBackend


def test_host_intervals_are_measured_and_missing_bandwidth_stays_unknown(monkeypatch):
    cpu = namedtuple("Cpu", "user idle")
    disk = namedtuple("Disk", "read_bytes write_bytes busy_time")
    times = iter([cpu(10, 10), cpu(14, 16), cpu(18, 22)])
    disks = iter([{"disk0": disk(100, 200, 100)}, {"disk0": disk(300, 400, 600)}, {"disk0": disk(1, 2, 3)}])
    clock = iter([10, 12, 14])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(module.psutil, "cpu_times", lambda: next(times))
    monkeypatch.setattr(module.psutil, "disk_io_counters", lambda **kwargs: next(disks))
    monkeypatch.setattr(module, "_gpu_usage", lambda: [{"name": "GPU", "utilization_percent": 0}])
    monkeypatch.setattr(module, "hardware_identity", lambda: SimpleNamespace(
        cpu_name="Apple M5 Max", cpu_cores=18, memory_bandwidth_bytes_per_second=614_000_000_000))
    monitor = ResourceMonitor()
    first = monitor._host_sample()
    assert first["cpu_name"] == "Apple M5 Max"
    assert first["cpu_cores"] == 18
    assert first["cpu_utilization_percent"] is None
    assert first["disks"][0]["read_bytes_per_second"] is None
    second = monitor._host_sample()
    assert second["cpu_utilization_percent"] == pytest.approx(40)
    assert second["disks"][0]["read_bytes_per_second"] == 100
    assert second["disks"][0]["write_bytes_per_second"] == 100
    assert second["disks"][0]["busy_percent"] == 25
    assert second["disks"][0]["bandwidth_utilization_percent"] is None
    assert second["memory_bandwidth_bytes_per_second"] is None
    assert second["memory_bandwidth_utilization_percent"] is None
    assert second["memory_bandwidth_limit_bytes_per_second"] == 614_000_000_000
    assert monitor._host_sample()["disks"][0]["read_bytes_per_second"] is None


def test_weight_rates_are_per_instance_and_never_reuse_missing_or_reset_counters(monkeypatch):
    clock = iter([10, 12, 14, 16, 18])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    monitor = ResourceMonitor()
    def status(identifier="a", value=0):
        return {"instance_id": identifier, "model": "Qwen", "ssd_expert_bytes_read": value,
                "ple_source_bytes_read": value, "ssd_ple_enabled": 1}
    assert monitor._weight_sample([status()])[0]["ple_read_bytes_per_second"] is None
    rates = monitor._weight_sample([status(value=2048), status("b", 10_000_000)])
    assert rates[0]["expert_read_bytes_per_second"] == 1024
    assert rates[0]["ple_read_bytes_per_second"] == 1024
    assert rates[1]["ple_read_bytes_per_second"] is None
    assert monitor._weight_sample([{"instance_id": "a", "model": "Qwen"}])[0]["ple_read_bytes_per_second"] is None
    assert monitor._weight_sample([status(value=4096)])[0]["ple_read_bytes_per_second"] is None
    assert monitor._weight_sample([status(value=0)])[0]["ple_read_bytes_per_second"] is None


def test_gpu_statistics_missing_is_not_idle(monkeypatch):
    import plistlib
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    devices = [{"model": "GPU 1", "gpu-core-count": 40,
                "PerformanceStatistics": {"Device Utilization %": 36}}, {"model": "GPU 2", "gpu-core-count": 0}]
    monkeypatch.setattr(module, "_command", lambda args: plistlib.dumps(devices).decode())
    assert module._gpu_usage() == [{"name": "GPU 1", "core_count": 40, "utilization_percent": 36},
                                   {"name": "GPU 2", "core_count": None, "utilization_percent": None}]


def test_resource_endpoint_works_without_loaded_models_and_coalesces_samples(tmp_path, monkeypatch):
    async def run():
        store = SessionStore(tmp_path / "server.sqlite")
        service = ServerService(store, FakeBackend())
        monitor = ResourceMonitor()
        calls = []
        def host():
            calls.append(1)
            monitor._sample_time = module.time.monotonic()
            return {"sampled_at": 100, "cpu_utilization_percent": None, "gpus": [], "disks": []}
        monkeypatch.setattr(monitor, "_host_sample", host)
        service.resource_monitor = monitor
        app = create_app(service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            responses = await asyncio.gather(*(client.get("/api/v1/runtime/resources") for _ in range(3)))
            assert all(response.status_code == 200 for response in responses)
            assert len(calls) == 1
            result = RuntimeResourceSnapshot.model_validate(responses[0].json())
            assert result.weights == []
            assert result.cpu_utilization_percent is None
            assert result.memory_bandwidth_utilization_percent is None
        await service.aclose()
    asyncio.run(run())
