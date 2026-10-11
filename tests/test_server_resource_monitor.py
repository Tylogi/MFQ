from __future__ import annotations

import asyncio
import plistlib
from collections import namedtuple
from types import SimpleNamespace

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.protocol.models import RuntimeGpuUtilization, RuntimeResourceSnapshot
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
    monitor._physical_disks = {"disk0"}
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


def test_macos_only_samples_whole_physical_disks(monkeypatch):
    disk = namedtuple("Disk", "read_bytes write_bytes")
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(module, "_command", lambda args: plistlib.dumps({"WholeDisks": ["disk0", "disk2"]}).decode())
    monkeypatch.setattr(module.psutil, "disk_io_counters", lambda **_: {
        name: disk(100, 200) for name in ("disk0", "disk0s2", "disk2", "disk3", "disk4", "disk5", "disk6", "disk7", "disk9")})
    monkeypatch.setattr(module, "_gpu_usage", lambda: [])
    monitor = ResourceMonitor()
    assert [item["name"] for item in monitor._host_sample()["disks"]] == ["disk0", "disk2"]
    assert set(monitor._disks) == {"disk0", "disk2"}


def test_returning_after_a_long_gap_resets_cpu_disk_and_weight_rate_baselines(monkeypatch):
    cpu = namedtuple("Cpu", "user idle")
    disk = namedtuple("Disk", "read_bytes write_bytes busy_time")
    times = iter([cpu(10, 10), cpu(1010, 1010), cpu(1014, 1016)])
    disks = iter([{"disk0": disk(100, 200, 100)}, {"disk0": disk(100_100, 100_200, 10_100)},
        {"disk0": disk(100_300, 100_400, 10_600)}])
    monkeypatch.setattr(module.psutil, "cpu_times", lambda: next(times))
    monkeypatch.setattr(module.psutil, "disk_io_counters", lambda **kwargs: next(disks))
    monkeypatch.setattr(module, "_gpu_usage", lambda: [])
    clock = iter([10, 10, 1000, 1000, 1002, 1002])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    monitor = ResourceMonitor()
    monitor._physical_disks = {"disk0"}
    def weights(value):
        return [{"instance_id": "a", "model": "Qwen", "ple_source_bytes_read": value}]
    monitor._host_sample()
    monitor._weight_sample(weights(0))
    returned = monitor._host_sample()
    assert returned["interval_seconds"] is None
    assert returned["cpu_utilization_percent"] is None
    assert returned["disks"][0]["read_bytes_per_second"] is None
    assert returned["disks"][0]["write_bytes_per_second"] is None
    assert returned["disks"][0]["busy_percent"] is None
    assert monitor._weight_sample(weights(1_000_000))[0]["ple_read_bytes_per_second"] is None
    recent = monitor._host_sample()
    assert recent["interval_seconds"] == 2
    assert recent["cpu_utilization_percent"] == 40
    assert recent["disks"][0]["read_bytes_per_second"] == 100
    assert recent["disks"][0]["busy_percent"] == 25
    assert monitor._weight_sample(weights(1_002_048))[0]["ple_read_bytes_per_second"] == 1024


def test_physical_disk_topology_refreshes_and_discovery_failure_never_exposes_virtual_disks(monkeypatch):
    calls = []
    replies = iter([plistlib.dumps({"WholeDisks": ["disk0"]}).decode(),
        plistlib.dumps({"WholeDisks": ["disk0", "disk2"]}).decode(), ""])
    def query(args):
        calls.append(args)
        return next(replies)
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(module, "_command", query)
    monitor = ResourceMonitor()
    assert monitor._physical_disk_names(100) == {"disk0"}
    assert monitor._physical_disk_names(101) == {"disk0"}
    assert len(calls) == 1
    assert monitor._physical_disk_names(131) == {"disk0", "disk2"}
    assert monitor._physical_disk_names(162) == {"disk0", "disk2"}
    assert all(args == ["diskutil", "list", "-plist", "physical"] for args in calls)
    monkeypatch.setattr(module, "_command", lambda _: "")
    assert ResourceMonitor()._physical_disk_names(100) == set()


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


def test_discrete_gpu_memory_is_measured_separately_from_host_ram(monkeypatch):
    monkeypatch.setattr(module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(module, "_command", lambda _: "NVIDIA RTX 3090 Ti, 50, 24576, 2048, 22528\nNVIDIA GB10, N/A, N/A, N/A, N/A")
    values = module._gpu_usage()
    assert values[0]["memory_total_bytes"] == 24 << 30
    assert values[0]["memory_used_bytes"] == 2 << 30
    assert values[0]["memory_available_bytes"] == 22 << 30
    assert values[1]["memory_total_bytes"] is None and values[1]["utilization_percent"] is None


def test_streamed_kv_rates_track_completed_file_io_per_instance(monkeypatch):
    clock = iter([10, 12, 14, 16, 18, 30, 32])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    monitor = ResourceMonitor()
    def status(identifier="a", read=0, written=0):
        return {"instance_id": identifier, "model": "Qwen", "qsa_kv_offload_enabled": 1,
            "qsa_kv_ssd_read_bytes": read, "qsa_kv_ssd_written_bytes": written}
    assert monitor._weight_sample([status()])[0]["kv_read_bytes_per_second"] is None
    rates = monitor._weight_sample([status(read=4096, written=8192), status("b", 100000, 200000)])
    assert rates[0]["kv_read_bytes_per_second"] == 2048
    assert rates[0]["kv_write_bytes_per_second"] == 4096
    assert rates[1]["kv_read_bytes_per_second"] is None
    missing = monitor._weight_sample([{"instance_id": "a", "model": "Qwen"}])[0]
    assert missing["kv_read_bytes_per_second"] is None and missing["kv_write_bytes_per_second"] is None
    assert monitor._weight_sample([status(read=8192, written=16384)])[0]["kv_read_bytes_per_second"] is None
    assert monitor._weight_sample([status()])[0]["kv_read_bytes_per_second"] is None
    assert monitor._weight_sample([status(read=100000, written=200000)])[0]["kv_write_bytes_per_second"] is None
    assert monitor._weight_sample([status(read=104096, written=208192)])[0]["kv_write_bytes_per_second"] == 4096


@pytest.mark.parametrize("statistics", ["unavailable", [], 17, False])
def test_macos_gpu_statistics_with_unexpected_shape_keep_identity_without_invented_usage(monkeypatch, statistics):
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    devices = [{"model": "Apple M5 Max", "gpu-core-count": 40, "PerformanceStatistics": statistics}]
    monkeypatch.setattr(module, "_command", lambda args: plistlib.dumps(devices).decode())
    assert module._gpu_usage() == [{"name": "Apple M5 Max", "core_count": 40, "utilization_percent": None}]


@pytest.mark.parametrize("devices", [17, False])
def test_macos_gpu_query_with_non_array_plist_does_not_fail_host_monitoring(monkeypatch, devices):
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(module, "_command", lambda args: plistlib.dumps(devices).decode())
    assert module._gpu_usage() == []


@pytest.mark.parametrize("name", [17, False, [], {}, b"unavailable", ""])
def test_malformed_gpu_name_does_not_break_resource_protocol(monkeypatch, name):
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(module, "_command", lambda args: plistlib.dumps([
        {"model": name, "PerformanceStatistics": {"Device Utilization %": 36}}]).decode())
    gpu = RuntimeGpuUtilization.model_validate(module._gpu_usage()[0])
    assert gpu.name == "GPU" and gpu.utilization_percent == 36


def test_missing_gpu_sample_preserves_previously_detected_device_identity(monkeypatch):
    monkeypatch.setattr(module, "_gpu_usage", lambda: [])
    monkeypatch.setattr(module, "hardware_identity", lambda: SimpleNamespace(
        cpu_name="Apple M5 Max", cpu_cores=18, gpu_names=("Apple M5 Max",), gpu_cores=40,
        memory_bandwidth_bytes_per_second=614_000_000_000))
    sample = ResourceMonitor()._host_sample()
    assert sample["gpus"] == [{"name": "Apple M5 Max", "core_count": 40, "utilization_percent": None}]
    assert sample["memory_bandwidth_bytes_per_second"] is None


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


@pytest.mark.parametrize("error", [TimeoutError("busy"), RuntimeError("status unavailable")])
def test_unavailable_runtime_counters_keep_loaded_models_without_invented_rates(monkeypatch, error):
    async def run():
        instance = SimpleNamespace(id="instance-a", model="busy-model", state="busy")
        monitor = ResourceMonitor()
        async def instances():
            return SimpleNamespace(data=[instance])
        async def status(identifier):
            assert identifier == instance.id
            raise error
        def host():
            monitor._sample_time = module.time.monotonic()
            return {"sampled_at": 100, "cpu_utilization_percent": None, "gpus": [], "disks": []}
        monkeypatch.setattr(monitor, "_host_sample", host)
        result = await monitor.sample(SimpleNamespace(runtime_instances=instances, runtime_status=status))
        assert result["weights"] == [{"instance_id": "instance-a", "model": "busy-model",
            "expert_read_bytes_per_second": None, "ple_read_bytes_per_second": None,
            "engram_read_bytes_per_second": None, "kv_read_bytes_per_second": None,
            "kv_write_bytes_per_second": None}]
    asyncio.run(run())


def test_slow_runtime_status_is_coalesced_for_concurrent_resource_readers(monkeypatch):
    async def run():
        monitor = ResourceMonitor()
        calls = []
        async def instances():
            return SimpleNamespace(data=[SimpleNamespace(id="instance-a", model="busy-model", state="busy")])
        async def status(identifier):
            calls.append(identifier)
            await asyncio.sleep(2)
        def host():
            monitor._sample_time = module.time.monotonic()
            return {"sampled_at": 100, "cpu_utilization_percent": None, "gpus": [], "disks": []}
        monkeypatch.setattr(monitor, "_host_sample", host)
        service = SimpleNamespace(runtime_instances=instances, runtime_status=status)
        results = await asyncio.gather(*(monitor.sample(service) for _ in range(3)))
        assert calls == ["instance-a"]
        assert all(result is results[0] for result in results)
        assert results[0]["weights"][0]["ple_read_bytes_per_second"] is None
    asyncio.run(run())
