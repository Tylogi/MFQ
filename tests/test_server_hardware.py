import ctypes
import json
import plistlib
import subprocess
from types import SimpleNamespace

import pytest

from mfq.server.services import hardware, hub


@pytest.fixture(autouse=True)
def clear_cache():
    hardware.hardware_identity.cache_clear()
    yield
    hardware.hardware_identity.cache_clear()


def test_apple_identity_is_cached_without_inference_imports(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    calls = []

    def command(argv):
        calls.append(argv)
        if argv[0] == "sysctl":
            return "Apple M5 Max\n18"
        return plistlib.dumps([{"model": "Apple M5 Max", "gpu-core-count": 40}]).decode()

    monkeypatch.setattr(hardware, "_command", command)
    first = hardware.hardware_identity()
    assert first.cpu_name == "Apple M5 Max"
    assert first.cpu_cores == 18
    assert first.gpu_names == ("Apple M5 Max",)
    assert first.gpu_cores == 40
    assert first.unified_memory is True
    assert first.memory_bandwidth_bytes_per_second == 614_000_000_000
    assert hardware.hardware_identity() is first
    assert len(calls) == 2


def test_windows_cpu_gpu_and_ram_names(monkeypatch):
    monkeypatch.setattr(hardware, "_nvidia_memory", lambda: ())
    monkeypatch.setattr(hardware, "_amd_memory", lambda: ())
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware, "_command", lambda argv: json.dumps({
        "cpu_name": "AMD Ryzen 5 9600X", "cpu_cores": 6,
        "gpu_names": ["NVIDIA GeForce RTX 5090"], "physical_memory_bytes": 64 << 30,
    }))
    value = hardware.hardware_identity()
    assert value.cpu_name == "AMD Ryzen 5 9600X"
    assert value.gpu_names == ("NVIDIA GeForce RTX 5090",)
    assert value.physical_memory_bytes == 64 << 30


def test_linux_physical_cores_not_smt_threads_and_multiple_gpus(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware.Path, "read_text", lambda self: "\n\n".join(
        f"model name : AMD Ryzen 5\nphysical id : 0\ncore id : {core}"
        for core in [0, 0, 1, 1]
    ))
    monkeypatch.setattr(hardware, "_command", lambda argv: "NVIDIA GeForce RTX 5090\nNVIDIA GeForce RTX 5090")
    value = hardware.hardware_identity()
    assert value.cpu_name == "AMD Ryzen 5"
    assert value.cpu_cores == 2
    assert len(value.gpu_names) == 2


@pytest.mark.parametrize("output", ["", "<plist><array>", plistlib.dumps({"unexpected": True}).decode()])
def test_missing_commands_leave_unavailable_fields_unknown(monkeypatch, output):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware, "_command", lambda argv: output)
    value = hardware.hardware_identity()
    assert value.gpu_names == () and value.gpu_cores is None


def test_rocm_gpu_identity_when_nvidia_is_absent(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware.Path, "read_text", lambda self: "model name : AMD Ryzen 9")
    monkeypatch.setattr(hardware, "_command", lambda argv: json.dumps({"card0": {"Card series": "AMD Instinct MI300X"}}) if argv[0] == "rocm-smi" else "")
    assert hardware.hardware_identity().gpu_names == ("AMD Instinct MI300X",)


def test_probe_timeout_is_bounded_and_not_exposed(monkeypatch):
    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 1.0
        raise subprocess.TimeoutExpired(args[0], 1.0)

    monkeypatch.setattr(hardware.subprocess, "run", timeout)
    assert hardware._command(["nvidia-smi"]) == ""


@pytest.mark.parametrize("name,cores,unified,bandwidth", [
    ("Apple M5 Max", 32, True, 460_000_000_000),
    ("Apple M5 Max", 40, True, 614_000_000_000),
    ("Apple M5 Max", None, True, None),
    ("Apple M4 Max", 40, True, None),
    ("Intel Core i9", 40, False, None),
])
def test_apple_bandwidth_requires_the_exact_chip_bin(monkeypatch, name, cores, unified, bandwidth):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware, "_command", lambda argv: f"{name}\n18" if argv[0] == "sysctl" else plistlib.dumps([
        {"model": name, "gpu-core-count": cores} if cores else {"model": name},
    ]).decode())
    value = hardware.hardware_identity()
    assert value.unified_memory is unified
    assert value.memory_bandwidth_bytes_per_second == bandwidth


@pytest.mark.parametrize("name,unified", [("NVIDIA GeForce RTX 5090", False), ("NVIDIA GB10", True)])
def test_nvml_reads_per_device_capacity_and_peak_bandwidth_without_cuda_context(monkeypatch, name, unified):
    calls = []
    library = _nvml_fixture(name, calls)
    monkeypatch.setattr(ctypes, "CDLL", lambda path: library)
    value = hardware._nvidia_memory()
    assert [item.capacity_bytes for item in value] == [32 << 30, 24 << 30]
    assert all(item.unified is unified for item in value)
    assert all(item.bandwidth_bytes_per_second == (273_000_000_000 if unified else 1_792_000_000_000) for item in value)
    assert calls == ["init", "shutdown"]


def test_nvml_unsupported_optional_queries_do_not_discard_capacity(monkeypatch):
    library = _nvml_fixture("NVIDIA GPU", [])
    del library.nvmlDeviceGetMaxClockInfo, library.nvmlDeviceGetMemoryBusWidth
    monkeypatch.setattr(ctypes, "CDLL", lambda path: library)
    assert hardware._nvidia_memory() == (hardware.GpuMemory("NVIDIA GPU", 32 << 30), hardware.GpuMemory("NVIDIA GPU", 24 << 30))


@pytest.mark.parametrize("gpus,architecture,capacity,hot", [
    ((hardware.GpuMemory("NVIDIA GB10", 128 << 30, unified=True),), "unified", None, "ram"),
    ((hardware.GpuMemory("RTX 3090 Ti", 24 << 30),), "discrete", 24 << 30, "vram"),
    ((hardware.GpuMemory("integrated CUDA GPU", 64 << 30, unified=True),), "unified", None, "ram"),
    ((hardware.GpuMemory("unknown discrete GPU"),), "discrete", None, "vram"),
    ((), "unknown", None, None),
])
def test_runtime_topology_uses_physical_memory_instead_of_backend(monkeypatch, gpus, architecture, capacity, hot):
    monkeypatch.setattr(hardware, "hardware_identity", lambda: hardware.HardwareIdentity(
        "CPU", 16, tuple(gpu.name for gpu in gpus), physical_memory_bytes=128 << 30, gpu_memory=gpus))
    value = hardware.memory_topology()
    assert value["memory_architecture"] == architecture
    assert value["device_memory_total_bytes"] == capacity
    assert value["prefix_cache_hot_tier"] == hot
    assert value["prefix_cache_cold_tier"] == "ssd"
    assert value["host_memory_total_bytes"] == 128 << 30


def test_gb10_keeps_unified_topology_when_capacity_queries_are_unavailable(monkeypatch):
    monkeypatch.setattr(hardware, "hardware_identity", lambda: hardware.HardwareIdentity(
        "CPU", 20, ("NVIDIA GB10",), physical_memory_bytes=128 << 30))
    value = hardware.memory_topology()
    assert value["memory_architecture"] == "unified" and value["device_memory_total_bytes"] is None
    assert value["prefix_cache_hot_tier"] == "ram"


def _nvml_fixture(name, calls):
    def initialize():
        calls.append("init")
        return 0

    def assign(output, value):
        output._obj.value = value
        return 0

    def query_name(handle, buffer, size):
        buffer.value = name.encode()
        return 0

    def query_memory(handle, memory):
        memory._obj.total = (32 if handle.value == 1 else 24) << 30
        return 0

    def query_clock(handle, kind, clock):
        assert kind.value == 2
        return assign(clock, 14000)

    return SimpleNamespace(
        nvmlInit_v2=initialize, nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetCount_v2=lambda count: assign(count, 2),
        nvmlDeviceGetHandleByIndex_v2=lambda index, handle: assign(handle, index.value + 1),
        nvmlDeviceGetName=query_name, nvmlDeviceGetMemoryInfo=query_memory,
        nvmlDeviceGetMaxClockInfo=query_clock,
        nvmlDeviceGetMemoryBusWidth=lambda handle, width: assign(width, 512),
    )


@pytest.mark.parametrize("rate", [{"value": 5325, "unit": "GB/s"}, "5325 GB/s", {"value": "N/A", "unit": "GB/s"}])
def test_amd_smi_units_and_unavailable_bandwidth(monkeypatch, rate):
    monkeypatch.setattr(hardware, "_command", lambda argv: json.dumps([{
        "asic": {"market_name": "AMD Instinct MI300X"},
        "vram": {"size": {"value": 196608, "unit": "MB"}, "max_bandwidth": rate},
    }]))
    value = hardware._amd_memory()[0]
    assert value.capacity_bytes == 192 << 30
    assert value.bandwidth_bytes_per_second == (None if isinstance(rate, dict) and rate["value"] == "N/A" else 5_325_000_000_000)
    assert value.unified is False


def test_cli_fallback_reads_vram_without_a_driver_library(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware, "_nvidia_memory", lambda: ())
    monkeypatch.setattr(hardware, "_amd_memory", lambda: ())
    monkeypatch.setattr(hardware, "_command", lambda argv: "NVIDIA RTX 5090, 32768\nNVIDIA RTX 4090, 24576")
    value = hardware.hardware_identity()
    assert value.gpu_names == ("NVIDIA RTX 5090", "NVIDIA RTX 4090")
    assert [item.capacity_bytes for item in value.gpu_memory] == [32 << 30, 24 << 30]


@pytest.mark.parametrize("unified", [False, True])
def test_hub_memory_pools_do_not_double_count_shared_memory_or_pool_multiple_gpus(monkeypatch, unified):
    monkeypatch.setattr(hub, "hardware_identity", lambda: hardware.HardwareIdentity(
        "Test CPU", 18, ("GPU 1", "GPU 2"), physical_memory_bytes=128 << 30,
        gpu_memory=(hardware.GpuMemory("GPU 1", 32 << 30, 1_792_000_000_000, unified),
                    hardware.GpuMemory("GPU 2", 24 << 30, 1_008_000_000_000)),
    ))
    monkeypatch.setattr(hub, "host_memory_snapshot", lambda: None)
    monkeypatch.setattr(hub, "total_physical_memory", lambda: None)
    value = hub.system_profile(backend="cuda", runtime_memory_budget_bytes=96 << 30)
    assert value.runtime_memory_budget_bytes == 96 << 30
    assert [item.kind for item in value.memory_pools] == (["vram", "uma"] if unified else ["vram", "vram", "ram"])
    assert value.memory_pools[-1].capacity_bytes == 128 << 30
    assert value.memory_pools[-1].bandwidth_bytes_per_second == (1_792_000_000_000 if unified else None)


@pytest.mark.parametrize("backend", ["metal", "cuda", "rocm"])
def test_hub_contract_preserves_hardware_and_backend_vendor(monkeypatch, backend):
    monkeypatch.setattr(hub, "hardware_identity", lambda: hardware.HardwareIdentity("Test CPU", 18, ("Test GPU",), 40, 128 << 30))
    monkeypatch.setattr(hub, "host_memory_snapshot", lambda: None)
    monkeypatch.setattr(hub, "total_physical_memory", lambda: None)
    monkeypatch.setattr(hub, "metal_recommended_working_set_size", lambda: None)
    value = hub.system_profile(backend=backend, runtime_memory_budget_bytes=96 << 30)
    assert value.backend == backend
    assert value.cpu_name == "Test CPU" and value.cpu_cores == 18
    assert value.gpu_names == ["Test GPU"] and value.gpu_cores == 40
    assert value.physical_memory_bytes == 128 << 30
