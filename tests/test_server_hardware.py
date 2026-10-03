import json
import plistlib
import subprocess

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
    assert hardware.hardware_identity() is first
    assert len(calls) == 2


def test_windows_cpu_gpu_and_ram_names(monkeypatch):
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
