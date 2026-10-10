from __future__ import annotations

import ctypes
import functools
import json
import os
import platform
import plistlib
import re
import subprocess
import psutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from xml.parsers.expat import ExpatError


def _unified_gpu_name(name: str) -> bool:
    return bool(re.search(r"\bGB10\b|\bMI300A\b|\bRadeon\s+(?:80[456]0S|8065S)\b", name, re.IGNORECASE))


@dataclass(frozen=True)
class GpuMemory:
    name: str
    capacity_bytes: int | None = None
    bandwidth_bytes_per_second: int | None = None
    unified: bool = False


@dataclass(frozen=True)
class HardwareIdentity:
    cpu_name: str | None
    cpu_cores: int | None
    gpu_names: tuple[str, ...] = ()
    gpu_cores: int | None = None
    physical_memory_bytes: int | None = None
    unified_memory: bool = False
    memory_bandwidth_bytes_per_second: int | None = None
    gpu_memory: tuple[GpuMemory, ...] = ()

    @property
    def memory_architecture(self) -> str:
        # CUDA managed allocations do not imply physically shared memory.
        # GB10 and other UMA devices are identified independently of backend.
        if self.gpu_memory:
            return "unified" if all(item.unified or _unified_gpu_name(item.name) for item in self.gpu_memory) else "discrete"
        if self.unified_memory:
            return "unified"
        if self.gpu_names:
            return "unified" if all(_unified_gpu_name(name) for name in self.gpu_names) else "discrete"
        return "unknown"


def memory_topology() -> dict[str, Any]:
    hardware = hardware_identity()
    architecture = hardware.memory_architecture
    capacities = [item.capacity_bytes for item in hardware.gpu_memory if not item.unified and not _unified_gpu_name(item.name)]
    return {
        "memory_architecture": architecture,
        "unified_memory": architecture == "unified" if architecture != "unknown" else None,
        "host_memory_total_bytes": hardware.physical_memory_bytes or host_memory_capacity()[0],
        "device_memory_total_bytes": sum(capacities) if capacities and all(value is not None for value in capacities) else None,
        "prefix_cache_hot_tier": "vram" if architecture == "discrete" else "ram" if architecture == "unified" else None,
        "prefix_cache_cold_tier": "ssd",
    }


def host_memory_capacity() -> tuple[int | None, int | None]:
    try:
        memory = psutil.virtual_memory()
        return int(memory.total), int(memory.available)
    except (OSError, psutil.Error):
        return None, None


def _command(argv: list[str]) -> str:
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, check=True, timeout=1.0
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return ""


def _positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result > 0 else None


def _nvidia_memory() -> tuple[GpuMemory, ...]:
    class MemoryInfo(ctypes.Structure):
        _fields_ = [(key, ctypes.c_ulonglong) for key in ("total", "free", "used")]

    try:
        library = ctypes.CDLL("nvml.dll" if platform.system() == "Windows" else "libnvidia-ml.so.1")
        if library.nvmlInit_v2() != 0:
            return ()
    except (AttributeError, OSError):
        return ()
    try:
        count = ctypes.c_uint()
        if library.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            return ()
        devices = []
        for index in range(count.value):
            handle = ctypes.c_void_p()
            if library.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(handle)) != 0:
                continue
            name = ctypes.create_string_buffer(96)
            if library.nvmlDeviceGetName(handle, name, ctypes.c_uint(len(name))) != 0:
                continue
            label = name.value.decode()
            memory = MemoryInfo()
            capacity = memory.total if library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) == 0 else None
            clock, width = ctypes.c_uint(), ctypes.c_uint()
            bandwidth = None
            clock_query = getattr(library, "nvmlDeviceGetMaxClockInfo", None)
            width_query = getattr(library, "nvmlDeviceGetMemoryBusWidth", None)
            if (clock_query is not None and width_query is not None
                    and clock_query(handle, ctypes.c_uint(2), ctypes.byref(clock)) == 0
                    and width_query(handle, ctypes.byref(width)) == 0):
                bandwidth = _positive_int(clock.value * 1_000_000 * 2 * width.value // 8)
            unified = bool(re.search(r"\bGB10\b", label, re.IGNORECASE))
            devices.append(GpuMemory(label, _positive_int(capacity), 273_000_000_000 if unified else bandwidth, unified))
        return tuple(devices)
    except (AttributeError, OSError, UnicodeError, ValueError):
        return ()
    finally:
        library.nvmlShutdown()


def _amd_memory() -> tuple[GpuMemory, ...]:
    try:
        info = json.loads(_command(["amd-smi", "static", "--asic", "--vram", "--json"]))
        if isinstance(info, dict):
            info = [info]
        devices = []
        for item in info if isinstance(info, list) else []:
            if not isinstance(item, dict):
                continue
            asic, vram = item.get("asic", {}), item.get("vram", {})
            if not isinstance(asic, dict) or not isinstance(vram, dict):
                continue
            name = asic.get("market_name")
            if not isinstance(name, str) or not name:
                continue
            size, rate = vram.get("size"), vram.get("max_bandwidth")
            if isinstance(size, dict):
                size = f"{size.get('value')} {size.get('unit')}"
            if isinstance(rate, dict):
                rate = f"{rate.get('value')} {rate.get('unit')}"
            size = re.fullmatch(r"(\d+)\s+MB", str(size))
            rate = re.fullmatch(r"(\d+(?:\.\d+)?)\s+GB/s", str(rate))
            capacity = int(size[1]) * (1 << 20) if size else None
            bandwidth = int(float(rate[1]) * 1_000_000_000) if rate else None
            unified = bool(re.search(r"\bMI300A\b|\bRadeon\s+(?:80[456]0S|8065S)\b", name, re.IGNORECASE))
            devices.append(GpuMemory(name, _positive_int(capacity), _positive_int(bandwidth), unified))
        return tuple(devices)
    except (ValueError, TypeError, OverflowError):
        return ()


@functools.lru_cache(maxsize=1)
def hardware_identity() -> HardwareIdentity:
    system = platform.system()
    cpu_name = platform.processor() or None
    cpu_cores = os.cpu_count()
    if system == "Darwin":
        cpu = _command(["sysctl", "-n", "machdep.cpu.brand_string", "hw.physicalcpu"]).splitlines()
        if len(cpu) == 2:
            cpu_name, cpu_cores = cpu[0], _positive_int(cpu[1])
        try:
            devices = plistlib.loads(_command(["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1", "-a"]).encode())
        except (ValueError, plistlib.InvalidFileException, ExpatError):
            devices = []
        devices = [item for item in devices if isinstance(item, dict)] if isinstance(devices, list) else []
        names = tuple(item["model"] for item in devices if isinstance(item.get("model"), str))
        cores = _positive_int(devices[0].get("gpu-core-count")) if devices else None
        unified = bool(cpu_name and re.fullmatch(r"Apple M\d+(?: (?:Pro|Max|Ultra))?", cpu_name))
        bandwidth = {32: 460_000_000_000, 40: 614_000_000_000}.get(cores) if cpu_name == "Apple M5 Max" else None
        return HardwareIdentity(cpu_name, cpu_cores, names, cores, unified_memory=unified,
                                memory_bandwidth_bytes_per_second=bandwidth)
    identity = HardwareIdentity(cpu_name, cpu_cores)
    if system == "Windows":
        script = (
            "$cpu = @(Get-CimInstance Win32_Processor); "
            "@{cpu_name = ($cpu.Name -join ' + '); "
            "cpu_cores = ($cpu | Measure-Object NumberOfCores -Sum).Sum; "
            "physical_memory_bytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory; "
            "gpu_names = @(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name)} "
            "| ConvertTo-Json -Compress"
        )
        try:
            info = json.loads(_command(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script]))
            if isinstance(info, dict):
                names = info.get("gpu_names", [])
                identity = HardwareIdentity(
                    info.get("cpu_name") or cpu_name,
                    _positive_int(info.get("cpu_cores")) or cpu_cores,
                    tuple(item for item in names if isinstance(item, str)) if isinstance(names, list) else (),
                    physical_memory_bytes=_positive_int(info.get("physical_memory_bytes")),
                )
        except (ValueError, TypeError):
            pass
    elif system == "Linux":
        try:
            cpu_info = Path("/proc/cpuinfo").read_text()
            name = re.search(r"^(?:model name|Hardware)\s*:\s*(.+)$", cpu_info, re.MULTILINE)
            if name:
                cpu_name = name[1].strip()
            cores = set()
            for block in cpu_info.split("\n\n"):
                fields = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
                fields = {key.strip(): value.strip() for key, value in fields.items()}
                if "physical id" in fields and "core id" in fields:
                    cores.add((fields["physical id"], fields["core id"]))
            if cores:
                cpu_cores = len(cores)
        except (OSError, UnicodeError):
            pass
        identity = HardwareIdentity(cpu_name, cpu_cores)
    gpu_memory = _nvidia_memory() or _amd_memory()
    names = [item.name for item in gpu_memory]
    if not names and not identity.gpu_names:
        names = _command(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]).splitlines()
        parsed = []
        for row in names:
            name, _, capacity = row.rpartition(",")
            if name.strip() and _positive_int(capacity.strip()):
                parsed.append(GpuMemory(name.strip(), int(capacity.strip()) << 20,
                                        unified=bool(re.search(r"\bGB10\b", name, re.IGNORECASE))))
            elif row.strip():
                parsed.append(GpuMemory(row.strip()))
        gpu_memory = tuple(parsed)
        names = [item.name for item in gpu_memory]
    if not names and not identity.gpu_names:
        try:
            info = json.loads(_command(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"]))
            if isinstance(info, dict):
                gpu_memory = tuple(GpuMemory(
                    item.get("Card series") or item.get("Card model"),
                    _positive_int(item.get("VRAM Total Memory (B)")),
                    unified=bool(re.search(r"\bMI300A\b|\bRadeon\s+(?:80[456]0S|8065S)\b",
                                           item.get("Card series") or item.get("Card model") or "", re.IGNORECASE)),
                ) for item in info.values() if isinstance(item, dict) and isinstance(item.get("Card series") or item.get("Card model"), str))
                names = [item.name for item in gpu_memory]
        except ValueError:
            pass
    gpu_memory = tuple(replace(item, unified=True) if _unified_gpu_name(item.name) else item for item in gpu_memory)
    return replace(identity, gpu_names=tuple(names) or identity.gpu_names, gpu_memory=gpu_memory)
