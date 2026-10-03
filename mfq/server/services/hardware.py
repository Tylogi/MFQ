from __future__ import annotations

import functools
import json
import os
import platform
import plistlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.parsers.expat import ExpatError


@dataclass(frozen=True)
class HardwareIdentity:
    cpu_name: str | None
    cpu_cores: int | None
    gpu_names: tuple[str, ...] = ()
    gpu_cores: int | None = None
    physical_memory_bytes: int | None = None


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
        return HardwareIdentity(cpu_name, cpu_cores, names, cores)
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
                return HardwareIdentity(
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
    names = _command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines()
    if not names:
        try:
            info = json.loads(_command(["rocm-smi", "--showproductname", "--json"]))
            if isinstance(info, dict):
                names = [item.get("Card series") or item.get("Card model") for item in info.values() if isinstance(item, dict)]
        except ValueError:
            pass
    return HardwareIdentity(cpu_name, cpu_cores, tuple(name.strip() for name in names if isinstance(name, str) and name.strip()))
