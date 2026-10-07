from __future__ import annotations

import asyncio
import math
import platform
import plistlib
import time
from pathlib import Path
from typing import Any
from xml.parsers.expat import ExpatError

import psutil

from mfq.server.services.hardware import _command, _positive_int, hardware_identity

_MAX_SAMPLE_INTERVAL_SECONDS = 10.0


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def _percent(value: Any) -> float | None:
    value = _number(value)
    return min(100.0, value) if value is not None else None


def _gpu_usage() -> list[dict[str, Any]]:
    if platform.system() == "Darwin":
        try:
            devices = plistlib.loads(_command(["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1", "-a"]).encode())
            result = []
            for item in devices if isinstance(devices, list) else []:
                if not isinstance(item, dict):
                    continue
                statistics = item.get("PerformanceStatistics")
                name = item.get("model")
                result.append({"name": name if isinstance(name, str) and name.strip() else "GPU",
                    "core_count": _positive_int(item.get("gpu-core-count")),
                    "utilization_percent": _percent(statistics.get("Device Utilization %"))
                        if isinstance(statistics, dict) else None})
            return result
        except (ValueError, plistlib.InvalidFileException, ExpatError):
            return []
    if platform.system() == "Linux":
        devices = []
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
            try:
                value = float((card / "device/gpu_busy_percent").read_text().strip())
            except (OSError, ValueError):
                continue
            devices.append({"name": card.name, "utilization_percent": _percent(value)})
        if devices:
            return devices
    rows = _command(["nvidia-smi", "--query-gpu=name,utilization.gpu", "--format=csv,noheader,nounits"])
    devices = []
    for row in rows.splitlines():
        name, _, value = row.rpartition(",")
        try:
            utilization = _percent(float(value.strip()))
        except ValueError:
            utilization = None
        if name.strip():
            devices.append({"name": name.strip(), "utilization_percent": utilization})
    if not devices:
        devices = [{"name": name, "utilization_percent": None} for name in hardware_identity().gpu_names]
    return devices


class ResourceMonitor:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._cached: dict[str, Any] | None = None
        self._cached_at = 0.0
        self._sample_time = 0.0
        self._cpu: tuple[float, float] | None = None
        self._disks: dict[str, tuple[float, float, float | None]] = {}
        self._weights: dict[str, tuple[float, dict[str, float]]] = {}
        self._physical_disks: set[str] | None = None
        self._disk_topology_time = 0.0

    def _physical_disk_names(self, now: float) -> set[str] | None:
        if platform.system() != "Darwin":
            return None
        if self._physical_disks is not None and now - self._disk_topology_time < 30:
            return self._physical_disks
        self._disk_topology_time = now
        try:
            topology = plistlib.loads(_command(["diskutil", "list", "-plist", "physical"]).encode())
            names = topology.get("WholeDisks") if isinstance(topology, dict) else None
            if isinstance(names, list):
                self._physical_disks = {name for name in names if isinstance(name, str)}
        except (ValueError, plistlib.InvalidFileException, ExpatError):
            pass
        if self._physical_disks is None:
            self._physical_disks = set()
        return self._physical_disks

    def _host_sample(self) -> dict[str, Any]:
        now = time.monotonic()
        elapsed = now - self._sample_time if self._sample_time else None
        if elapsed is not None and not 0 < elapsed <= _MAX_SAMPLE_INTERVAL_SECONDS:
            elapsed = None
        cpu = None
        try:
            times = psutil.cpu_times()._asdict()
            total = sum(times.values()) - times.get("guest", 0) - times.get("guest_nice", 0)
            idle = times.get("idle", 0) + times.get("iowait", 0)
            if elapsed is not None and self._cpu is not None and total > self._cpu[0] and idle >= self._cpu[1]:
                cpu = _percent(100 * (1 - (idle - self._cpu[1]) / (total - self._cpu[0])))
            self._cpu = (total, idle)
        except (OSError, psutil.Error):
            self._cpu = None
        disks = []
        current = {}
        try:
            counters = psutil.disk_io_counters(perdisk=True, nowrap=False) or {}
            physical = self._physical_disk_names(now)
            for name, item in counters.items():
                if physical is not None and name not in physical:
                    continue
                read, write = float(item.read_bytes), float(item.write_bytes)
                busy = _number(getattr(item, "busy_time", None))
                previous = self._disks.get(name)
                read_rate = write_rate = busy_percent = None
                if elapsed and previous is not None:
                    if read >= previous[0]:
                        read_rate = (read - previous[0]) / elapsed
                    if write >= previous[1]:
                        write_rate = (write - previous[1]) / elapsed
                    if busy is not None and previous[2] is not None and busy >= previous[2]:
                        busy_percent = _percent((busy - previous[2]) / (elapsed * 10))
                disks.append({"name": name, "read_bytes_per_second": read_rate,
                              "write_bytes_per_second": write_rate, "busy_percent": busy_percent,
                              "bandwidth_utilization_percent": None})
                current[name] = (read, write, busy)
        except (OSError, psutil.Error):
            pass
        self._disks = current
        self._sample_time = now
        hardware = hardware_identity()
        gpus = _gpu_usage()
        if not gpus:
            gpus = [{"name": name, "core_count": hardware.gpu_cores if len(hardware.gpu_names) == 1 else None,
                     "utilization_percent": None} for name in hardware.gpu_names]
        return {"sampled_at": time.time(), "interval_seconds": elapsed,
                "cpu_name": hardware.cpu_name, "cpu_cores": hardware.cpu_cores,
                "cpu_utilization_percent": cpu, "gpus": gpus, "disks": disks,
                "memory_bandwidth_bytes_per_second": None,
                "memory_bandwidth_limit_bytes_per_second": hardware.memory_bandwidth_bytes_per_second,
                "memory_bandwidth_utilization_percent": None}

    def _weight_sample(self, statuses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = time.monotonic()
        result = []
        current = {}
        for status in statuses:
            key = str(status.get("instance_id") or status.get("model"))
            counters = {}
            for label, field in (("experts", "ssd_expert_bytes_read"), ("ple", "ple_source_bytes_read"), ("engram", "engram_bytes_read")):
                value = _number(status.get(field))
                if value is not None:
                    counters[label] = value
                elif status.get({"experts": "ssd_expert_enabled", "ple": "ssd_ple_enabled"}.get(label, "")) == 0:
                    counters[label] = 0.0
            previous = self._weights.get(key)
            rates = {}
            for label, value in counters.items():
                if previous and label in previous[1] and value >= previous[1][label] and 0 < now - previous[0] <= _MAX_SAMPLE_INTERVAL_SECONDS:
                    rates[label] = (value - previous[1][label]) / (now - previous[0])
            current[key] = (now, counters)
            result.append({"instance_id": key, "model": status.get("model", key),
                           "expert_read_bytes_per_second": rates.get("experts"),
                           "ple_read_bytes_per_second": rates.get("ple"),
                           "engram_read_bytes_per_second": rates.get("engram")})
        self._weights = current
        return result

    async def sample(self, service: Any) -> dict[str, Any]:
        async with self._lock:
            if self._cached is not None and time.monotonic() - self._cached_at < 1.0:
                return self._cached
            instances = (await service.runtime_instances()).data
            async def status(instance: Any) -> dict[str, Any] | None:
                if instance.state not in {"ready", "busy"}:
                    return None
                try:
                    async with asyncio.timeout(1.0):
                        return await service.runtime_status(instance.id)
                except (TimeoutError, RuntimeError):
                    return {"instance_id": str(instance.id), "model": instance.model}
            host, statuses = await asyncio.gather(
                asyncio.to_thread(self._host_sample),
                asyncio.gather(*(status(instance) for instance in instances)),
            )
            host["weights"] = self._weight_sample([item for item in statuses if item is not None])
            self._cached = host
            self._cached_at = time.monotonic()
            return host
