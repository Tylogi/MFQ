"""Low-overhead host-memory telemetry for runtime admission."""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import os
import sys
from dataclasses import dataclass

_HOST_VM_INFO64 = 4
_HOST_INFO64_MAX_COUNT = 256
_VM_STATS_MIN_COUNT = 4


class _VmStatistics64(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint32)
        for name in ("free", "active", "inactive", "wired")
    ] + [
        (name, ctypes.c_uint64)
        for name in (
            "zero_fill", "reactivations", "pageins", "pageouts", "faults",
            "cow_faults", "lookups", "hits", "purges",
        )
    ] + [
        (name, ctypes.c_uint32)
        for name in ("purgeable", "speculative")
    ] + [
        (name, ctypes.c_uint64)
        for name in ("decompressions", "compressions", "swapins", "swapouts")
    ] + [(name, ctypes.c_uint32) for name in (
        "compressor_pages", "throttled_pages", "external_pages", "internal_pages")]


class _RusageInfoV0(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "user_time", "system_time", "idle_wakeups", "interrupt_wakeups", "pageins",
            "wired_size", "resident_size", "phys_footprint", "start_time", "exit_time")]


def _open_process_usage():
    if sys.platform != "darwin":
        return None
    try:
        function = ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage
        function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        function.restype = ctypes.c_int
        return function
    except (AttributeError, OSError):
        return None


_PROCESS_USAGE = _open_process_usage()


def process_physical_footprint(pid: int) -> int | None:
    if _PROCESS_USAGE is None or pid <= 0:
        return None
    value = _RusageInfoV0()
    if _PROCESS_USAGE(pid, 0, ctypes.byref(value)) != 0:
        return None
    return int(value.phys_footprint)


def automatic_memory_reserve(total: int) -> int:
    return min(8 << 30, max(3 << 30, total * 8 // 100))


@dataclass(frozen=True)
class HostMemorySnapshot:
    """Stable leading counters from macOS ``vm_statistics64`` in bytes."""

    total: int
    free: int
    active: int
    inactive: int
    wired: int
    compression_bytes: int | None = None
    swapout_bytes: int | None = None
    external: int = 0

    def reclaimable(self, *, active_ratio: float = 0.5) -> int:
        ratio = min(1.0, max(0.0, active_ratio))
        file_active = max(0, self.external - max(0, self.inactive))
        return max(0, self.free) + max(0, self.inactive, self.external) + int(
            max(0, self.active - file_active) * ratio
        )


def total_physical_memory() -> int | None:
    """Return total physical memory without starting a helper process."""

    try:
        total = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, OverflowError, TypeError, ValueError):
        return None
    return total if total > 0 else None


@functools.lru_cache(maxsize=1)
def metal_recommended_working_set_size() -> int | None:
    """Return Metal's process working-set ceiling when MLX exposes it."""

    if sys.platform != "darwin":
        return None
    try:
        import mlx.core as mx

        value = mx.device_info().get("max_recommended_working_set_size")
        result = int(value)
    except (
        AttributeError,
        ImportError,
        KeyError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None
    return result if result > 0 else None


def _open_mach_host() -> tuple[ctypes.CDLL, int, int] | None:
    if sys.platform != "darwin":
        return None
    try:
        library = ctypes.util.find_library("c")
        if library is None:
            return None
        libc = ctypes.CDLL(library)
        libc.mach_host_self.restype = ctypes.c_uint
        host = int(libc.mach_host_self())
        page_size = ctypes.c_uint(0)
        if libc.host_page_size(host, ctypes.byref(page_size)) != 0:
            return None
        if host <= 0 or page_size.value <= 0:
            return None
        return libc, host, int(page_size.value)
    except (
        AttributeError,
        ctypes.ArgumentError,
        OSError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        return None


_MACH_HOST = _open_mach_host()


def metal_allocation_limit() -> int | None:
    if _MACH_HOST is not None:
        value, size = ctypes.c_int64(), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int64))
        try:
            function = _MACH_HOST[0].sysctlbyname
            if function(b"iogpu.wired_limit_mb", ctypes.byref(value), ctypes.byref(size), None, 0) == 0 and value.value > 0:
                return value.value << 20
        except (AttributeError, ctypes.ArgumentError, OSError):
            pass
    return metal_recommended_working_set_size()


def host_memory_snapshot() -> HostMemorySnapshot | None:
    """Read macOS VM counters through the sub-microsecond Mach host call."""

    if _MACH_HOST is None:
        return None
    libc, host, page_size = _MACH_HOST
    try:
        stats = (ctypes.c_uint32 * _HOST_INFO64_MAX_COUNT)()
        count = ctypes.c_uint(_HOST_INFO64_MAX_COUNT)
        result = libc.host_statistics64(
            host,
            _HOST_VM_INFO64,
            stats,
            ctypes.byref(count),
        )
        if result != 0 or count.value < _VM_STATS_MIN_COUNT:
            return None
        total = total_physical_memory()
        if total is None:
            total = sum(max(0, int(stats[index])) for index in range(4)) * page_size
        extended = (
            _VmStatistics64.from_buffer(stats)
            if count.value * ctypes.sizeof(ctypes.c_uint32) >= _VmStatistics64.swapouts.offset + 8
            else None
        )
        return HostMemorySnapshot(
            total=total,
            free=max(0, int(stats[0])) * page_size,
            active=max(0, int(stats[1])) * page_size,
            inactive=max(0, int(stats[2])) * page_size,
            wired=max(0, int(stats[3])) * page_size,
            compression_bytes=int(extended.compressions) * page_size if extended is not None else None,
            swapout_bytes=int(extended.swapouts) * page_size if extended is not None else None,
            external=int(extended.external_pages) * page_size if extended is not None
                and count.value * ctypes.sizeof(ctypes.c_uint32) >= _VmStatistics64.external_pages.offset + 4 else 0,
        )
    except (
        AttributeError,
        ctypes.ArgumentError,
        OSError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        return None
