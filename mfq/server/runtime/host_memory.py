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


@dataclass(frozen=True)
class HostMemorySnapshot:
    """Stable leading counters from macOS ``vm_statistics64`` in bytes."""

    total: int
    free: int
    active: int
    inactive: int
    wired: int

    def reclaimable(self, *, active_ratio: float = 0.5) -> int:
        ratio = min(1.0, max(0.0, active_ratio))
        return max(0, self.free) + max(0, self.inactive) + int(
            max(0, self.active) * ratio
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


def host_memory_snapshot() -> HostMemorySnapshot | None:
    """Read macOS VM counters through the sub-microsecond Mach host call."""

    if _MACH_HOST is None:
        return None
    libc, host, page_size = _MACH_HOST
    try:
        stats = (ctypes.c_int * _HOST_INFO64_MAX_COUNT)()
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
        return HostMemorySnapshot(
            total=total,
            free=max(0, int(stats[0])) * page_size,
            active=max(0, int(stats[1])) * page_size,
            inactive=max(0, int(stats[2])) * page_size,
            wired=max(0, int(stats[3])) * page_size,
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
