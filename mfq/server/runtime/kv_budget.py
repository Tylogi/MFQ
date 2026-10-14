"""CUDA L1/L2/L3 capacity planning; byte budgets are independent of live usage."""
from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mfq.server.protocol.models import KvMemorySettings, KvQuantizationSettings
    from mfq.server.state.catalog import DiscoveredModel


def kv_resident_fraction(context: int) -> float:
    """Monotone cubic interpolation through 0/128K/256K/1M presets."""
    x = (0.0, 131072.0, 262144.0, 1048576.0)
    y = (1.0, 0.6, 0.3, 0.1)
    if context <= 0:
        return y[0]
    if context >= x[-1]:
        return y[-1]
    h = tuple(b - a for a, b in zip(x, x[1:]))
    d = tuple((b - a) / step for a, b, step in zip(y, y[1:], h))
    slopes = [(3 * d[0] - d[1]) / 2]
    for i in (1, 2):
        w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
        slopes.append((w1 + w2) / (w1 / d[i - 1] + w2 / d[i]))
    slopes.append(0.0)
    i = next(i for i in range(3) if context <= x[i + 1])
    t = (context - x[i]) / h[i]
    value = ((2*t**3 - 3*t**2 + 1)*y[i] + (t**3 - 2*t**2 + t)*h[i]*slopes[i]
        + (-2*t**3 + 3*t**2)*y[i+1] + (t**3 - t**2)*h[i]*slopes[i+1])
    return max(y[i+1], min(y[i], value))


@dataclass(frozen=True)
class KvTierBudget:
    indexer_bytes: int
    raw_bytes: int
    gpu_budget_bytes: int
    ram_required_bytes: int
    fraction: float


def kv_tier_budget(context: int, indexer_bytes: int, raw_bytes: int,
    gpu_budget_bytes: int | None = None) -> KvTierBudget:
    if min(context, indexer_bytes, raw_bytes) < 0:
        raise ValueError("negative KV capacity")
    fraction = kv_resident_fraction(context)
    budget = indexer_bytes + math.ceil(raw_bytes * fraction) if gpu_budget_bytes is None else gpu_budget_bytes
    if budget < indexer_bytes:
        raise ValueError(f"kv_budget_below_indexer: VRAM KV budget must be at least {indexer_bytes} bytes")
    return KvTierBudget(indexer_bytes, raw_bytes, budget, max(0, indexer_bytes + raw_bytes - budget), fraction)


def checkpoint_kv_budget(artifact: DiscoveredModel, context: int,
    settings: KvMemorySettings, quantization: KvQuantizationSettings) -> KvTierBudget:
    from mfq.server.services.model_memory import checkpoint_cache_profile

    if settings.allocation == "dynamic" and settings.placement not in {"auto", "vram"}:
        raise ValueError("kv_dynamic_unavailable: dynamic VRAM requires GPU-resident KV")
    profile = checkpoint_cache_profile(artifact, cuda=True)
    if profile is None:
        raise ValueError("kv_memory_unsupported: KV capacity metadata is unavailable")
    indexer = sum(item.bytes for item in profile.fixed_components if item.name.startswith("indexer"))
    raw = profile.fixed_bytes - indexer
    for item in profile.components:
        row_bytes = item.bytes_per_row
        if quantization.enabled and item.name == "raw_kv":
            if not item.head_dimension or not item.kv_heads or not item.layers:
                raise ValueError("kv_memory_unsupported: packed KV geometry is unavailable")
            width = 1 << (item.head_dimension - 1).bit_length()
            words = 2 + math.ceil(width * math.floor(quantization.bits) / 32) + math.ceil(width * math.ceil(quantization.bits) / 32)
            row_bytes = item.layers * item.kv_heads * words * 4
        size = math.ceil(context / item.tokens_per_row) * row_bytes
        if item.subgroup == "indexer" or (item.name or "").startswith("indexer"):
            indexer += size
        else:
            raw += size
    budget = settings.gpu_budget_bytes
    if settings.placement == "vram" or settings.allocation == "dynamic":
        budget = indexer + raw if budget is None else budget
        if budget < indexer + raw:
            code = "kv_dynamic_unavailable" if settings.allocation == "dynamic" else "kv_vram_insufficient"
            raise ValueError(f"{code}: GPU-resident KV budget cannot hold the target context")
    elif settings.placement == "ram":
        budget = indexer if budget is None else budget
    plan = kv_tier_budget(context, indexer, raw, budget)
    row = raw // context if context else 0
    required = raw if settings.placement == "ram" else math.ceil(plan.ram_required_bytes / row) * row if row else 0
    return KvTierBudget(indexer, raw, plan.gpu_budget_bytes, required, plan.fraction)


def check_ssd_capacity(directory: Path, required: int) -> None:
    # Check the containing volume without creating/preallocating a spill file.
    existing = directory
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    try:
        free = shutil.disk_usage(existing).free
    except OSError as error:
        raise ValueError(f"kv_ssd_insufficient: cannot inspect SSD KV volume: {error}") from error
    if free < required:
        raise ValueError(f"kv_ssd_insufficient: SSD KV needs {required} bytes; {free} bytes available")
