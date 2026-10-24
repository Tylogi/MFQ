"""Multi-request concurrency policy and queue accounting in the runtime pool."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from mfq.server.protocol.models import (
    ModelArtifactResource,
    ResponsePerformance,
    RuntimeInstanceState,
    SamplingParams,
)
from mfq.server.state.catalog import DiscoveredModel
from mfq.server.runtime.backend import BackendDelta, BackendError
from mfq.server.runtime.native import (
    DEFAULT_RUNTIME_CONCURRENCY,
    MAX_RUNTIME_CONCURRENCY,
    native_request_capacity,
)
from mfq.server.runtime.runtime_pool import RuntimePool, _Runtime


def _artifact(name: str = "model") -> DiscoveredModel:
    return DiscoveredModel(
        resource=ModelArtifactResource(
            id="0" * 32,
            name=name,
            architecture="qwen35",
            format="mfq",
            shard_count=1,
            total_bytes=1,
            tensor_count=1,
            record_count=1,
            complete=True,
            loadable=True,
            modified_at=datetime.now(timezone.utc),
        ),
        path=Path("model.mfq"),
    )


class _FakeProcess:
    returncode: int | None = None
    pid = 4242


class _FakeBackend:
    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.gate = gate

    async def stream(self, **_kwargs: Any):
        if self.gate is not None:
            await self.gate.wait()
        yield BackendDelta(
            content_delta="a",
            performance=ResponsePerformance(
                prefill_tokens=1,
                ttft_ms=1.0,
                prefill_ms=1.0,
                prefill_tps=1.0,
                decode_ms=1.0,
                decode_tps=1.0,
                generation_ms=1.0,
                generation_tps=1.0,
                sampling=SamplingParams(),
            ),
        )

    async def runtime_status(self) -> dict[str, Any]:
        return {}

    async def aclose(self) -> None:
        return None


def _instance(
    *,
    capacity: int,
    backend: _FakeBackend | None = None,
) -> _Runtime:
    return _Runtime(
        id=uuid4(),
        artifact=_artifact(),
        process=_FakeProcess(),
        backend=backend or _FakeBackend(),
        port=43123,
        context_size=4096,
        state=RuntimeInstanceState.READY,
        request_slots=asyncio.Semaphore(capacity),
        request_capacity=capacity,
        batch_capacity=capacity,
    )


def _pool(monkeypatch: pytest.MonkeyPatch, *, queue_timeout: float = 0.0,
          backend: str = "cuda", max_queued: int = 8) -> RuntimePool:
    pool = RuntimePool(
        None,  # type: ignore[arg-type]
        Path("mfq-runtime"),
        backend=backend,
        max_requests_per_instance=MAX_RUNTIME_CONCURRENCY,
        max_queued_requests_per_instance=max_queued,
        queue_timeout_seconds=queue_timeout,
    )

    async def noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(pool, "_rebalance_prefix_cache_budget", noop)
    monkeypatch.setattr(pool, "_rebalance_resident_budget", noop)
    return pool


def _attach(pool: RuntimePool, instance: _Runtime) -> None:
    pool._instances[instance.id] = instance


async def _select(model: str, *, session_id: Any = None) -> _Runtime | None:
    raise AssertionError("tests replace _select")


def test_requested_capacity_is_bounded_per_backend() -> None:
    assert native_request_capacity(backend="cuda", routed_expert_bytes=0, requested=8) == 8
    assert native_request_capacity(backend="metal", routed_expert_bytes=0, requested=8) == 8
    assert native_request_capacity(backend="hip", routed_expert_bytes=0, requested=8) == 8
    assert native_request_capacity(
        backend="cuda",
        routed_expert_bytes=1 << 40,
        requested=MAX_RUNTIME_CONCURRENCY + 5,
    ) == MAX_RUNTIME_CONCURRENCY
    assert native_request_capacity(backend="cpu", routed_expert_bytes=0, requested=8) == 1
    with pytest.raises(ValueError):
        native_request_capacity(backend="cuda", routed_expert_bytes=0, requested=0)
    assert DEFAULT_RUNTIME_CONCURRENCY <= MAX_RUNTIME_CONCURRENCY


def test_worker_reported_capacity_shrinks_idle_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _pool(monkeypatch)
    instance = _instance(capacity=8)
    _attach(pool, instance)

    pool._apply_batch_capacity(instance, 3)

    assert instance.request_capacity == 3
    assert instance.batch_capacity == 3
    assert instance.request_slots is not None and instance.request_slots._value == 3


def test_reported_capacity_never_grows_or_interrupts_pending_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _pool(monkeypatch)
    instance = _instance(capacity=2)
    _attach(pool, instance)
    pool._apply_batch_capacity(instance, 8)
    assert instance.request_capacity == 2

    instance.active_requests = 1
    pool._apply_batch_capacity(instance, 1)
    assert instance.request_capacity == 2
    assert instance.batch_capacity == 1

    instance.active_requests = 0
    instance.queued_requests = 1
    pool._apply_batch_capacity(instance, 1)
    assert instance.request_capacity == 2


def test_queue_wait_is_measured_and_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        gate = asyncio.Event()
        backend = _FakeBackend(gate)
        pool = _pool(monkeypatch)
        instance = _instance(capacity=1, backend=backend)
        _attach(pool, instance)
        monkeypatch.setattr(pool, "_select", lambda model, *, session_id=None: _resolved(instance))

        async def drain() -> list[BackendDelta]:
            return [
                delta
                async for delta in pool.stream(model="model", messages=[], sampling=SamplingParams())  # type: ignore[arg-type]
            ]

        first = asyncio.create_task(drain())
        await asyncio.sleep(0.02)
        second = asyncio.create_task(drain())
        await asyncio.sleep(0.05)
        gate.set()
        await asyncio.gather(first, second)

        assert instance.queue_wait_samples == 2
        assert instance.queue_wait_ms_max > 0.0
        assert instance.queue_wait_ms_total >= instance.queue_wait_ms_max
        status = pool._concurrency_status(instance)
        assert status["request_capacity"] == 1
        assert status["queue_wait_ms_avg"] > 0.0

    asyncio.run(run())


def test_queue_timeout_releases_the_queue_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        gate = asyncio.Event()
        backend = _FakeBackend(gate)
        pool = _pool(monkeypatch, queue_timeout=0.05)
        instance = _instance(capacity=1, backend=backend)
        _attach(pool, instance)
        monkeypatch.setattr(pool, "_select", lambda model, *, session_id=None: _resolved(instance))

        async def drain() -> None:
            async for _delta in pool.stream(model="model", messages=[], sampling=SamplingParams()):  # type: ignore[arg-type]
                pass

        first = asyncio.create_task(drain())
        await asyncio.sleep(0.02)
        with pytest.raises(BackendError) as failure:
            await drain()
        assert failure.value.code == "runtime_queue_timeout"
        assert failure.value.retryable is True
        assert instance.queued_requests == 0
        gate.set()
        await first

    asyncio.run(run())


def test_full_queue_counts_rejections(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        gate = asyncio.Event()
        backend = _FakeBackend(gate)
        pool = _pool(monkeypatch, max_queued=0)
        instance = _instance(capacity=1, backend=backend)
        _attach(pool, instance)
        monkeypatch.setattr(pool, "_select", lambda model, *, session_id=None: _resolved(instance))

        async def drain() -> None:
            async for _delta in pool.stream(model="model", messages=[], sampling=SamplingParams()):  # type: ignore[arg-type]
                pass

        first = asyncio.create_task(drain())
        await asyncio.sleep(0.02)
        with pytest.raises(BackendError) as failure:
            await drain()
        assert failure.value.code == "runtime_queue_full"
        assert instance.rejections == 1
        gate.set()
        await first

    asyncio.run(run())


async def _resolved(instance: _Runtime) -> _Runtime:
    return instance
