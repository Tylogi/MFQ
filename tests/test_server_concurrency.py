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


@pytest.mark.parametrize("backend", ["cuda", "hip", "metal"])
def test_model_overrides_inherit_defaults_and_reach_native_launch(monkeypatch, backend):
    from mfq.server.protocol.models import ModelLoadRequest

    pool = _pool(monkeypatch, backend=backend, queue_timeout=30)
    defaults = pool.resolve_concurrency(ModelLoadRequest(model="model"))
    assert defaults.model_dump() == {
        "max_requests": 32, "max_queued_requests": 8, "queue_timeout_seconds": 30,
    }
    request = ModelLoadRequest(model="model", max_requests=16, max_queued_requests=0, queue_timeout_seconds=0.5)
    assert pool.resolve_concurrency(request).max_requests == 16
    command, _ = pool._launch_configuration(_artifact(), request, port=1234)
    assert command[command.index("--continuous-batching") + 1] == "16"
    assert pool.resolve_concurrency(request).max_queued_requests == 0


@pytest.mark.parametrize("settings,limit", [
    ({"max_requests": 17}, "16"),
    ({"max_queued_requests": 9}, "8"),
    ({"queue_timeout_seconds": 31}, "30"),
    ({"queue_timeout_seconds": 0}, "30"),
])
def test_limit_validation_precedes_load_job_and_live_mutation(monkeypatch, tmp_path, settings, limit):
    import httpx
    from mfq.server.api import create_app
    from mfq.server.services.service import ServerService
    from mfq.server.state.storage import SessionStore

    async def run():
        pool = _pool(monkeypatch, queue_timeout=30)
        pool.max_requests_per_instance = 16
        instance = _instance(capacity=8)
        _attach(pool, instance)
        service = ServerService(SessionStore(tmp_path / "server.sqlite3"), pool, runtime_manager=pool)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)),
                                     base_url="http://test") as client:
            for route, body in [
                ("/api/v1/models/load", {"model": "model", **settings}),
                (f"/api/v1/runtime/instances/{instance.id}/concurrency", settings),
                ("/api/v1/runtime/concurrency/reload", {"instance_id": str(instance.id), **settings}),
            ]:
                response = await client.request("PUT" if "/instances/" in route else "POST", route, json=body)
                assert response.status_code == 422
                assert limit in response.json()["error"]["message"]
            policy = (await client.get("/api/v1/runtime/concurrency")).json()
            assert policy["limits"]["max_requests"] == 16
        assert instance.concurrency_settings is None
        assert service.store.list_jobs() == []
        pool._instances.clear()
        await service.aclose()

    asyncio.run(run())


def test_live_queue_edits_report_reload_and_preserve_worker_capacity(monkeypatch):
    from mfq.server.protocol.models import ModelLoadRequest, UpdateRuntimeConcurrencyRequest

    async def run():
        pool = _pool(monkeypatch)
        instance = _instance(capacity=8)
        instance.concurrency_settings = pool.resolve_concurrency(ModelLoadRequest(model="model", max_requests=16))
        pool._load_requests["model"] = ModelLoadRequest(model="model", max_requests=16)
        _attach(pool, instance)
        slots = instance.request_slots
        result = await pool.update_concurrency(instance.id, UpdateRuntimeConcurrencyRequest(
            max_requests=24, max_queued_requests=2, queue_timeout_seconds=0.5,
        ))
        assert result.reload_required
        assert result.concurrency.requested.max_requests == 16
        assert result.concurrency.effective.max_requests == 8
        assert result.concurrency.limits.max_requests == 32
        assert instance.request_slots is slots
        assert instance.request_capacity == 8
        assert instance.concurrency_settings.max_queued_requests == 2
        assert pool._load_requests["model"].queue_timeout_seconds == 0.5
        assert pool._load_requests["model"].max_requests == 16
        result = await pool.update_concurrency(instance.id, UpdateRuntimeConcurrencyRequest(max_queued_requests=0))
        assert not result.reload_required

    asyncio.run(run())


def test_existing_waiters_adopt_timeout_edits_and_cleanup(monkeypatch):
    from mfq.server.protocol.models import UpdateRuntimeConcurrencyRequest

    async def run():
        pool = _pool(monkeypatch)
        gate = asyncio.Event()
        instance = _instance(capacity=1, backend=_FakeBackend(gate))
        _attach(pool, instance)
        monkeypatch.setattr(pool, "_select", lambda model, *, session_id=None: _resolved(instance))

        async def drain():
            return [delta async for delta in pool.stream(model="model", messages=[], sampling=SamplingParams())]

        first = asyncio.create_task(drain())
        while not instance.active_requests:
            await asyncio.sleep(0)
        waiter = asyncio.create_task(drain())
        while not instance.queued_requests:
            await asyncio.sleep(0)
        await pool.update_concurrency(instance.id, UpdateRuntimeConcurrencyRequest(
            max_queued_requests=0, queue_timeout_seconds=0.02,
        ))
        with pytest.raises(BackendError, match="runtime slot") as failure:
            await asyncio.wait_for(waiter, timeout=1)
        assert failure.value.code == "runtime_queue_timeout"
        assert instance.rejections == 1 and instance.queued_requests == 0
        gate.set()
        await first
        assert instance.request_slots._value == 1

    asyncio.run(run())


def test_nonbatch_model_exposes_requested_and_effective_capacity(monkeypatch):
    from mfq.server.protocol.models import ModelLoadRequest

    async def run():
        pool = _pool(monkeypatch)
        instance = _instance(capacity=16)
        instance.concurrency_settings = pool.resolve_concurrency(ModelLoadRequest(model="model", max_requests=16))
        _attach(pool, instance)
        pool._apply_batch_capacity(instance, 1)
        resource = (await pool.instances()).data[0]
        assert resource.concurrency.requested.max_requests == 16
        assert resource.concurrency.effective.max_requests == 1
        assert resource.batch_capacity == resource.request_capacity == 1
        assert resource.concurrency.effective.max_queued_requests == 8

    asyncio.run(run())


def test_explicit_reload_preserves_full_load_policy_and_recovers_on_failure(monkeypatch):
    from unittest.mock import AsyncMock
    from mfq.server.protocol.models import ModelLoadRequest
    from tests.test_server_models import _TestJobContext

    async def run():
        pool = _pool(monkeypatch)
        instance = _instance(capacity=8)
        _attach(pool, instance)
        previous = ModelLoadRequest(model="model", context_size=65536, device_ids=["1"],
            prefill_chunk_size=512, prefix_cache_enabled=False, max_requests=8)
        pool._load_requests["model"] = previous
        pool._retire_instance = AsyncMock(side_effect=lambda item: pool._instances.pop(item.id, None))
        pool.load = AsyncMock(side_effect=[RuntimeError("load failed"), {"restored": True}])
        with pytest.raises(RuntimeError, match="load failed"):
            await pool.configure_concurrency(_TestJobContext(), {
                "instance_id": str(instance.id), "max_requests": 16, "max_queued_requests": 2,
            })
        payload = pool.load.await_args_list[0].args[1]
        assert payload["max_requests"] == 16 and payload["max_queued_requests"] == 2
        for name in ["context_size", "device_ids", "prefill_chunk_size", "prefix_cache_enabled"]:
            assert payload[name] == getattr(previous, name)
        assert pool.load.await_args_list[1].args[1] == previous.model_dump(mode="json")
        assert pool._load_requests["model"] == previous

    asyncio.run(run())


def test_busy_reload_does_not_retire_a_worker(monkeypatch):
    from unittest.mock import AsyncMock
    from mfq.server.services.jobs import JobExecutionError
    from tests.test_server_models import _TestJobContext

    async def run():
        pool = _pool(monkeypatch)
        instance = _instance(capacity=8)
        instance.active_requests = 1
        _attach(pool, instance)
        pool._retire_instance = AsyncMock()
        with pytest.raises(JobExecutionError) as failure:
            await pool.configure_concurrency(_TestJobContext(), {
                "instance_id": str(instance.id), "max_requests": 16,
            })
        assert failure.value.detail.code == "runtime_busy"
        assert failure.value.detail.retryable
        pool._retire_instance.assert_not_awaited()

    asyncio.run(run())


def test_cancelled_waiter_releases_queue_without_leaking_a_slot(monkeypatch):
    import time

    async def run():
        pool = _pool(monkeypatch)
        instance = _instance(capacity=1)
        await instance.request_slots.acquire()
        waiter = asyncio.create_task(pool._acquire_request_slot(instance, time.monotonic(), "model"))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        instance.request_slots.release()
        assert instance.request_slots._value == 1

    asyncio.run(run())


def test_runtime_routes_apply_queue_settings_and_create_explicit_reload_job(monkeypatch, tmp_path):
    import httpx
    from mfq.server.api import create_app
    from mfq.server.protocol.models import ModelLoadRequest
    from mfq.server.services.service import ServerService
    from mfq.server.state.storage import SessionStore

    async def run():
        pool = _pool(monkeypatch)
        instance = _instance(capacity=8)
        instance.concurrency_settings = pool.resolve_concurrency(ModelLoadRequest(model="model", max_requests=16))
        _attach(pool, instance)
        pool._load_requests["model"] = ModelLoadRequest(model="model", max_requests=16)
        service = ServerService(SessionStore(tmp_path / "server.sqlite3"), pool, runtime_manager=pool)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)),
                                     base_url="http://test") as client:
            response = await client.put(f"/api/v1/runtime/instances/{instance.id}/concurrency", json={
                "max_requests": 24, "max_queued_requests": 0, "queue_timeout_seconds": 1.5,
            })
            assert response.status_code == 200
            assert response.json()["reload_required"] is True
            assert response.json()["concurrency"]["requested"]["max_requests"] == 16
            assert response.json()["concurrency"]["effective"]["max_queued_requests"] == 0
            assert pool._instances[instance.id] is instance
            assert not service.store.list_jobs()
            response = await client.post("/api/v1/runtime/concurrency/reload", json={
                "instance_id": str(instance.id), "max_requests": 24,
            })
            assert response.status_code == 202
            assert response.json()["kind"] == "runtime.concurrency.configure"
            assert response.json()["payload"]["max_requests"] == 24
        pool._instances.clear()
        await service.aclose()

    asyncio.run(run())
