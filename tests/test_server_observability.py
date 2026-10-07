from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest

import mfq.server.state.storage as storage_module
from mfq.server.api import create_app
from mfq.server.protocol.models import ResponsePerformance, RuntimeInstanceState, RuntimeLogLevel, SamplingParams, TokenUsage
from mfq.server.runtime.backend import BackendDelta, BackendError
from mfq.server.runtime.runtime_pool import RuntimePool, _Runtime
from mfq.server.state.catalog import ModelCatalog
from mfq.server.services.service import ServerService
from mfq.server.state.storage import SessionStore
from tests.test_server_models import IdleBackend, _model

INSTANCE_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)


class StatusBackend:
    async def runtime_status(self):
        return {
            "instance_id": str(INSTANCE_ID),
            "model": "model-a",
            "total_requests": 3,
            "last_request": {"id": "request-a", "decode_tps": 24.5},
        }

    async def aclose(self):
        return None


def test_runtime_metrics_and_logs_persist_and_filter(tmp_path: Path) -> None:
    async def run() -> None:
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        service = ServerService(store, StatusBackend())  # type: ignore[arg-type]
        store.append_runtime_log(
            RuntimeLogLevel.INFO,
            "loaded",
            instance_id=INSTANCE_ID,
            fields={"source": "test"},
            now=NOW,
        )
        store.append_runtime_log(RuntimeLogLevel.ERROR, "failed", now=NOW)
        transport = httpx.ASGITransport(app=create_app(service))
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                status = await client.get("/api/v1/runtime/status")
                assert status.status_code == 200
                repeated_status = await client.get("/api/v1/runtime/status")
                assert repeated_status.status_code == 200
                metrics = await client.get(
                    "/api/v1/runtime/metrics", params={"instance_id": str(INSTANCE_ID)}
                )
                assert metrics.status_code == 200
                assert len(metrics.json()["data"]) == 1
                assert metrics.json()["data"][0]["values"]["total_requests"] == 3
                assert metrics.json()["data"][0]["model"] == "model-a"

                logs = await client.get(
                    "/api/v1/runtime/logs",
                    params={"instance_id": str(INSTANCE_ID), "level": "info"},
                )
                assert logs.status_code == 200
                assert [entry["message"] for entry in logs.json()["data"]] == ["loaded"]
        finally:
            await service.aclose()

        reopened = SessionStore(tmp_path / "mfq.server.sqlite3")
        assert reopened.list_runtime_metrics()[0].values["last_request"]["id"] == "request-a"
        assert len(reopened.list_runtime_logs()) == 2

    asyncio.run(run())


def test_runtime_metric_history_has_a_bounded_retention_window(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(storage_module, "_MAX_RUNTIME_METRICS", 3)
    store = SessionStore(tmp_path / "mfq.server.sqlite3")
    for value in range(5):
        store.append_runtime_metric({"value": value})

    assert [entry.values["value"] for entry in store.list_runtime_metrics()] == [2, 3, 4]


def test_completed_requests_are_persisted_without_status_polling(tmp_path: Path) -> None:
    class Backend(IdleBackend):
        count = 0

        async def stream(self, **kwargs):
            self.count += 1
            request_id = f"request-{self.count}"
            yield BackendDelta(content_delta="done", finish_reason="stop", backend_request_id=request_id)
            performance = ResponsePerformance(prefill_tokens=16, ttft_ms=5, prefill_ms=4,
                prefill_tps=4000, decode_ms=50, decode_tps=60, generation_ms=55,
                generation_tps=72.7, sampling=SamplingParams())
            for _ in range(2):
                yield BackendDelta(backend_request_id=request_id, performance=performance,
                    usage=TokenUsage(prompt_tokens=16, completion_tokens=4, total_tokens=20))

    async def run():
        path = tmp_path / "model-a.mfq"
        _model(path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        pool = RuntimePool(catalog, tmp_path / "runtime", automatic_memory_budget=False)
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        pool.store = store
        instance = _Runtime(id=uuid4(), artifact=await catalog.resolve_path(path),
            backend=Backend(), process=SimpleNamespace(returncode=None), port=0,
            context_size=4096, state=RuntimeInstanceState.READY,
            request_slots=asyncio.Semaphore(1))
        pool._instances[instance.id] = instance
        for _ in range(3):
            assert len([delta async for delta in pool.stream(model="model-a", messages=[],
                sampling=SamplingParams())]) == 3
        rows = store.list_runtime_metrics()
        assert [row.values["last_request"]["id"] for row in rows] == ["request-1", "request-2", "request-3"]
        assert all(row.instance_id == instance.id and row.model == "model-a" for row in rows)
        assert all(row.values["last_request"]["decode_tps"] == 60 for row in rows)
        assert all(row.values["last_request"]["completion_tokens"] == 4 for row in rows)
        assert all(row.values["last_request"]["finish_reason"] == "stop" for row in rows)
        assert all(row.values["last_request"]["generation_ms"] == 55 for row in rows)
        assert all("complete_generation_ms" not in row.values["last_request"] for row in rows)
        assert instance.active_requests == 0
        assert instance.request_slots._value == 1

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["queued", "admitting", "active"])
def test_cancelled_requests_release_slots_even_if_cancelled_again_during_cleanup(tmp_path, phase):
    async def run():
        entered = asyncio.Event()
        acquired = asyncio.Event()
        class Backend(IdleBackend):
            async def stream(self, **kwargs):
                entered.set()
                await asyncio.Event().wait()
                yield BackendDelta(content_delta="done")
        path = tmp_path / "model-a.mfq"
        _model(path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        pool = RuntimePool(catalog, tmp_path / "runtime", automatic_memory_budget=False)
        pool.store = SessionStore(tmp_path / "state.sqlite3")
        class AdmissionSemaphore(asyncio.Semaphore):
            async def acquire(self):
                result = await super().acquire()
                await pool._lock.acquire()
                acquired.set()
                return result
        slots = AdmissionSemaphore(1) if phase == "admitting" else asyncio.Semaphore(0 if phase == "queued" else 1)
        instance = _Runtime(id=uuid4(), artifact=await catalog.resolve_path(path), backend=Backend(),
            process=SimpleNamespace(returncode=None), port=0, context_size=4096,
            state=RuntimeInstanceState.READY, request_slots=slots)
        pool._instances[instance.id] = instance
        async def consume():
            async for _ in pool.stream(model="model-a", messages=[], sampling=SamplingParams()):
                pass
        request = asyncio.create_task(consume())
        if phase == "admitting":
            await asyncio.wait_for(acquired.wait(), 1)
        elif phase == "active":
            await asyncio.wait_for(entered.wait(), 1)
            await pool._lock.acquire()
        else:
            while not instance.queued_requests:
                await asyncio.sleep(0)
            await pool._lock.acquire()
        request.cancel()
        await asyncio.sleep(0)
        request.cancel()
        await asyncio.sleep(0)
        pool._lock.release()
        with pytest.raises(asyncio.CancelledError):
            await request
        await pool._drain_control_lease_releases()
        assert instance.active_requests == instance.queued_requests == 0
        assert slots._value == (0 if phase == "queued" else 1)
        assert instance.state == RuntimeInstanceState.READY
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("operation", ["runtime_status", "capabilities"])
def test_read_only_runtime_monitoring_does_not_reject_inference_or_allow_eviction(tmp_path, operation):
    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()
        class Backend(IdleBackend):
            async def runtime_status(self):
                entered.set()
                await release.wait()
                return {"model": "model-a"}

            async def capabilities(self):
                entered.set()
                await release.wait()
                return {"backend": "metal"}

            async def stream(self, **kwargs):
                yield BackendDelta(content_delta="done", finish_reason="stop")

        path = tmp_path / "model-a.mfq"
        _model(path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        pool = RuntimePool(catalog, tmp_path / "runtime", automatic_memory_budget=False)
        instance = _Runtime(id=uuid4(), artifact=await catalog.resolve_path(path), backend=Backend(),
            process=SimpleNamespace(returncode=None), port=0, context_size=4096,
            state=RuntimeInstanceState.READY, request_slots=asyncio.Semaphore(1))
        pool._instances[instance.id] = instance
        monitoring = asyncio.create_task(getattr(pool, operation)(instance.id))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert instance.control_leases == 1
            async with pool._lock:
                assert pool._claim_lru_instance_for_unload_locked() is None
            deltas = [delta async for delta in pool.stream(model="model-a", messages=[], sampling=SamplingParams())]
            assert len(deltas) == 1 and deltas[0].content_delta == "done"
            assert instance.active_requests == instance.queued_requests == 0
            assert instance.control_leases == instance.read_control_leases == 1
        finally:
            release.set()
            await monitoring
        assert instance.control_leases == instance.read_control_leases == 0
        assert instance.request_slots._value == 1
    asyncio.run(run())


def test_exclusive_maintenance_still_blocks_inference_when_read_only_monitoring_is_active(tmp_path):
    async def run():
        class Backend(IdleBackend):
            async def stream(self, **kwargs):
                yield BackendDelta(content_delta="done", finish_reason="stop")
        path = tmp_path / "model-a.mfq"
        _model(path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        pool = RuntimePool(catalog, tmp_path / "runtime", automatic_memory_budget=False)
        instance = _Runtime(id=uuid4(), artifact=await catalog.resolve_path(path), backend=Backend(),
            process=SimpleNamespace(returncode=None), port=0, context_size=4096,
            state=RuntimeInstanceState.READY, request_slots=asyncio.Semaphore(1))
        pool._instances[instance.id] = instance
        async with pool._runtime_control_lease(instance.id, read_only=True):
            async with pool._runtime_control_lease(instance.id):
                assert instance.control_leases == 2 and instance.read_control_leases == 1
                with pytest.raises(BackendError) as rejected:
                    async for _ in pool.stream(model="model-a", messages=[], sampling=SamplingParams()):
                        pass
                assert rejected.value.code == "runtime_reconfiguring" and rejected.value.status_code == 409
                assert instance.active_requests == instance.queued_requests == 0
            assert len([delta async for delta in pool.stream(model="model-a", messages=[], sampling=SamplingParams())]) == 1
        assert instance.control_leases == instance.read_control_leases == 0
    asyncio.run(run())


def test_runtime_logs_default_to_latest_window_and_preserve_incremental_order(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "mfq.server.sqlite3")
    for value in range(8):
        store.append_runtime_log(RuntimeLogLevel.INFO, str(value), instance_id=INSTANCE_ID)
    store.append_runtime_log(RuntimeLogLevel.ERROR, "service error")
    assert [item.message for item in store.list_runtime_logs(limit=3)] == ["6", "7", "service error"]
    assert [item.message for item in store.list_runtime_logs(after=0, limit=3)] == ["0", "1", "2"]
    assert [item.message for item in store.list_runtime_logs(after=3, limit=3)] == ["3", "4", "5"]
    assert [item.message for item in store.list_runtime_logs(instance_id=INSTANCE_ID, limit=2)] == ["6", "7"]
    assert [item.message for item in store.list_runtime_logs(level=RuntimeLogLevel.INFO, limit=2)] == ["6", "7"]


def test_runtime_log_api_distinguishes_latest_and_explicit_cursor(tmp_path: Path) -> None:
    async def run() -> None:
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        for value in range(5):
            store.append_runtime_log(RuntimeLogLevel.INFO, str(value))
        service = ServerService(store, StatusBackend())  # type: ignore[arg-type]
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
                latest = await client.get("/api/v1/runtime/logs", params={"limit": 2})
                assert latest.status_code == 200
                assert [item["message"] for item in latest.json()["data"]] == ["3", "4"]
                first = await client.get("/api/v1/runtime/logs", params={"after": 0, "limit": 2})
                assert [item["message"] for item in first.json()["data"]] == ["0", "1"]
                following = await client.get("/api/v1/runtime/logs", params={"after": 2, "limit": 2})
                assert [item["message"] for item in following.json()["data"]] == ["2", "3"]
                assert (await client.get("/api/v1/runtime/logs?after=-1")).status_code == 422
        finally:
            await service.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("preflight", [False, True])
@pytest.mark.parametrize("logging_fails", [False, True])
def test_queue_rejections_are_observable_without_logging_prompts_or_changing_errors(tmp_path, monkeypatch, preflight, logging_fails):
    async def run():
        path = tmp_path / "model-a.mfq"
        _model(path)
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        pool = RuntimePool(catalog, tmp_path / "runtime", automatic_memory_budget=False,
            max_queued_requests_per_instance=1)
        store = SessionStore(tmp_path / "server.sqlite3")
        pool.store = store
        instance = _Runtime(id=uuid4(), artifact=await catalog.resolve_path(path), backend=IdleBackend(),
            process=SimpleNamespace(returncode=None), port=0, context_size=4096,
            state=RuntimeInstanceState.BUSY, request_slots=asyncio.Semaphore(0), active_requests=1, queued_requests=1)
        pool._instances[instance.id] = instance
        if logging_fails:
            def fail(*args, **kwargs):
                raise OSError("log storage unavailable")
            monkeypatch.setattr(store, "append_runtime_log", fail)
        options = {"model": instance.artifact.resource.name,
            "messages": [{"role": "user", "content": "private prompt must not be logged"}], "sampling": SamplingParams()}
        with pytest.raises(BackendError) as rejected:
            if preflight:
                await pool.preflight(**options)
            else:
                async for _ in pool.stream(**options):
                    pass
        assert rejected.value.code == "runtime_queue_full" and rejected.value.status_code == 429
        assert rejected.value.retryable
        assert instance.active_requests == instance.queued_requests == 1
        records = store.list_runtime_logs()
        if logging_fails:
            assert not records
        else:
            assert len(records) == 1 and records[0].instance_id == instance.id
            assert records[0].level == RuntimeLogLevel.WARNING
            assert records[0].fields == {"source": "runtime.admission", "model": instance.artifact.resource.name,
                "code": "runtime_queue_full", "status_code": 429}
            assert "private prompt" not in records[0].model_dump_json()
    asyncio.run(run())
