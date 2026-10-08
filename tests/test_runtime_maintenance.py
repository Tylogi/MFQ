import asyncio
import threading
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.protocol.models import RuntimeInstanceState, RuntimeMemoryResources, SamplingParams
from mfq.server.runtime.backend import BackendDelta, BackendError
from mfq.server.runtime.runtime_pool import RuntimePool, _Runtime
from mfq.server.state.catalog import ModelCatalog
from tests.test_resident_memory_budget import BudgetBackend
from tests.test_runtime_memory_policy import GIB, artifact
from tests.test_server_openai_compat import _Service


class MaintenanceBackend(BudgetBackend):
    def __init__(self):
        super().__init__(GIB, 3 * GIB, GIB, 2 * GIB, 8 * GIB, [])
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.inference_calls = 0

    async def trim_runtime_cache(self, target):
        self.entered.set()
        await self.release.wait()
        return await super().trim_runtime_cache(target)

    async def stream(self, **kwargs):
        self.inference_calls += 1
        yield BackendDelta(content_delta="ok", finish_reason="stop")


def runtime(tmp_path):
    backend = MaintenanceBackend()
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda",
        automatic_memory_budget=False, max_runtime_memory_bytes=8 * GIB)
    instance = _Runtime(id=uuid4(), artifact=artifact(tmp_path, "probe", 8, 6), backend=backend,
        process=SimpleNamespace(returncode=None), port=0, context_size=4096,
        state=RuntimeInstanceState.READY, request_slots=asyncio.Semaphore(1),
        resident_dynamic_budget=True, resident_budget_limit=8 * GIB,
        resident_memory_used=backend.used, reclaimable_weight_bytes=backend.experts, kv_bytes=backend.prefix)
    pool._instances[instance.id] = instance
    return pool, instance, backend


async def request(pool):
    return [delta async for delta in pool.stream(model="probe", messages=[], sampling=SamplingParams())]


async def wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)
    await asyncio.wait_for(wait(), 1)


def test_automatic_cache_maintenance_queues_inference_and_yields_to_waiters(tmp_path):
    async def run():
        pool, instance, backend = runtime(tmp_path)
        maintenance = asyncio.create_task(pool._trim_idle_prefix_caches_for_budget(memory_target_bytes=6 * GIB))
        await asyncio.wait_for(backend.entered.wait(), 1)
        pending = asyncio.create_task(request(pool))
        await wait_until(lambda: instance.queued_requests == 1)
        assert not pending.done() and backend.inference_calls == 0
        assert instance.active_requests == 0
        await pool._rebalance_resident_budget()
        await pool._trim_idle_prefix_caches_for_budget(memory_target_bytes=6 * GIB)
        assert instance.control_leases == 1 and not backend.events
        backend.release.set()
        await maintenance
        assert [delta.content_delta for delta in await asyncio.wait_for(pending, 1)] == ["ok"]
        assert backend.inference_calls == 1
        assert instance.control_leases == instance.active_requests == instance.queued_requests == 0
        assert instance.request_slots._value == 1
    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_maintenance_timeout_and_cancellation_release_queue_and_slot(tmp_path, monkeypatch, cancel):
    monkeypatch.setattr("mfq.server.runtime.runtime_pool._RUNTIME_MAINTENANCE_WAIT_SECONDS", .03)
    async def run():
        pool, instance, backend = runtime(tmp_path)
        async with pool._runtime_control_lease(instance.id):
            pending = asyncio.create_task(request(pool))
            await wait_until(lambda: instance.queued_requests == 1)
            if cancel:
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                with pytest.raises(BackendError) as error:
                    await pending
                assert error.value.code == "runtime_maintenance_timeout"
                assert error.value.retryable and error.value.status_code == 503
            assert instance.queued_requests == instance.active_requests == backend.inference_calls == 0
            assert instance.request_slots._value == 1
        assert [delta.content_delta for delta in await request(pool)] == ["ok"]
    asyncio.run(run())


def test_maintenance_waiters_are_bounded_by_existing_queue_capacity(tmp_path):
    async def run():
        pool, instance, backend = runtime(tmp_path)
        pool.max_queued_requests_per_instance = 1
        async with pool._runtime_control_lease(instance.id):
            pending = [asyncio.create_task(request(pool)) for _ in range(2)]
            await wait_until(lambda: instance.queued_requests == 2)
            with pytest.raises(BackendError) as error:
                await request(pool)
            assert error.value.code == "runtime_queue_full" and error.value.status_code == 429
            assert backend.inference_calls == 0
        results = await asyncio.wait_for(asyncio.gather(*pending), 1)
        assert len(results) == backend.inference_calls == 2
        assert instance.active_requests == instance.queued_requests == 0
        assert instance.request_slots._value == 1
    asyncio.run(run())


def test_unload_wakes_maintenance_waiter_without_running_model(tmp_path):
    async def run():
        pool, instance, backend = runtime(tmp_path)
        async with pool._runtime_control_lease(instance.id):
            pending = asyncio.create_task(request(pool))
            await wait_until(lambda: instance.queued_requests == 1)
            async with pool._lock:
                pool._mark_instance_unloading_locked(instance)
            with pytest.raises(BackendError) as error:
                await asyncio.wait_for(pending, 1)
            assert error.value.code == "model_not_ready" and error.value.status_code == 503
            assert backend.inference_calls == instance.active_requests == instance.queued_requests == 0
            assert instance.request_slots._value == 1
    asyncio.run(run())


def test_noop_resident_budget_does_not_acquire_exclusive_control(tmp_path, monkeypatch):
    async def run():
        pool, instance, backend = runtime(tmp_path)
        async def unexpected_release(*args, **kwargs):
            pytest.fail("unchanged budgets must not take an exclusive control lease")
        monkeypatch.setattr(pool, "_release_control_lease", unexpected_release)
        await pool._rebalance_resident_budget()
        assert not backend.events and instance.control_leases == 0
    asyncio.run(run())


def test_requests_do_not_wait_for_background_budget_scan(tmp_path):
    async def run():
        pool, instance, backend = runtime(tmp_path)
        async with pool._prefix_budget_lock, pool._resident_budget_lock:
            assert [delta.content_delta for delta in await asyncio.wait_for(request(pool), 1)] == ["ok"]
        assert backend.inference_calls == 1
    asyncio.run(run())


def test_budget_statistics_refresh_does_not_hold_exclusive_lease(tmp_path, monkeypatch):
    async def run():
        pool, instance, backend = runtime(tmp_path)
        instance.resident_budget_limit = backend.limit = 7 * GIB
        entered, release = asyncio.Event(), asyncio.Event()
        async def slow_stats(item):
            entered.set()
            await release.wait()
        monkeypatch.setattr(pool, "_refresh_instance_usage", slow_stats)
        maintenance = asyncio.create_task(pool._rebalance_resident_budget())
        await asyncio.wait_for(entered.wait(), 1)
        assert instance.control_leases == 0
        assert [delta.content_delta for delta in await asyncio.wait_for(request(pool), 1)] == ["ok"]
        release.set()
        await maintenance
    asyncio.run(run())


@pytest.mark.parametrize("automatic", [False, True])
def test_concurrent_prefix_trim_does_not_block_generation(tmp_path, automatic):
    async def run():
        pool, instance, backend = runtime(tmp_path)
        instance.prefix_concurrent_maintenance = True
        maintenance = asyncio.create_task(pool._trim_idle_prefix_caches_for_budget(memory_target_bytes=6 * GIB)
            if automatic else pool.trim_runtime_cache(instance_id=instance.id))
        await asyncio.wait_for(backend.entered.wait(), 1)
        assert instance.control_leases == instance.read_control_leases == 1
        assert [delta.content_delta for delta in await asyncio.wait_for(request(pool), 1)] == ["ok"]
        assert not maintenance.done() and backend.inference_calls == 1
        backend.release.set()
        await maintenance
        assert instance.control_leases == instance.read_control_leases == 0
    asyncio.run(run())


def test_prefix_budget_does_not_resend_unchanged_disk_limit(tmp_path):
    async def run():
        pool, instance, backend = runtime(tmp_path)
        instance.prefix_dynamic_budget = instance.prefix_concurrent_maintenance = True
        instance.prefix_disk_capacity = 100 * GIB
        instance.memory = RuntimeMemoryResources(prefix_cache_bytes=0, prefix_cache_limit_bytes=GIB)
        async def set_budget(target, **kwargs):
            assert target == 2 * GIB and not kwargs
            backend.entered.set()
            await backend.release.wait()
            return {"status": "ok"}
        backend.set_prefix_cache_budget = set_budget
        maintenance = asyncio.create_task(pool._rebalance_prefix_cache_budget())
        await asyncio.wait_for(backend.entered.wait(), 1)
        assert instance.control_leases == instance.read_control_leases == 1
        assert [delta.content_delta for delta in await asyncio.wait_for(request(pool), 1)] == ["ok"]
        backend.release.set()
        await maintenance
        assert instance.control_leases == instance.read_control_leases == 0
    asyncio.run(run())


@pytest.mark.parametrize("over_budget", [False, True])
def test_disk_scan_is_nonexclusive_but_actual_eviction_protects_requests(tmp_path, monkeypatch, over_budget):
    async def run():
        pool, instance, backend = runtime(tmp_path)
        entered, release = threading.Event(), threading.Event()
        def maintain(directory, limit, *, evict):
            if evict or not over_budget:
                entered.set()
                assert release.wait(2)
            return {"prefix_cache_total_disk_bytes": 2 if over_budget and not evict else 0,
                "prefix_cache_total_disk_max_bytes": 1}
        monkeypatch.setattr("mfq.server.runtime.runtime_pool.maintain_prefix_disk_budget", maintain)
        maintenance = asyncio.create_task(pool._maintain_prefix_disk_budget(force=True))
        await wait_until(entered.is_set)
        pending = asyncio.create_task(request(pool))
        if over_budget:
            await wait_until(lambda: instance.queued_requests == 1)
            assert instance.control_leases == 1 and backend.inference_calls == 0
            maintenance.cancel()
            await asyncio.sleep(0)
            assert instance.control_leases == 1
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await maintenance
        else:
            assert instance.control_leases == 0
            assert [delta.content_delta for delta in await asyncio.wait_for(pending, 1)] == ["ok"]
            release.set()
            await maintenance
        assert [delta.content_delta for delta in await asyncio.wait_for(pending, 1)] == ["ok"]
        assert instance.control_leases == 0
    asyncio.run(run())


def test_memory_plan_finishes_before_waiter_reserves_queue_capacity(tmp_path):
    async def run():
        pool, instance, backend = runtime(tmp_path)
        async with pool._lock:
            pool._memory_configuration_job = uuid4()
        pending = asyncio.create_task(request(pool))
        await wait_until(lambda: bool(pool._request_state_changed._waiters))
        assert instance.queued_requests == instance.active_requests == backend.inference_calls == 0
        async with pool._lock:
            pool._memory_configuration_job = None
            pool._request_state_changed.notify_all()
        assert [delta.content_delta for delta in await asyncio.wait_for(pending, 1)] == ["ok"]
    asyncio.run(run())


def test_maintenance_starting_after_preflight_also_waits(tmp_path):
    async def run():
        pool, instance, backend = runtime(tmp_path)
        await pool.preflight(model="probe", messages=[], sampling=SamplingParams())
        async with pool._runtime_control_lease(instance.id):
            pending = asyncio.create_task(request(pool))
            await wait_until(lambda: instance.queued_requests == 1)
            assert backend.inference_calls == 0 and not pending.done()
        assert [delta.content_delta for delta in await asyncio.wait_for(pending, 1)] == ["ok"]
    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("count", [1, 16])
def test_openai_request_waits_for_maintenance_without_returning_error(tmp_path, stream, count):
    async def run():
        pool, instance, backend = runtime(tmp_path)
        app = create_app(_Service(pool))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            async with pool._runtime_control_lease(instance.id):
                pending = [asyncio.create_task(client.post("/v1/chat/completions", json={
                    "model": "probe", "messages": [{"role": "user", "content": "hello"}], "stream": stream})) for _ in range(count)]
                await wait_until(lambda: (len(pool._request_state_changed._waiters) if stream else instance.queued_requests) == count)
                assert not any(task.done() for task in pending) and backend.inference_calls == 0
            responses = await asyncio.wait_for(asyncio.gather(*pending), 1)
        assert all(response.status_code == 200 and '"error"' not in response.text for response in responses)
        assert all('"ok"' in response.text for response in responses) and backend.inference_calls == count
        assert instance.active_requests == instance.queued_requests == 0
    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_openai_maintenance_timeout_is_http_503_with_retry_after(tmp_path, monkeypatch, stream):
    monkeypatch.setattr("mfq.server.runtime.runtime_pool._RUNTIME_MAINTENANCE_WAIT_SECONDS", .01)
    async def run():
        pool, instance, backend = runtime(tmp_path)
        app = create_app(_Service(pool))
        async with (
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
            pool._runtime_control_lease(instance.id),
        ):
            response = await client.post("/v1/chat/completions", json={
                "model": "probe", "messages": [{"role": "user", "content": "hello"}], "stream": stream})
        assert response.status_code == 503
        assert response.headers["retry-after"] == "1"
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["error"]["type"] == "runtime_maintenance_timeout"
        assert backend.inference_calls == instance.queued_requests == instance.active_requests == 0
    asyncio.run(run())
