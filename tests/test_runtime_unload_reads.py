"""Verify manual unload drains status readers without bypassing mutation or inference guards."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import mfq.server.runtime.runtime_pool as runtime_module
from mfq.server.protocol.models import RuntimeInstanceState
from mfq.server.runtime.backend import BackendError
from mfq.server.runtime.runtime_pool import RuntimePool, _Runtime
from mfq.server.services.jobs import JobExecutionError
from mfq.server.state.catalog import ModelCatalog


class Context:
    """Expose job progress and cleanup without running a worker process."""

    def __init__(self):
        self.started = asyncio.Event()
        self.cleanup = []

    def add_cleanup(self, callback):
        self.cleanup.append(callback)

    async def progress(self, *args, **kwargs):
        self.started.set()


def make_pool():
    """Build a ready instance with a controllable asynchronous status reader."""
    entered, release = asyncio.Event(), asyncio.Event()

    class Backend:
        async def runtime_status(self):
            entered.set()
            await release.wait()
            return {'model': 'probe'}

    pool = RuntimePool(ModelCatalog([]), Path('unused-runtime'))
    pool._runtime_memory_pressure_locked = lambda: ('normal', 0, None, 0)
    instance = _Runtime.model_construct(
        id=uuid4(), artifact=SimpleNamespace(resource=SimpleNamespace(name='probe')),
        process=SimpleNamespace(returncode=None), backend=Backend(), port=0,
        context_size=4096, state=RuntimeInstanceState.READY,
    )
    pool._instances[instance.id] = instance
    pool._stop_process = AsyncMock()
    return pool, instance, entered, release


@pytest.mark.parametrize('cancel_read', [False, True])
def test_unload_waits_for_status_then_stops_once(cancel_read):
    async def run():
        pool, instance, entered, release = make_pool()
        read = asyncio.create_task(pool.runtime_status(instance.id))
        await asyncio.wait_for(entered.wait(), 1)
        assert instance.control_leases == instance.read_leases == 1
        context = Context()
        unloading = asyncio.create_task(pool.unload(context, {'instance_id': str(instance.id)}))
        await asyncio.wait_for(context.started.wait(), 1)
        assert instance.state == RuntimeInstanceState.UNLOADING
        assert not unloading.done()
        pool._stop_process.assert_not_called()
        # Reads during unloading return a snapshot and cannot prolong the drain.
        assert (await pool.runtime_status(instance.id))['runtime_state'] == 'unloading'
        assert instance.read_leases == 1
        with pytest.raises(BackendError):
            async with pool._runtime_control_lease(instance.id):
                pass
        if cancel_read:
            read.cancel()
            with pytest.raises(asyncio.CancelledError):
                await read
        else:
            release.set()
            await read
        assert (await asyncio.wait_for(unloading, 1))['unloaded']
        assert instance.control_leases == instance.read_leases == 0
        assert instance.id not in pool._instances
        pool._stop_process.assert_awaited_once()
        for cleanup in context.cleanup:
            await cleanup()
        pool._stop_process.assert_awaited_once()
    asyncio.run(run())


@pytest.mark.parametrize('field', ['active_requests', 'queued_requests', 'control_leases'])
def test_unload_still_rejects_inference_and_mutating_controls(field):
    async def run():
        pool, instance, _, _ = make_pool()
        setattr(instance, field, 1)
        with pytest.raises(JobExecutionError) as error:
            await pool.unload(Context(), {'instance_id': str(instance.id)})
        assert error.value.detail.code == 'runtime_busy'
        assert instance.state == RuntimeInstanceState.READY
        pool._stop_process.assert_not_called()
    asyncio.run(run())


def test_stalled_status_read_times_out_and_unload_finishes(monkeypatch):
    monkeypatch.setattr(runtime_module, '_RUNTIME_READ_TIMEOUT_SECONDS', 0.03)

    async def run():
        pool, instance, entered, _ = make_pool()
        read = asyncio.create_task(pool.runtime_status(instance.id))
        await asyncio.wait_for(entered.wait(), 1)
        unloading = asyncio.create_task(pool.unload(Context(), {'instance_id': str(instance.id)}))
        with pytest.raises(asyncio.TimeoutError):
            await read
        assert (await asyncio.wait_for(unloading, 1))['unloaded']
        pool._stop_process.assert_awaited_once()
    asyncio.run(run())


def test_cancelled_unload_keeps_reader_safe_and_cleanup_finishes_retirement():
    async def run():
        pool, instance, entered, release = make_pool()
        read = asyncio.create_task(pool.runtime_status(instance.id))
        await asyncio.wait_for(entered.wait(), 1)
        context = Context()
        unloading = asyncio.create_task(pool.unload(context, {'instance_id': str(instance.id)}))
        await asyncio.wait_for(context.started.wait(), 1)
        unloading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await unloading
        pool._stop_process.assert_not_called()
        release.set()
        await read
        for cleanup in context.cleanup:
            await asyncio.wait_for(cleanup(), 1)
        assert instance.id not in pool._instances
        pool._stop_process.assert_awaited_once()
    asyncio.run(run())
