from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.api.auth import ApiKeyManager
from mfq.server.protocol.models import RuntimeInstanceState
from mfq.server.runtime.runtime_pool import RuntimePool
from mfq.server.services.service import ServerService
from mfq.server.state.storage import SessionStore
from tests.test_server_models import IdleBackend


def record(store, id, *, instance=None, completed=100, tokens=101, prefill=100, prefill_ms=50, decode_ms=1000):
    request = {"id": id, "completed_at": completed, "finish_reason": "stop", "prompt_tokens": 200,
        "prefill_tokens": prefill, "completion_tokens": tokens, "prefill_ms": prefill_ms,
        "decode_ms": decode_ms, "ttft_ms": 100}
    store.append_runtime_metric({"last_request": request}, instance_id=instance, model="model")
    return request


def test_serving_statistics_deduplicate_persist_and_weight_timings(tmp_path):
    path = tmp_path / 'stats.sqlite3'
    store = SessionStore(path)
    instance = uuid4()
    request = record(store, 'one', instance=instance)
    for _ in range(3):
        store.append_runtime_metric({'last_request': request}, instance_id=instance, model='model')
    record(store, 'two', instance=instance, completed=200, tokens=201, prefill_ms=150, decode_ms=4000)
    result = SessionStore(path).serving_statistics()
    assert result['completed_requests'] == 2
    assert result['decode_tps'] == pytest.approx(60)
    assert result['prefill_tps'] == pytest.approx(1000)
    assert result['cache_hit_rate'] == 50
    assert result['ttft_ms'] == 100
    assert store.serving_statistics(since=150)['completed_requests'] == 1
    assert store.serving_statistics(model='other')['completed_requests'] == 0


def test_reused_native_id_in_another_instance_is_a_distinct_request(tmp_path):
    store = SessionStore(tmp_path / 'stats.sqlite3')
    for _ in range(2):
        record(store, 'same', instance=uuid4())
    assert store.serving_statistics()['completed_requests'] == 2


def test_empty_and_untimed_requests_have_no_invented_rate(tmp_path):
    store = SessionStore(tmp_path / 'stats.sqlite3')
    assert store.serving_statistics()['decode_tps'] is None
    record(store, 'zero', tokens=1, prefill=0, decode_ms=0, prefill_ms=0)
    result = store.serving_statistics()
    assert result['decode_tps'] is None and result['prefill_tps'] is None
    store.append_runtime_metric({'last_request': {'id': 'cancelled', 'finish_reason': 'cancelled'}})
    assert store.serving_statistics()['completed_requests'] == 1


def test_serving_endpoint_separates_process_session_from_history(tmp_path):
    async def run():
        store = SessionStore(tmp_path / 'stats.sqlite3')
        record(store, 'historical')
        service = ServerService(store, IdleBackend())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url='http://test') as client:
            history = (await client.get('/api/v1/runtime/serving?scope=alltime')).json()
            current = (await client.get('/api/v1/runtime/serving?scope=session')).json()
            assert history['totals']['completed_requests'] == 1
            assert current['totals']['completed_requests'] == 0
            assert current['activity'] == []
            assert (await client.get('/api/v1/runtime/serving?scope=bad')).status_code == 422
            record(store, 'current', completed=datetime.now(timezone.utc).timestamp())
            current = (await client.get('/api/v1/runtime/serving')).json()
            assert current['totals']['completed_requests'] == 1
        await service.aclose()
    asyncio.run(run())


def test_existing_retained_metrics_are_backfilled_once_without_duplicate_requests(tmp_path):
    store = SessionStore(tmp_path / 'stats.sqlite3')
    request = record(store, 'existing')
    with store._connection() as connection:
        connection.execute('DELETE FROM serving_requests')
        connection.execute("DELETE FROM schema_meta WHERE key='serving_history_imported'")
        connection.execute('INSERT INTO runtime_metrics(model,values_json,captured_at) VALUES (?,?,?)',
            ('model', json.dumps({'last_request': request}), '2026-10-17T00:00:00+00:00'))
    assert SessionStore(store.path).serving_statistics()['completed_requests'] == 1
    assert SessionStore(store.path).serving_statistics()['completed_requests'] == 1


def test_live_request_snapshots_coalesce_pollers_and_never_wait_for_generation(tmp_path):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        class Backend:
            calls = 0
            async def runtime_status(self):
                self.calls += 1
                entered.set()
                await release.wait()
                return {'live_requests': {'native-id': {'generated_tokens': 24, 'phase': 'generating'}}}
        backend = Backend()
        instance = SimpleNamespace(id=uuid4(), artifact=SimpleNamespace(resource=SimpleNamespace(name='model')),
            state=RuntimeInstanceState.BUSY, active_requests=1, queued_requests=2,
            serving_snapshot={}, serving_sampled_at=0, serving_refreshing=False, backend=backend)
        pool = object.__new__(RuntimePool)
        pool._instances = {instance.id: instance}
        first = asyncio.create_task(pool.serving_activity())
        await entered.wait()
        assert (await pool.serving_activity())[0]['requests'] == []
        assert backend.calls == 1
        release.set()
        rows = await first
        assert rows[0]['requests'][0]['id'] == 'native-id'
        assert rows[0]['requests'][0]['generated_tokens'] == 24
        assert rows[0]['queued_requests'] == 2
        await pool.serving_activity()
        assert backend.calls == 1
        instance.active_requests = 0
        assert (await pool.serving_activity())[0]['requests'] == []
    asyncio.run(run())


def test_serving_statistics_respect_api_authentication(tmp_path):
    async def run():
        store = SessionStore(tmp_path / 'stats.sqlite3')
        service = ServerService(store, IdleBackend())
        app = create_app(service, api_keys=ApiKeyManager(store, 'test-serving-credential'))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            assert (await client.get('/api/v1/runtime/serving')).status_code == 401
            response = await client.get('/api/v1/runtime/serving', headers={'Authorization': 'Bearer test-serving-credential'})
            assert response.status_code == 200
        await service.aclose()
    asyncio.run(run())
