"""Regress durable request completion, concurrent metrics, and replayable telemetry SSE."""

import asyncio
from uuid import UUID

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.protocol.models import RuntimeLogLevel, TokenUsage
from mfq.server.runtime.backend import BackendDelta, BackendError
from mfq.server.services.service import ServerService
from mfq.server.services.telemetry_stream import stream_history
from mfq.server.state.storage import SessionStore


class RequestBackend:
    """Return request-local metrics without offering a status endpoint."""

    async def stream(self, **kwargs):
        number = int(kwargs['messages'][0]['content'])
        await asyncio.sleep(0)
        yield BackendDelta(content_delta='ok', backend_request_id=f'request-{number}',
                           usage=TokenUsage(prompt_tokens=number, completion_tokens=1, total_tokens=number + 1),
                           finish_reason='stop')

    async def aclose(self):
        pass


def test_all_requests_persist_without_status_reads(tmp_path):
    async def run():
        store = SessionStore(tmp_path / 'requests.sqlite3')
        service = ServerService(store, RequestBackend())
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url='http://test') as client:
                async def complete(number):
                    result = await client.post('/v1/chat/completions', json={
                        'model': 'test', 'messages': [{'role': 'user', 'content': str(number)}],
                        'stream': number % 2 == 0,
                    })
                    assert result.status_code == 200, result.text
                await asyncio.gather(*(complete(number) for number in range(1, 9)))
                for number in range(9, 12):
                    await complete(number)
                rows = (await client.get('/api/v1/runtime/requests')).json()['data']
                assert len(rows) == 11
                for row in rows:
                    request = row['values']['last_request']
                    assert request['prompt_tokens'] == int(request['id'].split('-')[1])
                    assert request['status'] == 'completed'
                assert store.list_runtime_metrics() == []
        finally:
            await service.aclose()
        assert len(SessionStore(store.path).list_runtime_requests()) == 11
    asyncio.run(run())


def test_idempotent_requests_are_scoped_to_instances(tmp_path):
    store = SessionStore(tmp_path / 'scope.sqlite3')
    values = {'last_request': {'id': 'same'}}
    first = store.append_runtime_request(values, instance_id=UUID(int=1))
    assert store.append_runtime_request(values, instance_id=UUID(int=1)).sequence == first.sequence
    store.append_runtime_request(values, instance_id=UUID(int=2))
    assert len(store.list_runtime_requests()) == 2
    store.append_runtime_metric(values)
    assert len(store.list_runtime_requests()) == 2


@pytest.mark.parametrize('channel', ['logs', 'requests'])
def test_sse_replays_wakes_reconnects_and_cleans_up(tmp_path, channel):
    async def run():
        store = SessionStore(tmp_path / 'stream.sqlite3')
        def append(number):
            if channel == 'logs':
                return store.append_runtime_log(RuntimeLogLevel.INFO, str(number))
            return store.append_runtime_request({'last_request': {'id': str(number)}})
        read = store.list_runtime_logs if channel == 'logs' else store.list_runtime_requests
        append(1)
        stream = stream_history(store, channel, read, heartbeat_seconds=0.02)
        first = await anext(stream)
        assert 'id: 1\n' in first
        assert await anext(stream) == ': keep-alive\n\n'
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        await asyncio.to_thread(append, 2)
        event = await asyncio.wait_for(pending, 1)
        assert 'id: 2\n' in event
        await stream.aclose()
        assert not store._telemetry_subscribers[channel]
        append(3)
        append(4)
        resumed = stream_history(store, channel, read, after=2)
        assert 'id: 3\n' in await anext(resumed)
        assert 'id: 4\n' in await anext(resumed)
        await resumed.aclose()
        assert not store._telemetry_subscribers[channel]
    asyncio.run(run())


@pytest.mark.parametrize('cancel', [False, True])
def test_failure_and_cancellation_persist_partial_metrics(tmp_path, cancel):
    class FailingBackend(RequestBackend):
        async def stream(self, **kwargs):
            yield BackendDelta(backend_request_id='partial', content_delta='a')
            raise BackendError('failed', 'test failure')

    async def run():
        service = ServerService(SessionStore(tmp_path / 'failed.sqlite3'), FailingBackend())
        stream = service.backend.stream(model='test')
        try:
            await anext(stream)
            if cancel:
                await stream.aclose()
            else:
                with pytest.raises(BackendError):
                    await anext(stream)
            rows = service.store.list_runtime_requests()
            assert len(rows) == 1
            metrics = rows[0].values['last_request']
            assert metrics['status'] == ('cancelled' if cancel else 'failed')
            assert 'completion_tokens' not in metrics
        finally:
            await service.aclose()
    asyncio.run(run())


def test_legacy_snapshot_migration_runs_once(tmp_path):
    path = tmp_path / 'migration.sqlite3'
    store = SessionStore(path)
    for _ in range(2):
        store.append_runtime_metric({'last_request': {'id': 'old'}})
    with store._connection() as connection:
        connection.execute("UPDATE schema_meta SET value = '18' WHERE key = 'schema_version'")
    migrated = SessionStore(path)
    assert len(migrated.list_runtime_requests()) == 1
    assert len(SessionStore(path).list_runtime_requests()) == 1
    migrated.append_runtime_request({'last_request': {'id': 'old'}})
    assert len(migrated.list_runtime_requests()) == 1


def test_sse_replays_more_than_one_page_and_applies_header_and_filters(tmp_path):
    from mfq.server.api.routes.runtime import stream_runtime_logs

    async def run():
        service = ServerService(SessionStore(tmp_path / 'burst.sqlite3'), RequestBackend())
        store = service.store
        try:
            for number in range(205):
                store.append_runtime_log(RuntimeLogLevel.INFO, str(number), instance_id=UUID(int=1))
            store.append_runtime_log(RuntimeLogLevel.ERROR, 'filtered', instance_id=UUID(int=2))
            response = await stream_runtime_logs(service, after=1, last_event_id=2,
                                                 instance_id=UUID(int=1), level=RuntimeLogLevel.INFO)
            assert response.media_type == 'text/event-stream'
            assert response.headers['x-accel-buffering'] == 'no'
            stream = response.body_iterator
            for sequence in range(3, 206):
                assert f'id: {sequence}\n' in await anext(stream)
            await stream.aclose()
            assert not store._telemetry_subscribers['logs']
        finally:
            await service.aclose()
    asyncio.run(run())


def test_native_response_replay_does_not_duplicate_telemetry(tmp_path):
    class NativeBackend(RequestBackend):
        async def stream(self, **kwargs):
            yield BackendDelta(content_delta='ok', backend_request_id='native', finish_reason='stop')

    async def run():
        service = ServerService(SessionStore(tmp_path / 'native.sqlite3'), NativeBackend())
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url='http://test') as client:
                session = (await client.post('/api/v1/sessions', json={'model': 'test', 'mode': 'text'})).json()
                body = {'request_id': str(UUID(int=1)), 'expected_revision': 0,
                        'input': [{'type': 'text', 'text': 'hello'}], 'stream': True}
                url = f"/api/v1/sessions/{session['id']}/responses"
                for _ in range(2):
                    response = await client.post(url, json=body)
                    assert response.status_code == 200, response.text
                assert len(service.store.list_runtime_requests()) == 1
        finally:
            await service.aclose()
    asyncio.run(run())
