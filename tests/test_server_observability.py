"""Verify runtime telemetry persistence, retention, and cursor-based history access."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import httpx

import mfq.server.state.storage as storage_module
from mfq.server.api import create_app
from mfq.server.protocol.models import RuntimeLogLevel
from mfq.server.services.service import ServerService
from mfq.server.state.storage import SessionStore

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
                assert metrics.json()["data"] == []
                assert store.list_runtime_requests() == []

                logs = await client.get(
                    "/api/v1/runtime/logs",
                    params={"instance_id": str(INSTANCE_ID), "level": "info"},
                )
                assert logs.status_code == 200
                assert [entry["message"] for entry in logs.json()["data"]] == ["loaded"]
        finally:
            await service.aclose()

        reopened = SessionStore(tmp_path / "mfq.server.sqlite3")
        assert reopened.list_runtime_metrics() == []
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


def test_runtime_history_pages_latest_older_and_live_without_duplicate_requests(tmp_path: Path) -> None:
    """Page distinct requests and logs without gaps while new records arrive."""
    async def run() -> None:
        store = SessionStore(tmp_path / 'history.sqlite3')
        service = ServerService(store, StatusBackend())  # type: ignore[arg-type]
        for number in range(1, 126):
            store.append_runtime_log(RuntimeLogLevel.INFO, f'event-{number}', now=NOW)
            for _ in range(3):
                store.append_runtime_request({'last_request': {'id': f'request-{number}'}}, now=NOW)
        store.append_runtime_metric({'runtime_state': 'idle'}, now=NOW)
        transport = httpx.ASGITransport(app=create_app(service))
        try:
            async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
                for path in ['/api/v1/runtime/logs', '/api/v1/runtime/requests']:
                    first = (await client.get(path, params={'limit': 50, 'order': 'desc'})).json()['data']
                    assert len(first) == 50
                    assert first[0]['sequence'] > first[-1]['sequence']
                    second = (await client.get(path, params={
                        'limit': 50, 'order': 'desc', 'before': first[-1]['sequence'],
                    })).json()['data']
                    third = (await client.get(path, params={
                        'limit': 50, 'order': 'desc', 'before': second[-1]['sequence'],
                    })).json()['data']
                    assert len(second) == 50
                    assert len(third) == 25
                    combined = first + second + third
                    assert len({row['sequence'] for row in combined}) == 125
                    if path.endswith('requests'):
                        assert [row['values']['last_request']['id'] for row in combined] == [
                            f'request-{number}' for number in range(125, 0, -1)
                        ]
                    for params in [{'before': 0}, {'after': -1}, {'order': 'invalid'}]:
                        assert (await client.get(path, params=params)).status_code == 422

                latest = store.list_runtime_requests(limit=1)[0].sequence
                store.append_runtime_request({'last_request': {'id': 'request-125'}}, now=NOW)
                store.append_runtime_request({'last_request': {'id': 'request-126'}}, now=NOW)
                live = (await client.get('/api/v1/runtime/requests', params={
                    'after': latest, 'order': 'asc', 'limit': 50,
                })).json()['data']
                assert [row['values']['last_request']['id'] for row in live] == ['request-126']
                store.append_runtime_log(RuntimeLogLevel.INFO, 'event-126', now=NOW)
                live_logs = (await client.get('/api/v1/runtime/logs', params={'after': 125})).json()['data']
                assert [row['message'] for row in live_logs] == ['event-126']
                assert [row.sequence for row in store.list_runtime_logs(limit=3)] == [1, 2, 3]
        finally:
            await service.aclose()
    asyncio.run(run())
