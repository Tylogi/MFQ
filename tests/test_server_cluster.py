from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.runtime.backend import BackendDelta, BackendError
from mfq.server.runtime.cluster import ClusterBackend
from mfq.server.protocol.models import (
    CreateRemoteNodeRequest,
    SamplingParams,
    UpdateRemoteNodeRequest,
)
from mfq.server.services.service import ServerService
from mfq.server.state.storage import SessionStore
from tests.test_server_service import FakeBackend


def _remote_app(
    response_requests: list[dict[str, object]] | None = None,
    requested_paths: list[str] | None = None,
    *,
    legacy_models_endpoint: bool = False,
    runtime_status_code: int = 200,
    response_frames: list[dict[str, object]] | None = None,
):
    session_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal session_count
        path = request.url.path
        if requested_paths is not None:
            requested_paths.append(path)
        if path == "/health":
            return httpx.Response(200, json={"service": "mfq-server"})
        if path == "/v1/models":
            if legacy_models_endpoint:
                return httpx.Response(404)
            return httpx.Response(200, json={"data": [{"id": "remote-model"}]})
        if path == "/api/v1/runtime/models" and legacy_models_endpoint:
            return httpx.Response(200, json={"data": [{"id": "remote-model"}]})
        if path == "/api/v1/runtime/status":
            if runtime_status_code != 200:
                return httpx.Response(runtime_status_code)
            return httpx.Response(200, json={"total_requests": 7, "process_resident_bytes": 1024})
        if path == "/api/v1/media":
            assert request.headers["content-type"] == "image/png"
            assert request.content == b"image"
            return httpx.Response(
                201,
                json={
                    "media": {
                        "id": "66666666-6666-4666-8666-666666666666",
                        "sha256": hashlib.sha256(b"image").hexdigest(),
                        "mime_type": "image/png",
                        "byte_size": 5,
                    },
                    "created_at": "2026-08-12T00:00:00Z",
                },
            )
        if path == "/api/v1/sessions":
            remote_session_id = (
                "11111111-1111-4111-8111-"
                f"{111111111111 + session_count:012d}"
            )
            session_count += 1
            return httpx.Response(
                201,
                json={
                    "id": remote_session_id,
                    "model": "remote-model",
                    "mode": "text",
                    "state": "idle",
                    "revision": 0,
                    "title": None,
                    "runtime_instance_id": None,
                    "created_at": "2026-08-12T00:00:00Z",
                    "updated_at": "2026-08-12T00:00:00Z",
                    "metadata": {},
                },
            )
        if path.endswith("/responses"):
            if response_requests is not None:
                response_requests.append(json.loads(request.content))
            default_frames = [
                {
                    "protocol_version": "1.0",
                    "session_id": "11111111-1111-4111-8111-111111111111",
                    "sequence": 0,
                    "timestamp": "2026-08-12T00:00:00Z",
                    "payload": {
                        "type": "response.text.delta",
                        "response_id": "22222222-2222-4222-8222-222222222222",
                        "delta": "remote",
                    },
                },
                {
                    "protocol_version": "1.0",
                    "session_id": "11111111-1111-4111-8111-111111111111",
                    "sequence": 1,
                    "timestamp": "2026-08-12T00:00:00Z",
                    "payload": {
                        "type": "response.completed",
                        "response_id": "22222222-2222-4222-8222-222222222222",
                        "finish_reason": "stop",
                    },
                },
                {
                    "protocol_version": "1.0",
                    "session_id": "11111111-1111-4111-8111-111111111111",
                    "sequence": 2,
                    "timestamp": "2026-08-12T00:00:00Z",
                    "payload": {"type": "session.state", "state": "idle", "revision": 2},
                },
            ]
            frames = default_frames if response_frames is None else response_frames
            content = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
            return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})
        if path.endswith("/responses/cancel"):
            assert request.method == "POST"
            return httpx.Response(200, json={"status": "cancelled"})
        if path.endswith("/fork"):
            return httpx.Response(
                201,
                json={
                    "id": "44444444-4444-4444-8444-444444444444",
                    "model": "remote-model",
                    "mode": "text",
                    "state": "idle",
                    "revision": 2,
                    "title": None,
                    "runtime_instance_id": None,
                    "created_at": "2026-08-12T00:00:00Z",
                    "updated_at": "2026-08-12T00:00:00Z",
                    "metadata": {},
                },
            )
        if request.method == "DELETE" and "/api/v1/sessions/" in path:
            return httpx.Response(204)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_cluster_registers_probes_and_routes_matching_model(tmp_path: Path) -> None:
    async def run() -> None:
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        response_requests: list[dict[str, object]] = []
        requested_paths: list[str] = []
        client = httpx.AsyncClient(
            transport=_remote_app(response_requests, requested_paths)
        )
        local = FakeBackend()
        cluster = ClusterBackend(local, store, client=client)
        service = ServerService(store, cluster, cluster=cluster)
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            created = await api.post(
                "/api/v1/cluster/nodes",
                json={"name": "worker-a", "url": "http://worker-a:8090"},
            )
            assert created.status_code == 201
            assert created.json()["healthy"] is True
            assert created.json()["models"] == ["remote-model"]
            node_id = created.json()["id"]
            listed = await api.get("/api/v1/cluster/nodes?refresh=true")
            assert listed.json()["data"][0]["healthy"] is True
            assert listed.json()["data"][0]["metrics"]["total_requests"] == 7
            assert "/v1/models" in requested_paths
            assert "/api/v1/runtime/models" not in requested_paths
            models = await cluster.runtime_models()
            assert any(item["id"] == "remote-model" for item in models["data"])

            chunks = []
            async for delta in cluster.stream(
                model="remote-model",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,"
                                    + base64.b64encode(b"image").decode("ascii")
                                },
                            }
                        ],
                    }
                ],
                sampling=__import__(
                    "mfq.server.protocol.models", fromlist=["SamplingParams"]
                ).SamplingParams(),
                session_id=UUID("33333333-3333-4333-8333-333333333333"),
            ):
                chunks.append(delta)
            assert "".join(item.content_delta for item in chunks) == "remote"
            assert chunks[-1].finish_reason == "stop"
            assert response_requests[0]["sampling"] == {}
            replacement = [
                delta
                async for delta in cluster.stream(
                    model="remote-model",
                    messages=[{"role": "user", "content": "edited"}],
                    sampling=__import__(
                        "mfq.server.protocol.models", fromlist=["SamplingParams"]
                    ).SamplingParams(),
                    session_id=UUID("33333333-3333-4333-8333-333333333333"),
                )
            ]
            assert replacement[-1].finish_reason == "stop"
            assert requested_paths.count("/api/v1/sessions") == 2
            assert (
                "/api/v1/sessions/11111111-1111-4111-8111-111111111111"
                in requested_paths
            )
            remote = cluster._sessions[
                (
                    UUID(node_id),
                    UUID("33333333-3333-4333-8333-333333333333"),
                )
            ]
            assert remote.remote_id == UUID(
                "11111111-1111-4111-8111-111111111112"
            )
            assert await cluster.cancel_response(
                UUID("33333333-3333-4333-8333-333333333333")
            )
            state = cluster._states[UUID(node_id)]
            tool_result = await cluster._message_parts(
                state.resource,
                {"role": "tool", "tool_call_id": "call-1", "content": "done"},
                {},
            )
            assert tool_result == [
                {
                    "type": "tool_result",
                    "call_id": "call-1",
                    "result": "done",
                    "is_error": False,
                }
            ]
            tool_call = await cluster._message_parts(
                state.resource,
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "checking",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                        }
                    ],
                },
                {},
            )
            assert [part["type"] for part in tool_call] == ["reasoning", "tool_call"]
            assert tool_call[1]["arguments"] == {"q": "x"}
            status = await cluster.runtime_status()
            assert status["cluster_total_requests"] == 7
            assert status["cluster_process_resident_bytes"] == 1024
            forked_id = UUID("55555555-5555-4555-8555-555555555555")
            assert await cluster.fork_session(
                UUID("33333333-3333-4333-8333-333333333333"), forked_id
            )
            assert await cluster.close_session(forked_id)

            assert (await api.delete(f"/api/v1/cluster/nodes/{node_id}")).status_code == 204
            assert cluster._sessions == {}

        await client.aclose()

    asyncio.run(run())


def test_remote_node_configuration_never_persists_secret(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REMOTE_NODE_TOKEN", "private-token")
    store = SessionStore(tmp_path / "mfq.server.sqlite3")
    node = store.create_remote_node(
        __import__(
            "mfq.server.protocol.models", fromlist=["CreateRemoteNodeRequest"]
        ).CreateRemoteNodeRequest(
            name="secure", url="https://worker.example", api_key_env="REMOTE_NODE_TOKEN"
        )
    )
    assert node.api_key_env == "REMOTE_NODE_TOKEN"
    assert b"private-token" not in (tmp_path / "mfq.server.sqlite3").read_bytes()


def test_cluster_falls_back_to_legacy_runtime_inventory(tmp_path: Path) -> None:
    async def run() -> None:
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        requested_paths: list[str] = []
        client = httpx.AsyncClient(
            transport=_remote_app(
                requested_paths=requested_paths,
                legacy_models_endpoint=True,
            )
        )
        cluster = ClusterBackend(FakeBackend(), store, client=client)
        service = ServerService(store, cluster, cluster=cluster)
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            created = await api.post(
                "/api/v1/cluster/nodes",
                json={"name": "legacy-worker", "url": "http://legacy-worker:8090"},
            )
            assert created.status_code == 201
            assert created.json()["models"] == ["remote-model"]
            assert "/v1/models" in requested_paths
            assert "/api/v1/runtime/models" in requested_paths

        await client.aclose()

    asyncio.run(run())


def test_stateless_remote_stream_releases_ephemeral_session(tmp_path: Path) -> None:
    async def run() -> None:
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        requested_paths: list[str] = []
        client = httpx.AsyncClient(
            transport=_remote_app(requested_paths=requested_paths)
        )
        cluster = ClusterBackend(FakeBackend(), store, client=client)
        service = ServerService(store, cluster, cluster=cluster)
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            created = await api.post(
                "/api/v1/cluster/nodes",
                json={"name": "worker-a", "url": "http://worker-a:8090"},
            )
            assert created.status_code == 201

            chunks = [
                delta
                async for delta in cluster.stream(
                    model="remote-model",
                    messages=[{"role": "user", "content": "hello"}],
                    sampling=__import__(
                        "mfq.server.protocol.models", fromlist=["SamplingParams"]
                    ).SamplingParams(),
                )
            ]
            assert chunks[-1].finish_reason == "stop"
            assert cluster._sessions == {}
            assert (
                "/api/v1/sessions/11111111-1111-4111-8111-111111111111"
                in requested_paths
            )

            interrupted = cluster.stream(
                model="remote-model",
                messages=[{"role": "user", "content": "stop early"}],
                sampling=__import__(
                    "mfq.server.protocol.models", fromlist=["SamplingParams"]
                ).SamplingParams(),
            )
            assert (await anext(interrupted)).content_delta == "remote"
            await interrupted.aclose()
            assert cluster._sessions == {}

        await client.aclose()

    asyncio.run(run())


def test_interrupted_remote_stream_discards_persistent_session(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        partial = {
            "protocol_version": "1.0",
            "session_id": "11111111-1111-4111-8111-111111111111",
            "sequence": 0,
            "timestamp": "2026-08-12T00:00:00Z",
            "payload": {
                "type": "response.text.delta",
                "response_id": "22222222-2222-4222-8222-222222222222",
                "delta": "partial",
            },
        }
        requested_paths: list[str] = []
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        store.create_remote_node(
            CreateRemoteNodeRequest(name="worker-a", url="http://worker-a:8090")
        )
        client = httpx.AsyncClient(
            transport=_remote_app(
                requested_paths=requested_paths,
                response_frames=[partial],
            )
        )
        cluster = ClusterBackend(FakeBackend(), store, client=client)
        stream = cluster.stream(
            model="remote-model",
            messages=[{"role": "user", "content": "hello"}],
            sampling=SamplingParams(),
            session_id=UUID("33333333-3333-4333-8333-333333333333"),
        )
        assert (await anext(stream)).content_delta == "partial"
        with pytest.raises(BackendError) as interrupted:
            await anext(stream)
        assert interrupted.value.code == "remote_node_protocol_error"
        assert cluster._sessions == {}
        assert any(
            path.startswith("/api/v1/sessions/") and not path.endswith("/responses")
            for path in requested_paths
        )
        await client.aclose()

    asyncio.run(run())


def test_remote_completion_without_terminal_revision_is_not_reused(
    tmp_path: Path,
) -> None:
    completed = {
        "protocol_version": "1.0",
        "session_id": "11111111-1111-4111-8111-111111111111",
        "sequence": 0,
        "timestamp": "2026-08-12T00:00:00Z",
        "payload": {
            "type": "response.completed",
            "response_id": "22222222-2222-4222-8222-222222222222",
            "finish_reason": "stop",
        },
    }

    async def run() -> None:
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        store.create_remote_node(
            CreateRemoteNodeRequest(name="worker-a", url="http://worker-a:8090")
        )
        client = httpx.AsyncClient(
            transport=_remote_app(response_frames=[completed])
        )
        cluster = ClusterBackend(FakeBackend(), store, client=client)
        deltas = [
            delta
            async for delta in cluster.stream(
                model="remote-model",
                messages=[{"role": "user", "content": "hello"}],
                sampling=SamplingParams(),
                session_id=UUID("33333333-3333-4333-8333-333333333333"),
            )
        ]
        assert deltas[-1].finish_reason == "stop"
        assert cluster._sessions == {}
        await client.aclose()

    asyncio.run(run())


def test_cluster_routes_when_remote_metrics_are_unavailable(tmp_path: Path) -> None:
    async def run() -> None:
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        client = httpx.AsyncClient(
            transport=_remote_app(runtime_status_code=404)
        )
        cluster = ClusterBackend(FakeBackend(), store, client=client)
        service = ServerService(store, cluster, cluster=cluster)
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            created = await api.post(
                "/api/v1/cluster/nodes",
                json={"name": "worker-a", "url": "http://worker-a:8090"},
            )
            assert created.status_code == 201
            assert created.json()["healthy"] is True
            assert created.json()["models"] == ["remote-model"]
            assert created.json()["metrics"] == {}

        await client.aclose()

    asyncio.run(run())


def test_cluster_serializes_probe_with_node_configuration_changes(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        old_probe_started = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            path = request.url.path
            if path == "/health":
                if host == "old-worker":
                    old_probe_started.set()
                    await asyncio.sleep(0.05)
                return httpx.Response(200)
            if path == "/v1/models":
                model = "old-model" if host == "old-worker" else "new-model"
                return httpx.Response(200, json={"data": [{"id": model}]})
            if path == "/api/v1/runtime/status":
                return httpx.Response(200, json={})
            return httpx.Response(404)

        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        resource = store.create_remote_node(
            CreateRemoteNodeRequest(
                name="worker-a",
                url="http://old-worker:8090",
            )
        )
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cluster = ClusterBackend(FakeBackend(), store, client=client)

        old_refresh = asyncio.create_task(cluster.nodes(force=True))
        await old_probe_started.wait()
        await asyncio.to_thread(
            store.update_remote_node,
            resource.id,
            UpdateRemoteNodeRequest(
                name="worker-a",
                url="http://new-worker:8090",
            ),
        )
        new_refresh = asyncio.create_task(cluster.nodes(force=True))
        await asyncio.gather(old_refresh, new_refresh)

        [node] = await cluster.nodes()
        assert node.url == "http://new-worker:8090"
        assert node.models == ["new-model"]
        await client.aclose()

    asyncio.run(run())


def test_disabled_remote_node_stops_advertising_stale_models(tmp_path: Path) -> None:
    async def run() -> None:
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        client = httpx.AsyncClient(transport=_remote_app())
        cluster = ClusterBackend(FakeBackend(), store, client=client)
        service = ServerService(store, cluster, cluster=cluster)
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            created = await api.post(
                "/api/v1/cluster/nodes",
                json={"name": "worker-a", "url": "http://worker-a:8090"},
            )
            node_id = created.json()["id"]
            assert created.json()["models"] == ["remote-model"]

            disabled = await api.put(
                f"/api/v1/cluster/nodes/{node_id}",
                json={
                    "name": "worker-a",
                    "url": "http://worker-a:8090",
                    "enabled": False,
                },
            )
            assert disabled.status_code == 200
            assert disabled.json()["healthy"] is False
            assert disabled.json()["models"] == []
            models = await cluster.runtime_models()
            assert all(item["id"] != "remote-model" for item in models["data"])

        await client.aclose()

    asyncio.run(run())


def test_cluster_bounds_control_and_media_requests(tmp_path: Path) -> None:
    async def run() -> None:
        observed_timeouts: dict[str, float] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            timeout = request.extensions.get("timeout", {})
            observed_timeouts[request.url.path] = float(timeout["read"])
            if request.url.path == "/api/v1/sessions":
                raise httpx.ReadTimeout("remote control stalled", request=request)
            if request.url.path == "/api/v1/media":
                return httpx.Response(
                    201,
                    json={
                        "media": {
                            "id": "66666666-6666-4666-8666-666666666666",
                            "sha256": hashlib.sha256(b"image").hexdigest(),
                            "mime_type": "image/png",
                            "byte_size": 5,
                        }
                    },
                )
            return httpx.Response(404)

        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        node = store.create_remote_node(
            CreateRemoteNodeRequest(name="worker-a", url="http://worker-a:8090")
        )
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cluster = ClusterBackend(
            FakeBackend(),
            store,
            client=client,
            control_timeout_seconds=7,
            media_upload_timeout_seconds=11,
        )

        with pytest.raises(BackendError) as stalled:
            await cluster._request_json(
                node,
                "POST",
                "/api/v1/sessions",
                {"model": "remote-model"},
                {},
            )
        assert stalled.value.code == "remote_node_timeout"
        assert stalled.value.retryable
        assert stalled.value.status_code == 504

        media = await cluster._upload_bytes(node, b"image", "image/png", {})
        assert media["byte_size"] == 5
        assert observed_timeouts["/api/v1/sessions"] == 7
        assert observed_timeouts["/api/v1/media"] == 11
        await client.aclose()

    asyncio.run(run())


def test_cluster_close_serializes_with_refresh_and_is_idempotent(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        probe_started = asyncio.Event()
        release_probe = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                probe_started.set()
                await release_probe.wait()
                return httpx.Response(200)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "remote-model"}]})
            if request.url.path == "/api/v1/runtime/status":
                return httpx.Response(200, json={})
            return httpx.Response(404)

        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        store.create_remote_node(
            CreateRemoteNodeRequest(name="worker-a", url="http://worker-a:8090")
        )
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        local = FakeBackend()
        cluster = ClusterBackend(local, store, client=client)

        refresh = asyncio.create_task(cluster.nodes(force=True))
        await probe_started.wait()
        close = asyncio.create_task(cluster.aclose())
        await asyncio.sleep(0)
        assert not close.done()
        release_probe.set()
        [node] = await refresh
        assert node.models == ["remote-model"]
        await close
        await cluster.aclose()

        with pytest.raises(BackendError) as closed:
            await cluster.nodes()
        assert closed.value.code == "backend_closed"
        assert closed.value.status_code == 503
        await client.aclose()

    asyncio.run(run())


def test_retired_node_sessions_are_released_concurrently(tmp_path: Path) -> None:
    async def run() -> None:
        active = 0
        maximum_active = 0
        all_started = asyncio.Event()
        release = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal active, maximum_active
            if request.method != "DELETE":
                return httpx.Response(404)
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 3:
                all_started.set()
            try:
                await release.wait()
            finally:
                active -= 1
            return httpx.Response(204)

        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        node = store.create_remote_node(
            CreateRemoteNodeRequest(
                name="worker-a",
                url="http://worker-a:8090",
            )
        )
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cluster = ClusterBackend(FakeBackend(), store, client=client)
        for _ in range(3):
            local_id = uuid4()
            cluster._sessions[(node.id, local_id)] = (
                cluster._remote_session_from_payload(
                    node,
                    {"id": str(uuid4()), "revision": 0},
                    synchronized_messages=0,
                )
            )

        cleanup = asyncio.create_task(cluster._release_node_sessions(node))
        await asyncio.wait_for(all_started.wait(), timeout=1)
        assert maximum_active == 3
        release.set()
        await asyncio.wait_for(cleanup, timeout=1)
        assert cluster._sessions == {}
        await client.aclose()

    asyncio.run(run())


def test_cluster_fails_over_before_the_first_remote_delta(tmp_path: Path) -> None:
    async def run() -> None:
        requested_hosts: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            path = request.url.path
            requested_hosts.append(f"{host}{path}")
            if path == "/health":
                return httpx.Response(200)
            if path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "remote-model"}]})
            if path == "/api/v1/runtime/status":
                return httpx.Response(200, json={})
            if path == "/api/v1/sessions" and host == "a-broken":
                raise httpx.ConnectError("worker disappeared", request=request)
            if path == "/api/v1/sessions":
                return httpx.Response(
                    201,
                    json={
                        "id": "11111111-1111-4111-8111-111111111111",
                        "revision": 0,
                    },
                )
            if path.endswith("/responses"):
                frames = [
                    {"payload": {"type": "response.text.delta", "delta": "backup"}},
                    {
                        "payload": {
                            "type": "response.completed",
                            "finish_reason": "stop",
                        }
                    },
                    {"payload": {"type": "session.state", "revision": 2}},
                ]
                return httpx.Response(
                    200,
                    text="".join(f"data: {json.dumps(frame)}\n\n" for frame in frames),
                )
            if request.method == "DELETE":
                return httpx.Response(204)
            return httpx.Response(404)

        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        broken = store.create_remote_node(
            CreateRemoteNodeRequest(name="a-broken", url="http://a-broken:8090")
        )
        backup = store.create_remote_node(
            CreateRemoteNodeRequest(name="b-backup", url="http://b-backup:8090")
        )
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cluster = ClusterBackend(FakeBackend(), store, client=client)

        chunks = [
            delta
            async for delta in cluster.stream(
                model="remote-model",
                messages=[{"role": "user", "content": "hello"}],
                sampling=__import__(
                    "mfq.server.protocol.models", fromlist=["SamplingParams"]
                ).SamplingParams(),
                session_id=UUID("33333333-3333-4333-8333-333333333333"),
            )
        ]

        assert "".join(item.content_delta for item in chunks) == "backup"
        assert chunks[-1].finish_reason == "stop"
        assert "a-broken/api/v1/sessions" in requested_hosts
        assert "b-backup/api/v1/sessions" in requested_hosts
        assert cluster._states[broken.id].healthy is False
        assert cluster._states[broken.id].models == []
        assert cluster._states[backup.id].healthy is True
        assert cluster._states[broken.id].active_requests == 0
        assert cluster._states[backup.id].active_requests == 0
        await cluster.aclose()
        await client.aclose()

    asyncio.run(run())


def test_cluster_never_replays_after_a_remote_delta(tmp_path: Path) -> None:
    async def run() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "remote-model"}]})
            if request.url.path == "/api/v1/runtime/status":
                return httpx.Response(200, json={})
            return httpx.Response(404)

        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        store.create_remote_node(
            CreateRemoteNodeRequest(name="worker-a", url="http://worker-a:8090")
        )
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        local = FakeBackend(
            (BackendDelta(content_delta="must-not-run", finish_reason="stop"),)
        )
        cluster = ClusterBackend(local, store, client=client)

        async def interrupted_remote(*_args: object, **_options: object):
            yield BackendDelta(content_delta="partial")
            raise BackendError(
                "remote_node_unavailable",
                "connection dropped",
                retryable=True,
                status_code=502,
            )

        cluster._remote_stream = interrupted_remote  # type: ignore[method-assign]
        received: list[BackendDelta] = []
        with pytest.raises(BackendError) as interrupted:
            async for delta in cluster.stream(
                model="remote-model",
                messages=[{"role": "user", "content": "hello"}],
                sampling=__import__(
                    "mfq.server.protocol.models", fromlist=["SamplingParams"]
                ).SamplingParams(),
            ):
                received.append(delta)

        assert interrupted.value.code == "remote_node_unavailable"
        assert [item.content_delta for item in received] == ["partial"]
        assert local.calls == []
        await cluster.aclose()
        await client.aclose()

    asyncio.run(run())


def test_remote_node_reconfiguration_retires_bound_sessions(tmp_path: Path) -> None:
    async def run() -> None:
        requests: list[tuple[str, str, str]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            path = request.url.path
            requests.append((host, request.method, path))
            if path == "/health":
                return httpx.Response(200)
            if path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "remote-model"}]})
            if path == "/api/v1/runtime/status":
                return httpx.Response(200, json={})
            if path == "/api/v1/sessions":
                remote_id = (
                    "11111111-1111-4111-8111-111111111111"
                    if host == "old-worker"
                    else "22222222-2222-4222-8222-222222222222"
                )
                return httpx.Response(201, json={"id": remote_id, "revision": 0})
            if path.endswith("/responses"):
                frames = [
                    {"payload": {"type": "response.text.delta", "delta": host}},
                    {"payload": {"type": "response.completed", "finish_reason": "stop"}},
                    {"payload": {"type": "session.state", "revision": 2}},
                ]
                return httpx.Response(
                    200,
                    text="".join(f"data: {json.dumps(frame)}\n\n" for frame in frames),
                    headers={"content-type": "text/event-stream"},
                )
            if request.method == "DELETE" and "/api/v1/sessions/" in path:
                return httpx.Response(204)
            return httpx.Response(404)

        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        node = store.create_remote_node(
            CreateRemoteNodeRequest(name="worker-a", url="http://old-worker:8090")
        )
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cluster = ClusterBackend(FakeBackend(), store, client=client)
        session_id = UUID("33333333-3333-4333-8333-333333333333")

        old = [
            delta
            async for delta in cluster.stream(
                model="remote-model",
                messages=[{"role": "user", "content": "hello"}],
                sampling=SamplingParams(),
                session_id=session_id,
            )
        ]
        assert "".join(delta.content_delta for delta in old) == "old-worker"
        assert cluster._sessions[(node.id, session_id)].node.url == (
            "http://old-worker:8090"
        )

        await asyncio.to_thread(
            store.update_remote_node,
            node.id,
            UpdateRemoteNodeRequest(
                name="worker-a",
                url="http://new-worker:8090",
            ),
        )
        await cluster.nodes(force=True)
        assert cluster._sessions == {}
        assert (
            "old-worker",
            "DELETE",
            "/api/v1/sessions/11111111-1111-4111-8111-111111111111",
        ) in requests

        new = [
            delta
            async for delta in cluster.stream(
                model="remote-model",
                messages=[{"role": "user", "content": "hello again"}],
                sampling=SamplingParams(),
                session_id=session_id,
            )
        ]
        assert "".join(delta.content_delta for delta in new) == "new-worker"
        assert cluster._sessions[(node.id, session_id)].node.url == (
            "http://new-worker:8090"
        )
        await cluster.aclose()
        await client.aclose()

    asyncio.run(run())


def test_session_creation_retries_after_node_reconfiguration(tmp_path: Path) -> None:
    async def run() -> None:
        old_creation_started = asyncio.Event()
        release_old_creation = asyncio.Event()
        requests: list[tuple[str, str, str]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            path = request.url.path
            requests.append((host, request.method, path))
            if path == "/health":
                return httpx.Response(200)
            if path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "remote-model"}]})
            if path == "/api/v1/runtime/status":
                return httpx.Response(200, json={})
            if path == "/api/v1/sessions":
                if host == "old-worker":
                    old_creation_started.set()
                    await release_old_creation.wait()
                remote_id = (
                    "11111111-1111-4111-8111-111111111111"
                    if host == "old-worker"
                    else "22222222-2222-4222-8222-222222222222"
                )
                return httpx.Response(201, json={"id": remote_id, "revision": 0})
            if path.endswith("/responses"):
                frames = [
                    {"payload": {"type": "response.text.delta", "delta": host}},
                    {"payload": {"type": "response.completed", "finish_reason": "stop"}},
                    {"payload": {"type": "session.state", "revision": 2}},
                ]
                return httpx.Response(
                    200,
                    text="".join(f"data: {json.dumps(frame)}\n\n" for frame in frames),
                    headers={"content-type": "text/event-stream"},
                )
            if request.method == "DELETE" and "/api/v1/sessions/" in path:
                return httpx.Response(204)
            return httpx.Response(404)

        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        node = store.create_remote_node(
            CreateRemoteNodeRequest(name="worker-a", url="http://old-worker:8090")
        )
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        local = FakeBackend()
        cluster = ClusterBackend(local, store, client=client)
        await cluster.nodes(force=True)
        session_id = UUID("33333333-3333-4333-8333-333333333333")

        async def collect() -> list[BackendDelta]:
            return [
                delta
                async for delta in cluster.stream(
                    model="remote-model",
                    messages=[{"role": "user", "content": "hello"}],
                    sampling=SamplingParams(),
                    session_id=session_id,
                )
            ]

        response = asyncio.create_task(collect())
        await old_creation_started.wait()
        await asyncio.to_thread(
            store.update_remote_node,
            node.id,
            UpdateRemoteNodeRequest(
                name="worker-a",
                url="http://new-worker:8090",
            ),
        )
        await cluster.nodes(force=True)
        release_old_creation.set()
        received = await response

        assert "".join(delta.content_delta for delta in received) == "new-worker"
        assert local.calls == []
        assert cluster._sessions[(node.id, session_id)].node.url == (
            "http://new-worker:8090"
        )
        assert (
            "old-worker",
            "DELETE",
            "/api/v1/sessions/11111111-1111-4111-8111-111111111111",
        ) in requests
        await cluster.aclose()
        await client.aclose()

    asyncio.run(run())
