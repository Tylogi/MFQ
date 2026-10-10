import asyncio
import socket
from pathlib import Path

import httpx
import uvicorn

from mfq.server.api import create_app
from mfq.server.api.auth import required_scope
from mfq.server.api.listener import RuntimeServer, saved_service_port


def test_listener_changes_port_without_restarting_service(tmp_path: Path) -> None:
    async def run() -> None:
        app = create_app()
        starts = []
        state = {"model": "resident", "context": 16384}

        @app.get("/state")
        async def get_state():
            return state

        server = RuntimeServer(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"), tmp_path / "listener.json")
        task = asyncio.create_task(server.serve())
        try:
            for _ in range(200):
                if server.started:
                    break
                await asyncio.sleep(0.01)
            assert server.started
            initial = server.servers[0].sockets[0].getsockname()[1]
            server.config.port = initial
            starts.append(server.lifespan)
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                new_port = probe.getsockname()[1]
            async with httpx.AsyncClient(trust_env=False) as client:
                result = await client.put(f"http://127.0.0.1:{initial}/api/v1/runtime/listener", json={"port": new_port})
                assert result.status_code == 200
                assert result.json()["port"] == new_port
                assert (await client.get(f"http://127.0.0.1:{new_port}/state")).json() == state
                assert (await client.get(f"http://127.0.0.1:{new_port}/api/v1/runtime/listener")).json()["configurable"] is True
            assert server.lifespan is starts[0]
            assert saved_service_port(tmp_path / "listener.json") == new_port
            with socket.socket() as probe:
                probe.settimeout(1)
                assert probe.connect_ex(("127.0.0.1", initial)) != 0

            with socket.socket() as occupied:
                occupied.bind(("127.0.0.1", 0))
                occupied.listen()
                unavailable = occupied.getsockname()[1]
                async with httpx.AsyncClient(trust_env=False) as client:
                    failed = await client.put(f"http://127.0.0.1:{new_port}/api/v1/runtime/listener", json={"port": unavailable})
                    assert failed.status_code == 409
                    assert (await client.get(f"http://127.0.0.1:{new_port}/state")).json() == state
            assert server.config.port == new_port
            assert saved_service_port(tmp_path / "listener.json") == new_port
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 5)

    asyncio.run(run())


def test_listener_config_requires_admin_and_valid_port() -> None:
    async def run() -> None:
        app = create_app(api_key="test-key")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.put("/api/v1/runtime/listener", json={"port": 8091})).status_code == 401
            headers = {"Authorization": "Bearer test-key"}
            assert (await client.put("/api/v1/runtime/listener", json={"port": 0}, headers=headers)).status_code == 422
            assert (await client.put("/api/v1/runtime/listener", json={"port": 65536}, headers=headers)).status_code == 422
            assert (await client.put("/api/v1/runtime/listener", json={"port": True}, headers=headers)).status_code == 422
            assert (await client.put("/api/v1/runtime/listener", json={"port": "8091"}, headers=headers)).status_code == 422
            assert (await client.put("/api/v1/runtime/listener", json={"port": 8091, "protocol": "invalid"}, headers=headers)).status_code == 422
            unavailable = await client.put("/api/v1/runtime/listener", json={"port": 8091}, headers=headers)
            assert unavailable.status_code == 409
            assert unavailable.json()["error"]["code"] == "listener_not_managed"
        assert required_scope("PUT", "/api/v1/runtime/listener") == "admin"
        assert required_scope("GET", "/api/v1/runtime/listener") == "inference"

    asyncio.run(run())


def test_saved_listener_port_ignores_invalid_configuration(tmp_path: Path) -> None:
    path = tmp_path / "listener.json"
    assert saved_service_port(path) == 8090
    for payload in ['{"port": 0}', '{"port": true}', '{"port": "8091"}', '{}', 'invalid']:
        path.write_text(payload)
        assert saved_service_port(path) == 8090


def test_independent_listeners_keep_live_requests_and_shared_service(tmp_path: Path) -> None:
    async def run() -> None:
        app = create_app()
        release = asyncio.Event()
        entered = asyncio.Event()

        @app.get("/pending")
        async def pending():
            entered.set()
            await release.wait()
            return {"model": "still resident"}

        server = RuntimeServer(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"),
                               tmp_path / "listener.json", anthropic_port=0)
        task = asyncio.create_task(server.serve())
        try:
            for _ in range(200):
                if server.started:
                    break
                await asyncio.sleep(.01)
            assert server.started
            oai, anthropic = server.config.port, server.anthropic_port
            lifespan = server.lifespan
            async with httpx.AsyncClient(trust_env=False) as client:
                inflight = asyncio.create_task(client.get(f"http://127.0.0.1:{anthropic}/pending"))
                await asyncio.wait_for(entered.wait(), 2)
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", 0))
                    new_port = probe.getsockname()[1]
                response = await client.put(f"http://127.0.0.1:{oai}/api/v1/runtime/listener",
                                            json={"port": new_port, "protocol": "anthropic"})
                assert response.status_code == 200
                assert response.json()["port"] == oai
                assert response.json()["anthropic_port"] == new_port
                assert (await client.get(f"http://127.0.0.1:{oai}/v1")).json()["endpoints"][-1] == "/v1/chat/completions"
                assert (await client.get(f"http://127.0.0.1:{new_port}/v1")).json()["endpoints"][-1] == "/v1/messages"
                collision = await client.put(f"http://127.0.0.1:{oai}/api/v1/runtime/listener",
                                             json={"port": oai, "protocol": "anthropic"})
                assert collision.status_code == 409
                assert server.anthropic_port == new_port
                assert saved_service_port(tmp_path / "listener.json", "anthropic") == new_port
                assert saved_service_port(tmp_path / "listener.json") == oai
                release.set()
                assert (await inflight).json() == {"model": "still resident"}
            assert server.lifespan is lifespan
        finally:
            release.set()
            server.should_exit = True
            await asyncio.wait_for(task, 5)
        for port in (server.config.port, server.anthropic_port):
            with socket.socket() as probe:
                assert probe.connect_ex(("127.0.0.1", port)) != 0
    asyncio.run(run())


def test_port_persistence_failure_keeps_both_listeners(tmp_path: Path) -> None:
    async def run():
        app = create_app()
        server = RuntimeServer(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"),
                               tmp_path / "listener.json", anthropic_port=0)
        task = asyncio.create_task(server.serve())
        try:
            for _ in range(200):
                if server.started:
                    break
                await asyncio.sleep(.01)
            assert server.started
            original = server.listener_status()
            def fail(*args):
                raise OSError("configuration is read-only")
            server._save_port = fail
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                new_port = probe.getsockname()[1]
            async with httpx.AsyncClient(trust_env=False) as client:
                failed = await client.put(f"http://127.0.0.1:{server.config.port}/api/v1/runtime/listener",
                                          json={"port": new_port, "protocol": "anthropic"})
                assert failed.status_code == 409
                assert server.listener_status() == original
                for port in (server.config.port, server.anthropic_port):
                    assert (await client.get(f"http://127.0.0.1:{port}/health")).status_code == 200
            assert not (tmp_path / "listener.json").exists()
            with socket.socket() as probe:
                assert probe.connect_ex(("127.0.0.1", new_port)) != 0
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 5)
    asyncio.run(run())
