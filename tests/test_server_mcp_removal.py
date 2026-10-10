import asyncio
import sqlite3

import httpx
import pytest

from mfq.server.api import create_contract_app
from mfq.server.api.openapi import build_openapi_schema
from mfq.server.api.routes.anthropic import router as anthropic_router
from mfq.server.api.routes.openai import router as openai_router
from mfq.server.state.storage import SessionStore


@pytest.mark.parametrize(("method", "path"), [
    ("GET", "/api/v1/mcp/servers"),
    ("POST", "/api/v1/mcp/servers"),
    ("PATCH", "/api/v1/mcp/servers/11111111-1111-4111-8111-111111111111"),
    ("DELETE", "/api/v1/mcp/servers/11111111-1111-4111-8111-111111111111"),
    ("GET", "/api/v1/mcp/tools"),
    ("POST", "/api/v1/mcp/tools/call"),
])
def test_removed_mcp_routes_are_not_available(method, path):
    async def check():
        transport = httpx.ASGITransport(app=create_contract_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.request(method, path)
        assert response.status_code == 404

    asyncio.run(check())


def test_contract_keeps_tool_calling_and_remote_routing_without_mcp():
    schema = build_openapi_schema()
    assert not any("/mcp/" in path for path in schema["paths"])
    assert not any("Mcp" in name for name in schema["components"]["schemas"])
    routes = {route.path for route in [*openai_router.routes, *anthropic_router.routes]}
    assert "/v1/chat/completions" in routes
    assert "/v1/messages" in routes
    assert "/api/v1/cluster/nodes" in schema["paths"]
    properties = schema["components"]["schemas"]["CreateResponseRequest"]["properties"]
    assert {"tools", "tool_choice"} <= properties.keys()


def test_new_store_does_not_create_a_mcp_registry(tmp_path):
    path = tmp_path / "new.sqlite3"
    SessionStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE name = 'mcp_servers'").fetchone() is None


def test_opening_a_legacy_store_preserves_existing_data(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE mcp_servers (name TEXT)")
        connection.execute("INSERT INTO mcp_servers VALUES ('legacy')")
    SessionStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT name FROM mcp_servers").fetchall() == [("legacy",)]
