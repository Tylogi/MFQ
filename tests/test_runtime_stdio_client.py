from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from mfq.server.runtime.client import BackendError, StdioRuntimeClient


class _FakeStdin:
    def __init__(
        self,
        stdout: asyncio.StreamReader,
        handler: Callable[[dict[str, Any]], list[dict[str, Any]]],
    ) -> None:
        self.stdout = stdout
        self.handler = handler
        self.closed = False

    def write(self, data: bytes) -> None:
        for raw in data.splitlines():
            request = json.loads(raw)
            for response in self.handler(request):
                self.stdout.feed_data(
                    (json.dumps(response, separators=(",", ":")) + "\n").encode()
                )

    async def drain(self) -> None:
        return None

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True
        self.stdout.feed_eof()

    async def wait_closed(self) -> None:
        return None


def _client(
    handler: Callable[[dict[str, Any]], list[dict[str, Any]]],
) -> StdioRuntimeClient:
    stdout = asyncio.StreamReader()
    stdout.feed_data(b'{"v":1,"type":"ready"}\n')
    process = SimpleNamespace(pid=123, stdout=stdout, stdin=_FakeStdin(stdout, handler))
    return StdioRuntimeClient(process)


def test_stdio_runtime_client_handles_unary_and_streaming_frames() -> None:
    async def run() -> None:
        generated_request: dict[str, Any] = {}

        def handler(request: dict[str, Any]) -> list[dict[str, Any]]:
            request_id = request.get("id")
            if request["op"] == "health":
                return [
                    {
                        "v": 1,
                        "id": request_id,
                        "type": "result",
                        "data": {"status": "ok", "model": "tiny"},
                    }
                ]
            if request["op"] == "generate":
                generated_request.update(request)
                return [
                    {
                        "v": 1,
                        "id": request_id,
                        "type": "event",
                        "data": {
                            "event": "delta",
                            "request_id": "run-1",
                            "created": 1,
                            "model": "tiny",
                            "delta": {"content": "hello"},
                        },
                    },
                    {"v": 1, "id": request_id, "type": "done"},
                ]
            return []

        client = _client(handler)
        assert await client.health() == {"status": "ok", "model": "tiny"}
        async with client.generate(
            {
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "stream": True,
            }
        ) as events:
            assert [event async for event in events] == [
                {
                    "id": "run-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "tiny",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "hello"},
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                },
                None,
            ]
        assert generated_request["params"] == {
            "model": "tiny",
            "input": {"messages": [{"role": "user", "content": "hi"}]},
            "sampling": {"max_new_tokens": 8},
            "stream": True,
            "include_usage": False,
        }
        await client.aclose()

    asyncio.run(run())


def test_stdio_runtime_client_maps_structured_errors() -> None:
    async def run() -> None:
        def handler(request: dict[str, Any]) -> list[dict[str, Any]]:
            return [
                {
                    "v": 1,
                    "id": request["id"],
                    "type": "error",
                    "error": {
                        "code": "conflict",
                        "message": "busy",
                        "status_code": 409,
                        "retryable": False,
                    },
                }
            ]

        client = _client(handler)
        with pytest.raises(BackendError) as raised:
            await client.status()
        assert raised.value.code == "conflict"
        assert raised.value.status_code == 409
        await client.aclose()

    asyncio.run(run())


def test_stdio_runtime_client_fails_pending_requests_on_eof() -> None:
    async def run() -> None:
        client = _client(lambda _request: [])
        request = asyncio.create_task(client.health())
        await asyncio.sleep(0)
        assert client._process.stdout is not None
        client._process.stdout.feed_eof()
        with pytest.raises(BackendError, match="closed stdout"):
            await request
        await client.aclose()

    asyncio.run(run())
