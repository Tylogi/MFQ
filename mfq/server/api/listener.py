from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import uvicorn


def saved_service_port(path: Path, protocol: str = "openai") -> int:
    try:
        port = json.loads(path.read_text(encoding="utf-8"))["anthropic_port" if protocol == "anthropic" else "port"]
        if isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535:
            return port
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return 8091 if protocol == "anthropic" else 8090


class RuntimeServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, settings_path: Path | None = None, *, anthropic_port: int | None = None) -> None:
        super().__init__(config)
        self.settings_path = settings_path
        self.anthropic_port = anthropic_port
        self._listen_lock = asyncio.Lock()
        config.app.state.listener = self

    def listener_status(self) -> dict[str, Any]:
        return {"host": self.config.host, "port": self.config.port, "anthropic_port": self.anthropic_port, "configurable": self.started}

    async def _new_listener(self, port: int) -> asyncio.Server:
        config = self.config
        return await asyncio.get_running_loop().create_server(
            lambda: config.http_protocol_class(
                config=config, server_state=self.server_state, app_state=self.lifespan.state,
            ),
            host=config.host, port=port, ssl=config.ssl, backlog=config.backlog, start_serving=False,
        )

    async def startup(self, sockets=None) -> None:
        await super().startup(sockets)
        self.started = False
        if self.config.port == 0 and self.servers[0].sockets:
            self.config.port = self.servers[0].sockets[0].getsockname()[1]
        try:
            if self.anthropic_port is not None:
                listener = await self._new_listener(self.anthropic_port)
                self.servers.append(listener)
                await listener.start_serving()
                self.anthropic_port = listener.sockets[0].getsockname()[1]
        except BaseException:
            for listener in self.servers:
                listener.close()
                await listener.wait_closed()
            await self.lifespan.shutdown()
            raise
        self.started = True

    def _save_port(self, port: int, anthropic_port: int | None) -> None:
        if self.settings_path is None:
            return
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=self.settings_path.parent, prefix=".listener-")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"port": port, "anthropic_port": anthropic_port}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.settings_path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    async def change_port(self, port: int, protocol: str = "openai") -> dict[str, Any]:
        if isinstance(port, bool) or not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        async with self._listen_lock:
            if not self.started or self.should_exit:
                raise RuntimeError("server listener is not running")
            if protocol not in {"openai", "anthropic"}:
                raise ValueError("unsupported API protocol")
            index = 1 if protocol == "anthropic" else 0
            if index == 1 and self.anthropic_port is None:
                raise RuntimeError("Anthropic listener is not enabled")
            previous_port = self.anthropic_port if index else self.config.port
            if port == previous_port:
                return self.listener_status()
            listener = await self._new_listener(port)
            try:
                await listener.start_serving()
                await asyncio.to_thread(self._save_port, port if not index else self.config.port, port if index else self.anthropic_port)
            except BaseException:
                listener.close()
                await listener.wait_closed()
                raise
            previous = self.servers[index]
            self.servers[index] = listener
            if index:
                self.anthropic_port = port
            else:
                self.config.port = port
            previous.close()
            return self.listener_status()
