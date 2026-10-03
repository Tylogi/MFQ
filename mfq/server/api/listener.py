from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import uvicorn


def saved_service_port(path: Path) -> int:
    try:
        port = json.loads(path.read_text(encoding="utf-8"))["port"]
        if isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535:
            return port
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return 8090


class RuntimeServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, settings_path: Path | None = None) -> None:
        super().__init__(config)
        self.settings_path = settings_path
        self._listen_lock = asyncio.Lock()
        config.app.state.listener = self

    def listener_status(self) -> dict[str, Any]:
        return {"host": self.config.host, "port": self.config.port, "configurable": self.started}

    def _save_port(self, port: int) -> None:
        if self.settings_path is None:
            return
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=self.settings_path.parent, prefix=".listener-")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"port": port}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.settings_path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    async def change_port(self, port: int) -> dict[str, Any]:
        if isinstance(port, bool) or not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        async with self._listen_lock:
            if not self.started or self.should_exit:
                raise RuntimeError("server listener is not running")
            if port == self.config.port:
                return self.listener_status()
            config = self.config
            listener = await asyncio.get_running_loop().create_server(
                lambda: config.http_protocol_class(
                    config=config, server_state=self.server_state, app_state=self.lifespan.state,
                ),
                host=config.host, port=port, ssl=config.ssl, backlog=config.backlog, start_serving=False,
            )
            try:
                await asyncio.to_thread(self._save_port, port)
                await listener.start_serving()
            except BaseException:
                listener.close()
                await listener.wait_closed()
                raise
            previous = self.servers
            self.servers = [listener]
            config.port = port
            for server in previous:
                server.close()
            return self.listener_status()
