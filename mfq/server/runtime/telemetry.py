"""Persist per-request outcomes independently of status reads and API transports."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from mfq.server.runtime.backend import BackendDelta, BackendError, ChatBackend, closing_backend_stream
from mfq.server.state.storage import SessionStore


class ObservedBackend:
    """Decorate the shared backend so every API consumer records terminal outcomes."""

    def __init__(self, backend: ChatBackend, store: SessionStore) -> None:
        self.backend = backend
        self.store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    async def stream(self, **kwargs: Any) -> AsyncIterator[BackendDelta]:
        """Capture metrics from this stream, then commit before reporting exhaustion."""
        request_id = str(uuid4())
        started = time.time()
        metrics: dict[str, Any] = {}
        instance_id = None
        source = 'default'
        status = 'failed'
        error = None
        finished = False
        try:
            async with closing_backend_stream(self.backend.stream(**kwargs)) as stream:
                async for delta in stream:
                    request_id = delta.backend_request_id or request_id
                    instance_id = delta.runtime_instance_id or instance_id
                    source = delta.runtime_source or source
                    if delta.usage is not None:
                        metrics.update(delta.usage.model_dump(mode='json'))
                    if delta.performance is not None:
                        metrics.update(delta.performance.model_dump(mode='json'))
                    if delta.finish_reason is not None:
                        metrics['finish_reason'] = delta.finish_reason
                        finished = True
                    yield delta
            if not finished:
                raise BackendError('backend_protocol_error', 'backend stream completed without a finish reason')
            status = 'completed'
        except (asyncio.CancelledError, GeneratorExit):
            status = 'cancelled'
            raise
        except Exception as cause:
            error = str(cause)
            raise
        finally:
            metrics.update(id=request_id, status=status, started_at=started, completed_at=time.time())
            if error is not None:
                metrics['error'] = error
            # A cancelled consumer must not cancel the short durable write.
            write = asyncio.create_task(asyncio.to_thread(
                self.store.append_runtime_request,
                {'last_request': metrics},
                instance_id=instance_id,
                model=kwargs.get('model'),
                source=source,
            ))
            try:
                await asyncio.shield(write)
            except asyncio.CancelledError:
                await write
                raise
