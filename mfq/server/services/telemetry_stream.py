"""Stream committed runtime history with shared replay, wakeup, and heartbeat logic."""

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any


async def stream_history(store: Any, channel: str, read: Callable[..., list], after: int = 0,
                         heartbeat_seconds: float = 15) -> AsyncIterator[str]:
    """Replay durable rows and wait for commits; cancellation releases the subscription."""
    cursor = after
    with store.subscribe_telemetry(channel) as changed:
        while True:
            changed.clear()
            rows = await asyncio.to_thread(read, after=cursor, limit=200, descending=False)
            for row in rows:
                yield f'id: {row.sequence}\nevent: {channel}\ndata: {row.model_dump_json()}\n\n'
                cursor = row.sequence
            if rows:
                continue
            try:
                await asyncio.wait_for(changed.wait(), timeout=heartbeat_seconds)
            except asyncio.TimeoutError:
                # Recheck on heartbeat too, allowing writes from another process/store.
                yield ': keep-alive\n\n'
