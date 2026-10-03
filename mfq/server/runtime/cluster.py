"""Health-aware routing across local and remote MFQ Server nodes."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import httpx

from mfq.server.runtime.backend import (
    BackendDelta,
    BackendError,
    BackendToolCallDelta,
    ChatBackend,
    closing_backend_stream,
    preflight_backend_request,
)
from mfq.server.runtime.client import iter_sse_data
from mfq.server.protocol.models import (
    RemoteNodeResource,
    ResponseFormat,
    ResponsePerformance,
    RuntimeCapabilitiesResource,
    SamplingParams,
    TokenUsage,
    ToolChoice,
    ToolDefinition,
)
from mfq.server.state.storage import SessionStore

_NodeVersion = tuple[UUID, str, str | None, bool, datetime]
_SESSION_CLEANUP_CONCURRENCY = 8


@dataclass
class _NodeState:
    resource: RemoteNodeResource
    models: list[str] = field(default_factory=list)
    healthy: bool = False
    active_requests: int = 0
    checked_at: float = 0.0
    checked_at_wall: datetime | None = None
    error: str | None = None
    status: dict[str, Any] = field(default_factory=dict)


@dataclass
class _RemoteSession:
    node: RemoteNodeResource
    remote_id: UUID
    revision: int
    synchronized_messages: int
    busy: bool = False


class ClusterBackend:
    """Route matching models to healthy remote MFQ Server nodes and aggregate inventory."""

    def __init__(
        self,
        local: ChatBackend,
        store: SessionStore,
        *,
        health_ttl_seconds: float = 5.0,
        probe_timeout_seconds: float = 5.0,
        control_timeout_seconds: float = 30.0,
        media_upload_timeout_seconds: float = 300.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.local = local
        self.store = store
        self.health_ttl_seconds = max(0.25, health_ttl_seconds)
        self.probe_timeout_seconds = max(0.25, probe_timeout_seconds)
        self.control_timeout_seconds = max(0.25, control_timeout_seconds)
        self.media_upload_timeout_seconds = max(
            self.control_timeout_seconds,
            media_upload_timeout_seconds,
        )
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=None, write=30.0, pool=3.0),
            trust_env=False,
        )
        self._states: dict[UUID, _NodeState] = {}
        self._sessions: dict[tuple[UUID, UUID], _RemoteSession] = {}
        self._lock = asyncio.Lock()
        self._refresh_lock = asyncio.Lock()
        self._closed = False

    @staticmethod
    def _headers(node: RemoteNodeResource) -> dict[str, str]:
        if not node.api_key_env:
            return {}
        token = os.environ.get(node.api_key_env, "")
        if not token:
            raise BackendError(
                "remote_node_credential_missing",
                f"environment variable is not set: {node.api_key_env}",
            )
        return {"Authorization": f"Bearer {token}"}

    async def refresh(self, *, force: bool = False) -> list[RemoteNodeResource]:
        async with self._refresh_lock:
            async with self._lock:
                if self._closed:
                    raise BackendError(
                        "backend_closed",
                        "cluster backend is closed",
                        status_code=503,
                    )
            resources = await asyncio.to_thread(self.store.list_remote_nodes)
            configured = {item.id for item in resources}
            retired: list[RemoteNodeResource] = []
            async with self._lock:
                for stale in set(self._states) - configured:
                    state = self._states.pop(stale, None)
                    if state is not None and state.active_requests == 0:
                        retired.append(state.resource)
                for resource in resources:
                    state = self._states.get(resource.id)
                    if state is None:
                        self._states[resource.id] = _NodeState(resource=resource)
                    elif state.resource != resource:
                        self._states[resource.id] = _NodeState(resource=resource)
                        if state.active_requests == 0:
                            retired.append(state.resource)
                    elif not resource.enabled:
                        state.healthy = False
                        state.models = []
                        state.status = {}
                        state.error = None
                states = list(self._states.values())
            if retired:
                await asyncio.gather(
                    *(self._release_node_sessions(node) for node in retired)
                )
            await asyncio.gather(
                *(
                    self._probe(state, force=force)
                    for state in states
                    if state.resource.enabled
                )
            )
            async with self._lock:
                return [self._public(state) for state in self._states.values()]

    async def _probe(self, state: _NodeState, *, force: bool) -> None:
        now = time.monotonic()
        if not force and now - state.checked_at < self.health_ttl_seconds:
            return
        state.checked_at = now
        state.checked_at_wall = datetime.now(timezone.utc)
        try:
            headers = self._headers(state.resource)
            health, models, status = await asyncio.gather(
                self._client.get(
                    f"{state.resource.url}/health",
                    headers=headers,
                    timeout=self.probe_timeout_seconds,
                ),
                self._remote_models(state.resource, headers),
                self._remote_status(state.resource, headers),
            )
            health.raise_for_status()
            models.raise_for_status()
            payload = models.json()
            data = payload.get("data", []) if isinstance(payload, dict) else []
            state.models = sorted(
                {str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id")}
            )
            state.healthy = True
            state.status = status
            state.error = None
        except Exception as error:
            state.healthy = False
            state.models = []
            state.status = {}
            state.error = str(error)[:512]

    async def _remote_models(
        self,
        node: RemoteNodeResource,
        headers: dict[str, str],
    ) -> httpx.Response:
        response = await self._client.get(
            f"{node.url}/v1/models",
            headers=headers,
            timeout=self.probe_timeout_seconds,
        )
        if response.status_code in {404, 405}:
            return await self._client.get(
                f"{node.url}/api/v1/runtime/models",
                headers=headers,
                timeout=self.probe_timeout_seconds,
            )
        return response

    async def _remote_status(
        self,
        node: RemoteNodeResource,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        try:
            response = await self._client.get(
                f"{node.url}/api/v1/runtime/status",
                headers=headers,
                timeout=self.probe_timeout_seconds,
            )
            if response.status_code >= 400:
                return {}
            payload = response.json()
            return payload if isinstance(payload, dict) else {}
        except (httpx.HTTPError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _public(state: _NodeState) -> RemoteNodeResource:
        return state.resource.model_copy(
            update={
                "healthy": state.healthy,
                "models": state.models,
                "active_requests": state.active_requests,
                "metrics": state.status,
                "last_checked_at": state.checked_at_wall,
                "error": state.error,
            }
        )

    def _node_is_current_locked(self, node: RemoteNodeResource) -> bool:
        state = self._states.get(node.id)
        return not self._closed and state is not None and state.resource == node

    @staticmethod
    def _node_version(node: RemoteNodeResource) -> _NodeVersion:
        return (
            node.id,
            node.url,
            node.api_key_env,
            node.enabled,
            node.updated_at,
        )

    @staticmethod
    def _remote_session_from_payload(
        node: RemoteNodeResource,
        payload: dict[str, Any],
        *,
        synchronized_messages: int,
        busy: bool = False,
    ) -> _RemoteSession:
        try:
            remote_id = UUID(payload["id"])
            revision = payload["revision"]
            if not isinstance(revision, int) or isinstance(revision, bool):
                raise TypeError("revision must be an integer")
        except (KeyError, TypeError, ValueError) as error:
            raise BackendError(
                "remote_node_protocol_error",
                f"remote session response is invalid: {error}",
                status_code=502,
            ) from error
        return _RemoteSession(
            node=node,
            remote_id=remote_id,
            revision=revision,
            synchronized_messages=synchronized_messages,
            busy=busy,
        )

    async def _select(
        self,
        model: str,
        *,
        exclude: set[_NodeVersion] | None = None,
    ) -> _NodeState | None:
        await self.refresh()
        excluded = exclude or set()
        async with self._lock:
            matches = [
                state
                for state in self._states.values()
                if self._node_version(state.resource) not in excluded
                and state.resource.enabled
                and state.healthy
                and model in state.models
            ]
        return (
            min(matches, key=lambda item: (item.active_requests, item.resource.name))
            if matches
            else None
        )

    async def preflight(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None = None,
        tools: Sequence[ToolDefinition] = (),
        tool_choice: ToolChoice = "auto",
        response_format: ResponseFormat | None = None,
    ) -> None:
        if await self._select(model) is not None:
            return
        await preflight_backend_request(
            self.local,
            model=model,
            messages=messages,
            sampling=sampling,
            session_id=session_id,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )

    async def stream(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None = None,
        tools: Sequence[ToolDefinition] = (),
        tool_choice: ToolChoice = "auto",
        response_format: ResponseFormat | None = None,
    ) -> AsyncIterator[BackendDelta]:
        attempted: set[_NodeVersion] = set()
        last_remote_failure: BackendError | None = None
        while True:
            node = await self._select(model, exclude=attempted)
            if node is None:
                break
            node_version = self._node_version(node.resource)
            claimed = False
            async with self._lock:
                if (
                    self._states.get(node.resource.id) is not node
                    or not node.resource.enabled
                    or not node.healthy
                    or model not in node.models
                ):
                    attempted.add(node_version)
                else:
                    node.active_requests += 1
                    claimed = True
            if not claimed:
                continue

            emitted = False
            retryable_failure: BackendError | None = None
            try:
                async with closing_backend_stream(
                    self._remote_stream(
                        node.resource,
                        model=model,
                        messages=messages,
                        sampling=sampling,
                        session_id=session_id,
                        tools=tools,
                        tool_choice=tool_choice,
                        response_format=response_format,
                    )
                ) as remote_stream:
                    async for delta in remote_stream:
                        emitted = True
                        yield delta
                return
            except BackendError as error:
                if emitted or not error.retryable:
                    raise
                retryable_failure = error
            finally:
                async with self._lock:
                    node.active_requests = max(0, node.active_requests - 1)
                    retired = (
                        node.active_requests == 0
                        and self._states.get(node.resource.id) is not node
                    )
                if retired:
                    await self._release_node_sessions(node.resource)

            assert retryable_failure is not None
            last_remote_failure = retryable_failure
            attempted.add(node_version)
            await self._record_node_failure(node, retryable_failure)
            if session_id is not None:
                await self._discard_remote_route(node.resource, session_id)

        local_emitted = False
        try:
            async with closing_backend_stream(
                self.local.stream(
                    model=model,
                    messages=messages,
                    sampling=sampling,
                    session_id=session_id,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                )
            ) as local_stream:
                async for delta in local_stream:
                    local_emitted = True
                    yield delta
        except BackendError as error:
            if (
                not local_emitted
                and last_remote_failure is not None
                and error.code in {"model_not_loaded", "model_artifact_not_found"}
            ):
                raise last_remote_failure from error
            raise

    async def _record_node_failure(
        self,
        node: _NodeState,
        error: BackendError,
    ) -> None:
        async with self._lock:
            if self._states.get(node.resource.id) is not node:
                return
            node.healthy = False
            node.models = []
            node.status = {}
            node.checked_at = time.monotonic()
            node.checked_at_wall = datetime.now(timezone.utc)
            node.error = f"{error.code}: {error}"[:512]

    async def _discard_remote_route(
        self,
        node: RemoteNodeResource,
        session_id: UUID,
    ) -> None:
        async with self._lock:
            key = (node.id, session_id)
            remote = self._sessions.get(key)
            if remote is not None and remote.node == node:
                self._sessions.pop(key, None)
            else:
                remote = None
        if remote is None:
            return
        await self._discard_routed_session(remote)

    async def _remote_stream(
        self,
        node: RemoteNodeResource,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None,
        tools: Sequence[ToolDefinition],
        tool_choice: ToolChoice,
        response_format: ResponseFormat | None,
    ) -> AsyncIterator[BackendDelta]:
        headers = self._headers(node)
        ephemeral = session_id is None
        key = (node.id, session_id or uuid4())
        remote = await self._acquire_remote_session(
            node,
            key=key,
            model=model,
            message_count=len(messages),
            headers=headers,
        )
        completed = False
        try:
            async for delta in self._remote_session_stream(
                node,
                messages=messages,
                sampling=sampling,
                remote=remote,
                headers=headers,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
            ):
                yield delta
            completed = True
        finally:
            reusable = remote.synchronized_messages == len(messages) + 1
            discard = False
            async with self._lock:
                if self._sessions.get(key) is remote:
                    if ephemeral or not completed or not reusable:
                        self._sessions.pop(key, None)
                        discard = True
                    else:
                        remote.busy = False
            if discard:
                await self._discard_remote_session(node, remote, headers)

    async def _acquire_remote_session(
        self,
        node: RemoteNodeResource,
        *,
        key: tuple[UUID, UUID],
        model: str,
        message_count: int,
        headers: dict[str, str],
    ) -> _RemoteSession:
        stale: _RemoteSession | None = None
        async with self._lock:
            if not self._node_is_current_locked(node):
                raise BackendError(
                    "remote_node_retired",
                    f"remote node changed before session acquisition: {node.name}",
                    retryable=True,
                    status_code=503,
                )
            remote = self._sessions.get(key)
            if remote is not None and remote.busy:
                raise BackendError(
                    "remote_session_busy",
                    "the remote session already has an active response",
                    status_code=409,
                )
            if remote is not None and (
                remote.node != node
                or remote.synchronized_messages > message_count
            ):
                self._sessions.pop(key, None)
                stale = remote
                remote = None
            if remote is not None:
                remote.busy = True
                return remote

        if stale is not None:
            await self._discard_routed_session(stale)

        created = await self._request_json(
            node,
            "POST",
            "/api/v1/sessions",
            {"model": model, "mode": "text"},
            headers,
        )
        candidate = self._remote_session_from_payload(
            node,
            created,
            synchronized_messages=0,
            busy=True,
        )
        discard_candidate = True
        displaced: _RemoteSession | None = None
        acquisition_error: BackendError | None = None
        try:
            async with self._lock:
                if not self._node_is_current_locked(node):
                    acquisition_error = BackendError(
                        "remote_node_retired",
                        f"remote node changed during session acquisition: {node.name}",
                        retryable=True,
                        status_code=503,
                    )
                    remote = None
                else:
                    incumbent = self._sessions.get(key)
                    if incumbent is None:
                        self._sessions[key] = candidate
                        remote = candidate
                        discard_candidate = False
                    elif incumbent.busy:
                        acquisition_error = BackendError(
                            "remote_session_busy",
                            "the remote session already has an active response",
                            status_code=409,
                        )
                        remote = None
                    elif incumbent.node != node or (
                        incumbent.synchronized_messages > message_count
                    ):
                        self._sessions[key] = candidate
                        displaced = incumbent
                        remote = candidate
                        discard_candidate = False
                    else:
                        incumbent.busy = True
                        remote = incumbent
        except BaseException:
            await self._discard_remote_session(node, candidate, headers)
            raise

        if discard_candidate:
            await self._discard_remote_session(node, candidate, headers)
        if displaced is not None:
            await self._discard_routed_session(displaced)
        if acquisition_error is not None:
            raise acquisition_error
        assert remote is not None
        return remote

    async def _remote_session_stream(
        self,
        node: RemoteNodeResource,
        *,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        remote: _RemoteSession,
        headers: dict[str, str],
        tools: Sequence[ToolDefinition],
        tool_choice: ToolChoice,
        response_format: ResponseFormat | None,
    ) -> AsyncIterator[BackendDelta]:
        prior = messages[remote.synchronized_messages : -1]
        for message in prior:
            role = str(message.get("role", "user"))
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            parts = await self._message_parts(node, message, headers)
            appended = await self._request_json(
                node,
                "POST",
                f"/api/v1/sessions/{remote.remote_id}/messages",
                {
                    "expected_revision": remote.revision,
                    "role": role,
                    "parts": parts,
                },
                headers,
            )
            remote.revision = int(appended["session"]["revision"])
        last = messages[-1] if messages else {"role": "user", "content": ""}
        input_parts = await self._message_parts(node, last, headers)
        request = {
            "request_id": str(uuid4()),
            "expected_revision": remote.revision,
            "input": input_parts,
            "input_role": "tool" if last.get("role") == "tool" else "user",
            "sampling": sampling.model_dump(mode="json", exclude_unset=True),
            "include_reasoning_history": True,
            "tools": [item.model_dump(mode="json") for item in tools],
            "tool_choice": tool_choice
            if isinstance(tool_choice, str)
            else tool_choice.model_dump(mode="json"),
            "response_format": (
                response_format.model_dump(mode="json", by_alias=True)
                if response_format is not None
                else {"type": "text"}
            ),
            "stream": True,
        }
        try:
            async with self._client.stream(
                "POST",
                f"{node.url}/api/v1/sessions/{remote.remote_id}/responses",
                json=request,
                headers=headers,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise BackendError(
                        "remote_node_error",
                        body.decode("utf-8", errors="replace")[:1024],
                        retryable=response.status_code >= 500,
                        status_code=response.status_code,
                    )
                saw_completed = False
                saw_terminal_state = False
                async for data in iter_sse_data(response):
                    frame = json.loads(data)
                    if not isinstance(frame, dict):
                        raise ValueError("remote SSE frame must be an object")
                    payload = frame.get("payload", {})
                    if not isinstance(payload, dict):
                        raise ValueError("remote SSE payload must be an object")
                    kind = payload.get("type")
                    if kind == "response.text.delta":
                        yield BackendDelta(content_delta=str(payload.get("delta", "")))
                    elif kind == "response.reasoning.delta":
                        yield BackendDelta(reasoning_delta=str(payload.get("delta", "")))
                    elif kind == "response.tool_call.delta":
                        yield BackendDelta(
                            tool_calls=(
                                BackendToolCallDelta(
                                    index=int(payload.get("index", 0)),
                                    call_id=payload.get("call_id"),
                                    name=payload.get("name"),
                                    arguments_delta=str(payload.get("arguments_delta", "")),
                                ),
                            )
                        )
                    elif kind == "response.completed":
                        if saw_completed:
                            raise ValueError("remote response completed more than once")
                        saw_completed = True
                        usage = (
                            TokenUsage.model_validate(payload["usage"])
                            if payload.get("usage")
                            else None
                        )
                        performance = (
                            ResponsePerformance.model_validate(payload["performance"])
                            if payload.get("performance")
                            else None
                        )
                        yield BackendDelta(
                            finish_reason=str(payload.get("finish_reason", "stop")),
                            usage=usage,
                            performance=performance,
                        )
                    elif kind == "session.state":
                        revision = payload.get("revision")
                        if not isinstance(revision, int) or isinstance(revision, bool):
                            raise ValueError(
                                "remote session.state is missing an integer revision"
                            )
                        remote.revision = revision
                        if saw_completed:
                            saw_terminal_state = True
                    elif kind == "error":
                        detail = payload.get("error", {})
                        raise BackendError(
                            str(detail.get("code", "remote_node_error")),
                            str(detail.get("message", "remote node failed")),
                            retryable=bool(detail.get("retryable", False)),
                        )
                if not saw_completed:
                    raise BackendError(
                        "remote_node_protocol_error",
                        "remote response stream ended without response.completed",
                        retryable=True,
                        status_code=502,
                    )
                if saw_terminal_state:
                    remote.synchronized_messages = len(messages) + 1
        except BackendError:
            raise
        except httpx.TimeoutException as error:
            raise BackendError(
                "remote_node_timeout",
                str(error),
                retryable=True,
                status_code=504,
            ) from error
        except httpx.HTTPError as error:
            raise BackendError(
                "remote_node_unavailable",
                str(error),
                retryable=True,
                status_code=502,
            ) from error
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise BackendError(
                "remote_node_protocol_error",
                str(error),
                status_code=502,
            ) from error

    async def _request_json(
        self,
        node: RemoteNodeResource,
        method: str,
        path: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        response = await self._control_request(
            node,
            method,
            path,
            headers=headers,
            json_body=body,
        )
        if response.status_code >= 400:
            raise BackendError(
                "remote_node_error",
                response.text[:1024],
                retryable=response.status_code >= 500,
                status_code=response.status_code,
            )
        try:
            value = response.json()
        except ValueError as error:
            raise BackendError(
                "remote_node_protocol_error",
                f"remote {path} response is not valid JSON: {error}",
                status_code=502,
            ) from error
        if not isinstance(value, dict):
            raise BackendError(
                "remote_node_protocol_error",
                f"remote {path} response must be an object",
                status_code=502,
            )
        return value

    async def _control_request(
        self,
        node: RemoteNodeResource,
        method: str,
        path: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
        content: bytes | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        try:
            return await self._client.request(
                method,
                f"{node.url}{path}",
                json=json_body,
                content=content,
                headers=headers,
                timeout=(
                    self.control_timeout_seconds
                    if timeout is None
                    else timeout
                ),
            )
        except httpx.TimeoutException as error:
            raise BackendError(
                "remote_node_timeout",
                str(error),
                retryable=True,
                status_code=504,
            ) from error
        except httpx.HTTPError as error:
            raise BackendError(
                "remote_node_unavailable",
                str(error),
                retryable=True,
                status_code=502,
            ) from error

    async def _release_node_sessions(self, node: RemoteNodeResource) -> None:
        async with self._lock:
            sessions = [
                self._sessions.pop(key)
                for key in list(self._sessions)
                if key[0] == node.id and self._sessions[key].node == node
            ]
        if not sessions:
            return
        for offset in range(0, len(sessions), _SESSION_CLEANUP_CONCURRENCY):
            await asyncio.gather(
                *(
                    self._discard_routed_session(remote)
                    for remote in sessions[
                        offset : offset + _SESSION_CLEANUP_CONCURRENCY
                    ]
                )
            )

    async def _discard_routed_session(self, remote: _RemoteSession) -> None:
        try:
            headers = self._headers(remote.node)
        except BackendError:
            return
        await self._discard_remote_session(remote.node, remote, headers)

    async def _discard_remote_session(
        self,
        node: RemoteNodeResource,
        remote: _RemoteSession,
        headers: dict[str, str],
    ) -> None:
        with suppress(BackendError):
            await self._control_request(
                node,
                "DELETE",
                f"/api/v1/sessions/{remote.remote_id}",
                headers=headers,
            )

    async def _parts(
        self,
        node: RemoteNodeResource,
        content: Any,
        headers: dict[str, str],
    ) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                kind = item.get("type")
                if kind == "text":
                    parts.append({"type": "text", "text": str(item.get("text", ""))})
                elif kind == "image_url":
                    url = item.get("image_url", {}).get("url")
                    if isinstance(url, str):
                        media = await self._upload_data_url(node, url, headers)
                        parts.append({"type": "image", "media": media})
                elif kind == "video_url":
                    url = item.get("video_url", {}).get("url")
                    if isinstance(url, str):
                        media = await self._upload_data_url(node, url, headers)
                        parts.append({"type": "video", "media": media})
                elif kind == "input_audio":
                    audio = item.get("input_audio", {})
                    encoded = audio.get("data")
                    if isinstance(encoded, str):
                        media = await self._upload_bytes(
                            node,
                            self._decode_base64(encoded),
                            self._audio_mime(str(audio.get("format", "wav"))),
                            headers,
                        )
                        parts.append(
                            {
                                "type": "audio",
                                "media": media,
                                "sample_rate_hz": int(audio.get("sample_rate_hz") or 16000),
                                "channels": int(audio.get("channels") or 1),
                            }
                        )
            return parts or [{"type": "text", "text": ""}]
        return [{"type": "text", "text": str(content or "")}]

    async def _message_parts(
        self,
        node: RemoteNodeResource,
        message: dict[str, Any],
        headers: dict[str, str],
    ) -> list[dict[str, Any]]:
        role = str(message.get("role", "user"))
        if role == "tool":
            return [
                {
                    "type": "tool_result",
                    "call_id": str(message.get("tool_call_id") or "remote-tool"),
                    "result": message.get("content", ""),
                    "is_error": False,
                }
            ]
        parts = await self._parts(node, message.get("content"), headers)
        if role != "assistant":
            return parts
        extras: list[dict[str, Any]] = []
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            extras.append({"type": "reasoning", "text": reasoning})
        for call in message.get("tool_calls", []):
            if not isinstance(call, dict):
                continue
            function = call.get("function", {})
            if not isinstance(function, dict) or not function.get("name"):
                continue
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
            extras.append(
                {
                    "type": "tool_call",
                    "call_id": str(call.get("id") or uuid4()),
                    "name": str(function["name"]),
                    "arguments": arguments,
                }
            )
        if extras and parts == [{"type": "text", "text": ""}]:
            parts = []
        return [*parts, *extras]

    async def _upload_data_url(
        self,
        node: RemoteNodeResource,
        url: str,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        if not url.startswith("data:") or ";base64," not in url:
            raise BackendError(
                "remote_media_unsupported",
                "remote routing accepts embedded base64 media only",
            )
        descriptor, encoded = url.split(",", 1)
        mime_type = descriptor[5:].split(";", 1)[0].strip().lower()
        return await self._upload_bytes(
            node,
            self._decode_base64(encoded),
            mime_type,
            headers,
        )

    async def _upload_bytes(
        self,
        node: RemoteNodeResource,
        data: bytes,
        mime_type: str,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        digest = hashlib.sha256(data).hexdigest()
        response = await self._control_request(
            node,
            "POST",
            "/api/v1/media",
            content=data,
            headers={
                **headers,
                "Content-Type": mime_type,
                "X-Content-SHA256": digest,
            },
            timeout=self.media_upload_timeout_seconds,
        )
        if response.status_code >= 400:
            raise BackendError(
                "remote_media_upload_failed",
                response.text[:1024],
                retryable=response.status_code >= 500,
                status_code=response.status_code,
            )
        try:
            value = response.json()
        except ValueError as error:
            raise BackendError(
                "remote_node_protocol_error",
                f"remote media response is not valid JSON: {error}",
                status_code=502,
            ) from error
        if not isinstance(value, dict) or not isinstance(value.get("media"), dict):
            raise BackendError(
                "remote_node_protocol_error",
                "remote media response is invalid",
                status_code=502,
            )
        return value["media"]

    @staticmethod
    def _decode_base64(value: str) -> bytes:
        try:
            return base64.b64decode(value, validate=True)
        except ValueError as error:
            raise BackendError("remote_media_invalid", "media base64 is invalid") from error

    @staticmethod
    def _audio_mime(format_name: str) -> str:
        return {
            "mp3": "audio/mpeg",
            "m4a": "audio/x-m4a",
            "wav": "audio/wav",
        }.get(format_name.lower(), f"audio/{format_name.lower()}")

    async def nodes(self, *, force: bool = False) -> list[RemoteNodeResource]:
        return await self.refresh(force=force)

    async def fork_session(self, source_session_id: UUID, target_session_id: UUID) -> bool:
        source: _RemoteSession | None = None
        async with self._lock:
            if not self._closed:
                source = next(
                    (
                        remote
                        for (node_id, local_id), remote in self._sessions.items()
                        if local_id == source_session_id
                        and node_id == remote.node.id
                        and self._node_is_current_locked(remote.node)
                    ),
                    None,
                )
        if source is not None:
            headers = self._headers(source.node)
            created = await self._request_json(
                source.node,
                "POST",
                f"/api/v1/sessions/{source.remote_id}/fork",
                {"include_message": True},
                headers,
            )
            candidate = self._remote_session_from_payload(
                source.node,
                created,
                synchronized_messages=source.synchronized_messages,
            )
            committed = False
            already_present = False
            try:
                async with self._lock:
                    current_source = self._sessions.get(
                        (source.node.id, source_session_id)
                    )
                    target_key = (source.node.id, target_session_id)
                    current_target = self._sessions.get(target_key)
                    if (
                        current_source is source
                        and self._node_is_current_locked(source.node)
                    ):
                        if current_target is None:
                            self._sessions[target_key] = candidate
                            committed = True
                        elif current_target.node == source.node:
                            already_present = True
            except BaseException:
                await self._discard_remote_session(source.node, candidate, headers)
                raise
            if not committed:
                await self._discard_remote_session(source.node, candidate, headers)
            if committed or already_present:
                return True
        return await self.local.fork_session(source_session_id, target_session_id)

    async def close_session(self, session_id: UUID) -> bool:
        async with self._lock:
            removed = [
                self._sessions.pop(key)
                for key in list(self._sessions)
                if key[1] == session_id
            ]
        for remote in removed:
            response = await self._control_request(
                remote.node,
                "DELETE",
                f"/api/v1/sessions/{remote.remote_id}",
                headers=self._headers(remote.node),
            )
            if response.status_code not in {204, 404}:
                raise BackendError(
                    "remote_node_error",
                    response.text[:1024],
                    retryable=response.status_code >= 500,
                    status_code=response.status_code,
                )
        return bool(removed) or await self.local.close_session(session_id)

    async def cancel_response(self, session_id: UUID) -> bool:
        async with self._lock:
            candidates = [
                remote
                for (_, local_session_id), remote in self._sessions.items()
                if local_session_id == session_id
            ]
        for remote in candidates:
            response = await self._control_request(
                remote.node,
                "POST",
                f"/api/v1/sessions/{remote.remote_id}/responses/cancel",
                headers=self._headers(remote.node),
            )
            if response.status_code == 200:
                return True
            raise BackendError(
                "remote_node_cancel_failed",
                response.text[:1024],
                retryable=response.status_code >= 500,
                status_code=response.status_code,
            )
        cancel = getattr(self.local, "cancel_response", None)
        return bool(await cancel(session_id)) if callable(cancel) else False

    async def capabilities(self) -> RuntimeCapabilitiesResource:
        return await self.local.capabilities()

    async def runtime_status(self) -> dict[str, Any]:
        status = dict(await self.local.runtime_status())
        nodes = await self.refresh()
        status["cluster_nodes"] = len(nodes)
        status["cluster_healthy_nodes"] = sum(item.healthy for item in nodes)
        status["cluster_active_requests"] = sum(item.active_requests for item in nodes)
        async with self._lock:
            states = list(self._states.values())
        status["cluster_process_resident_bytes"] = sum(
            int(item.status.get("process_resident_bytes") or 0) for item in states
        )
        status["cluster_device_active_bytes"] = sum(
            int(item.status.get("mlx_active_bytes") or item.status.get("cuda_allocated_bytes") or 0)
            for item in states
        )
        status["cluster_total_requests"] = sum(
            int(item.status.get("total_requests") or 0) for item in states
        )
        return status

    async def runtime_models(self) -> dict[str, Any]:
        local = await self.local.runtime_models()
        data = list(local.get("data", [])) if isinstance(local, dict) else []
        for node in await self.refresh():
            data.extend(
                {"id": model, "object": "model", "node_id": str(node.id), "node": node.name}
                for model in node.models
            )
        return {"object": "list", "data": data}

    async def realtime_capabilities(self) -> dict[str, Any]:
        return await self.local.realtime_capabilities()

    async def voice_output_status(self) -> dict[str, Any]:
        return await self.local.voice_output_status()

    async def enable_realtime(self) -> dict[str, Any]:
        return await self.local.enable_realtime()

    async def realtime_serve(self, client: Any, *, mode: str = "audio") -> bool:
        return await self.local.realtime_serve(client, mode=mode)

    async def reload_runtime(
        self,
        context_size: int,
        instance_id: UUID | None = None,
    ) -> dict[str, Any]:
        if instance_id is None:
            return await self.local.reload_runtime(context_size)
        return await self.local.reload_runtime(context_size, instance_id)

    async def clear_runtime_cache(
        self,
        instance_id: UUID | None = None,
    ) -> dict[str, Any]:
        if instance_id is None:
            return await self.local.clear_runtime_cache()
        return await self.local.clear_runtime_cache(instance_id)

    async def trim_runtime_cache(
        self,
        target_bytes: int = 0,
        instance_id: UUID | None = None,
    ) -> dict[str, Any]:
        if instance_id is None:
            return await self.local.trim_runtime_cache(target_bytes)
        return await self.local.trim_runtime_cache(target_bytes, instance_id)

    def realtime_connect(self, *, mode: str = "audio") -> Any:
        return self.local.realtime_connect(mode=mode)

    async def aclose(self) -> None:
        async with self._refresh_lock:
            async with self._lock:
                if self._closed:
                    return
                self._closed = True
                nodes = [state.resource for state in self._states.values()]
            await asyncio.gather(
                *(self._release_node_sessions(node) for node in nodes)
            )
            async with self._lock:
                self._sessions.clear()
            await self.local.aclose()
            if self._owns_client:
                await self._client.aclose()
