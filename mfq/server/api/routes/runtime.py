"""Runtime profile, instance, cache, component, and realtime routes."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Query, Response, WebSocket, WebSocketDisconnect

from mfq.server.api.dependencies import (
    ServiceDependency,
    WebSocketServiceDependency,
    authorize_websocket,
)
from mfq.server.api.routes import ERROR_RESPONSES
from mfq.server.protocol.models import (
    CreateRuntimeProfileRequest,
    OperationAccepted,
    RuntimeCacheClearRequest,
    RuntimeCacheTrimRequest,
    RuntimeCapabilitiesResource,
    RuntimeInstanceList,
    RuntimeInstanceResource,
    RuntimeLogLevel,
    RuntimeLogList,
    RuntimeMetricList,
    RuntimeProfileList,
    RuntimeProfileLoadRequest,
    RuntimeProfileResource,
    RuntimeReloadRequest,
    UpdateRuntimeInstanceRequest,
    UpdateRuntimeProfileRequest,
)

profile_router = APIRouter()
router = APIRouter()


@profile_router.post(
    "/api/v1/runtime/profiles",
    response_model=RuntimeProfileResource,
    status_code=201,
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def create_runtime_profile(
    service: ServiceDependency,
    body: CreateRuntimeProfileRequest,
) -> RuntimeProfileResource:
    return await service.create_runtime_profile(body)


@profile_router.get(
    "/api/v1/runtime/profiles",
    response_model=RuntimeProfileList,
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def list_runtime_profiles(
    service: ServiceDependency,
) -> RuntimeProfileList:
    return await service.list_runtime_profiles()


@profile_router.get(
    "/api/v1/runtime/profiles/{profile_id}",
    response_model=RuntimeProfileResource,
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def get_runtime_profile(
    service: ServiceDependency, profile_id: UUID
) -> RuntimeProfileResource:
    return await service.get_runtime_profile(profile_id)


@profile_router.put(
    "/api/v1/runtime/profiles/{profile_id}",
    response_model=RuntimeProfileResource,
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def update_runtime_profile(
    service: ServiceDependency, profile_id: UUID, body: UpdateRuntimeProfileRequest
) -> RuntimeProfileResource:
    return await service.update_runtime_profile(profile_id, body)


@profile_router.delete(
    "/api/v1/runtime/profiles/{profile_id}",
    status_code=204,
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def delete_runtime_profile(service: ServiceDependency, profile_id: UUID) -> Response:
    await service.delete_runtime_profile(profile_id)
    return Response(status_code=204)


@profile_router.post(
    "/api/v1/runtime/profiles/{profile_id}/load",
    response_model=OperationAccepted,
    status_code=202,
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def load_runtime_profile(
    service: ServiceDependency, profile_id: UUID, body: RuntimeProfileLoadRequest
) -> OperationAccepted:
    return await service.load_runtime_profile(profile_id, body)


@router.get(
    "/api/v1/runtime/instances",
    response_model=RuntimeInstanceList,
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def runtime_instances(
    service: ServiceDependency,
) -> RuntimeInstanceList:
    return await service.runtime_instances()


@router.patch(
    "/api/v1/runtime/instances/{instance_id}",
    response_model=RuntimeInstanceResource,
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def update_runtime_instance(
    service: ServiceDependency,
    instance_id: UUID,
    body: UpdateRuntimeInstanceRequest,
) -> RuntimeInstanceResource:
    return await service.update_runtime_instance(instance_id, body)


@router.get(
    "/api/v1/runtime/capabilities",
    response_model=RuntimeCapabilitiesResource,
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def runtime_capabilities(
    service: ServiceDependency,
    instance_id: UUID | None = None,
) -> RuntimeCapabilitiesResource:
    return await service.runtime_capabilities(instance_id)


@router.get(
    "/api/v1/runtime/status",
    response_model=dict[str, Any],
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def runtime_status(
    service: ServiceDependency, instance_id: UUID | None = None
) -> dict[str, Any]:
    return await service.runtime_status(instance_id)


@router.get(
    "/api/v1/runtime/metrics",
    response_model=RuntimeMetricList,
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def runtime_metrics(
    service: ServiceDependency,
    instance_id: UUID | None = None,
    since: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=2000)] = 200,
) -> RuntimeMetricList:
    return await service.runtime_metrics(instance_id=instance_id, since=since, limit=limit)


@router.get(
    "/api/v1/runtime/logs",
    response_model=RuntimeLogList,
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def runtime_logs(
    service: ServiceDependency,
    instance_id: UUID | None = None,
    level: RuntimeLogLevel | None = None,
    after: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=2000)] = 200,
) -> RuntimeLogList:
    return await service.runtime_logs(
        instance_id=instance_id, level=level, after=after, limit=limit
    )


@router.get(
    "/api/v1/runtime/models",
    response_model=dict[str, Any],
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def runtime_models(
    service: ServiceDependency,
) -> dict[str, Any]:
    return await service.runtime_models()


@router.get(
    "/api/v1/runtime/realtime/capabilities",
    response_model=dict[str, Any],
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def realtime_capabilities(
    service: ServiceDependency,
) -> dict[str, Any]:
    return await service.realtime_capabilities()


@router.post(
    "/api/v1/runtime/reload",
    response_model=dict[str, Any],
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def reload_runtime(service: ServiceDependency, body: RuntimeReloadRequest) -> dict[str, Any]:
    return await service.reload_runtime(
        body.context_size,
        instance_id=body.instance_id,
    )


@router.post(
    "/api/v1/runtime/cache/clear",
    response_model=dict[str, Any],
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def clear_runtime_cache(
    service: ServiceDependency,
    body: RuntimeCacheClearRequest | None = None,
) -> dict[str, Any]:
    return await service.clear_runtime_cache(
        instance_id=body.instance_id if body is not None else None,
    )


@router.post(
    "/api/v1/runtime/cache/trim",
    response_model=dict[str, Any],
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def trim_runtime_cache(
    service: ServiceDependency,
    body: RuntimeCacheTrimRequest | None = None,
) -> dict[str, Any]:
    return await service.trim_runtime_cache(
        body.target_bytes if body is not None else 0,
        instance_id=body.instance_id if body is not None else None,
    )


@router.get(
    "/api/v1/components/voice-output",
    response_model=dict[str, Any],
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def voice_output_component_status(
    service: ServiceDependency,
) -> dict[str, Any]:
    return await service.voice_output_component_status()


@router.post(
    "/api/v1/components/voice-output/install",
    response_model=OperationAccepted,
    status_code=202,
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def install_voice_output_component(
    service: ServiceDependency,
) -> OperationAccepted:
    return await service.install_voice_output_component()


@router.post(
    "/api/v1/components/voice-output/activate",
    response_model=dict[str, Any],
    responses=ERROR_RESPONSES,
    tags=["runtime"],
)
async def activate_voice_output_component(
    service: ServiceDependency,
) -> dict[str, Any]:
    return await service.activate_voice_output_component()


@router.websocket("/api/v1/runtime/realtime")
async def runtime_realtime(service: WebSocketServiceDependency, websocket: WebSocket) -> None:
    if not await authorize_websocket(websocket):
        return
    mode = websocket.query_params.get("mode", "audio")
    if mode != "audio":
        await websocket.close(code=1008, reason="audio mode is required")
        return
    try:
        await websocket.accept()
        if await service.realtime_serve(websocket, mode=mode):
            return
        connector = service.realtime_connect(mode=mode)
        async with connector as upstream:

            async def send_upstream() -> None:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    if message.get("text") is not None:
                        await upstream.send(message["text"])
                    elif message.get("bytes") is not None:
                        await upstream.send(message["bytes"])

            async def send_client() -> None:
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)

            tasks = {
                asyncio.create_task(
                    send_upstream(),
                    name="mfq-realtime-client-to-runtime",
                ),
                asyncio.create_task(
                    send_client(),
                    name="mfq-realtime-runtime-to-client",
                ),
            }
            try:
                done, _ = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    task.result()
            finally:
                # The route itself can be cancelled during application
                # shutdown before either relay finishes.  Always tear down
                # both directions so the upstream runtime socket cannot
                # outlive its client or the ASGI request task.
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            with suppress(Exception):
                await websocket.close(code=1000)
    except WebSocketDisconnect:
        return
    except Exception as error:
        with suppress(Exception):
            await websocket.send_json({"type": "error", "error": {"message": str(error)}})
        with suppress(Exception):
            await websocket.close(code=1011, reason="realtime proxy failed")


@router.websocket("/api/v1/realtime")
async def realtime(websocket: WebSocket) -> None:
    if not await authorize_websocket(websocket):
        return
    await websocket.close(code=1013, reason="Realtime audio transport is not available")
