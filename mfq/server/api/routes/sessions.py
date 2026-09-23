"""Session, response, and generation-preset routes."""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Query, Response
from fastapi.responses import StreamingResponse

from mfq.server.api.dependencies import ServiceDependency
from mfq.server.api.routes import ERROR_RESPONSES
from mfq.server.protocol.models import (
    AppendMessageRequest,
    AppendMessageResult,
    CreateGenerationPresetRequest,
    CreateResponseRequest,
    CreateSessionRequest,
    ForkSessionRequest,
    GenerationPresetList,
    GenerationPresetResource,
    MessageList,
    ResponseList,
    ResponseResource,
    RewindSessionRequest,
    SessionArchive,
    SessionImportResult,
    SessionList,
    SessionResource,
    UpdateGenerationPresetRequest,
    UpdateSessionRequest,
)

initial_router = APIRouter()
router = APIRouter()


@initial_router.post(
    "/api/v1/sessions",
    response_model=SessionResource,
    status_code=201,
    responses=ERROR_RESPONSES,
    tags=["sessions"],
)
async def create_session(service: ServiceDependency, body: CreateSessionRequest) -> SessionResource:
    return await service.create_session(body)


@initial_router.post(
    "/api/v1/sessions/import",
    response_model=SessionImportResult,
    status_code=201,
    responses=ERROR_RESPONSES,
    tags=["sessions"],
)
async def import_session(service: ServiceDependency, body: SessionArchive) -> SessionImportResult:
    return await service.import_session(body)


@initial_router.post(
    "/api/v1/presets",
    response_model=GenerationPresetResource,
    status_code=201,
    responses=ERROR_RESPONSES,
    tags=["presets"],
)
async def create_generation_preset(
    service: ServiceDependency,
    body: CreateGenerationPresetRequest,
) -> GenerationPresetResource:
    return await service.create_generation_preset(body)


@initial_router.get(
    "/api/v1/presets",
    response_model=GenerationPresetList,
    responses=ERROR_RESPONSES,
    tags=["presets"],
)
async def list_generation_presets(
    service: ServiceDependency,
) -> GenerationPresetList:
    return await service.list_generation_presets()


@initial_router.put(
    "/api/v1/presets/{preset_id}",
    response_model=GenerationPresetResource,
    responses=ERROR_RESPONSES,
    tags=["presets"],
)
async def update_generation_preset(
    service: ServiceDependency, preset_id: UUID, body: UpdateGenerationPresetRequest
) -> GenerationPresetResource:
    return await service.update_generation_preset(preset_id, body)


@initial_router.delete(
    "/api/v1/presets/{preset_id}",
    status_code=204,
    responses=ERROR_RESPONSES,
    tags=["presets"],
)
async def delete_generation_preset(service: ServiceDependency, preset_id: UUID) -> Response:
    await service.delete_generation_preset(preset_id)
    return Response(status_code=204)


@router.get(
    "/api/v1/sessions",
    response_model=SessionList,
    responses=ERROR_RESPONSES,
    tags=["sessions"],
)
async def list_sessions(
    service: ServiceDependency,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SessionList:
    return await service.list_sessions(limit=limit, offset=offset)


@router.get(
    "/api/v1/sessions/{session_id}",
    response_model=SessionResource,
    responses=ERROR_RESPONSES,
    tags=["sessions"],
)
async def get_session(service: ServiceDependency, session_id: UUID) -> SessionResource:
    return await service.get_session(session_id)


@router.patch(
    "/api/v1/sessions/{session_id}",
    response_model=SessionResource,
    responses=ERROR_RESPONSES,
    tags=["sessions"],
)
async def update_session(
    service: ServiceDependency,
    session_id: UUID,
    body: UpdateSessionRequest,
) -> SessionResource:
    return await service.update_session(session_id, body)


@router.get(
    "/api/v1/sessions/{session_id}/messages",
    response_model=MessageList,
    responses=ERROR_RESPONSES,
    tags=["sessions"],
)
async def list_messages(service: ServiceDependency, session_id: UUID) -> MessageList:
    return await service.list_messages(session_id)


@router.get(
    "/api/v1/sessions/{session_id}/export",
    response_model=SessionArchive,
    responses=ERROR_RESPONSES,
    tags=["sessions"],
)
async def export_session(service: ServiceDependency, session_id: UUID) -> SessionArchive:
    return await service.export_session(session_id)


@router.post(
    "/api/v1/sessions/{session_id}/messages",
    response_model=AppendMessageResult,
    status_code=201,
    responses=ERROR_RESPONSES,
    tags=["sessions"],
)
async def append_message(
    service: ServiceDependency,
    session_id: UUID,
    body: AppendMessageRequest,
) -> AppendMessageResult:
    return await service.append_message(session_id, body)


@router.get(
    "/api/v1/sessions/{session_id}/responses",
    response_model=ResponseList,
    responses=ERROR_RESPONSES,
    tags=["sessions"],
)
async def list_responses(
    service: ServiceDependency,
    session_id: UUID,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> ResponseList:
    return await service.list_responses(session_id, limit=limit)


@router.post(
    "/api/v1/sessions/{session_id}/responses",
    response_model=ResponseResource,
    responses={
        **ERROR_RESPONSES,
        200: {
            "description": "Completed response or an SSE event stream",
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/ResponseResource"}},
                "text/event-stream": {"schema": {"type": "string"}},
            },
        },
    },
    tags=["sessions"],
)
async def create_response(
    service: ServiceDependency, session_id: UUID, body: CreateResponseRequest
) -> Any:
    daemon = service
    prepared = await daemon.prepare_response(session_id, body)
    if body.stream:
        return StreamingResponse(
            daemon.stream_response(prepared),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    return await daemon.collect_response(prepared)


@router.post(
    "/api/v1/sessions/{session_id}/responses/cancel",
    response_model=ResponseResource,
    responses=ERROR_RESPONSES,
    tags=["sessions"],
)
async def cancel_response(service: ServiceDependency, session_id: UUID) -> ResponseResource:
    return await service.cancel_response(session_id)


@router.post(
    "/api/v1/sessions/{session_id}/fork",
    response_model=SessionResource,
    status_code=201,
    responses=ERROR_RESPONSES,
    tags=["sessions"],
)
async def fork_session(
    service: ServiceDependency, session_id: UUID, body: ForkSessionRequest
) -> SessionResource:
    return await service.fork_session(session_id, body)


@router.post(
    "/api/v1/sessions/{session_id}/rewind",
    response_model=SessionResource,
    responses=ERROR_RESPONSES,
    tags=["sessions"],
)
async def rewind_session(
    service: ServiceDependency,
    session_id: UUID,
    body: RewindSessionRequest,
) -> SessionResource:
    return await service.rewind_session(session_id, body)


@router.delete(
    "/api/v1/sessions/{session_id}",
    status_code=204,
    responses=ERROR_RESPONSES,
    tags=["sessions"],
)
async def delete_session(service: ServiceDependency, session_id: UUID) -> Response:
    await service.delete_session(session_id)
    return Response(status_code=204)
