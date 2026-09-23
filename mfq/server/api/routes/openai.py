"""OpenAI-compatible health, model, and chat-completion routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from mfq.server.api.dependencies import ServiceDependency
from mfq.server.api.openai_compat import (
    OpenAIRequestError,
    backend_error_status,
    collect_chat_completion,
    parse_chat_request,
    stream_chat_completion,
)
from mfq.server.api.openai_compat import (
    error_body as openai_error_body,
)
from mfq.server.runtime.backend import BackendError, preflight_backend_request

router = APIRouter()


@router.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "mfq-server",
        "protocol_version": "1.0",
    }


@router.get("/v1", include_in_schema=False)
async def openai_root() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "mfq-server",
        "endpoints": ["/v1/models", "/v1/chat/completions"],
    }


@router.get("/v1/models", include_in_schema=False)
async def openai_models(
    service: ServiceDependency,
) -> dict[str, Any]:
    models = await service.advertised_models()
    data = models.get("data") if isinstance(models, dict) else None
    return {
        "object": "list",
        "data": [
            {
                "id": item["id"],
                "object": "model",
                "created": 0,
                "owned_by": "mfq",
            }
            for item in (data if isinstance(data, list) else [])
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
        ],
    }


@router.post("/v1/chat/completions", include_in_schema=False)
async def openai_chat_completions(service: ServiceDependency, request: Request) -> Response:
    try:
        body = await request.json()
    except ValueError as error:
        return JSONResponse(
            status_code=400,
            content=openai_error_body(str(error)),
        )
    try:
        parsed = parse_chat_request(body)
    except OpenAIRequestError as error:
        return JSONResponse(
            status_code=400,
            content=openai_error_body(str(error), param=error.param),
        )
    backend = service.backend
    if parsed.stream:
        try:
            await preflight_backend_request(
                backend,
                model=parsed.model,
                messages=parsed.messages,
                sampling=parsed.sampling,
                session_id=parsed.session_id,
                tools=parsed.tools,
                tool_choice=parsed.tool_choice,
                response_format=parsed.response_format,
            )
        except BackendError as error:
            return JSONResponse(
                status_code=backend_error_status(error),
                content=openai_error_body(
                    str(error),
                    error_type=error.code,
                    code=error.status_code,
                ),
            )
        return StreamingResponse(
            stream_chat_completion(
                backend,
                parsed,
                disconnected=request.is_disconnected,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    try:
        result = await collect_chat_completion(backend, parsed)
    except BackendError as error:
        return JSONResponse(
            status_code=backend_error_status(error),
            content=openai_error_body(
                str(error),
                error_type=error.code,
                code=error.status_code,
            ),
        )
    return JSONResponse(content=result)
