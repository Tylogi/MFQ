"""FastAPI application for the public ``mfq serve`` API."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from mfq.server.api.auth import ApiKeyManager, required_scope
from mfq.server.api.routes import ERROR_RESPONSES as ERROR_RESPONSES
from mfq.server.api.routes.auth import router as auth_router
from mfq.server.api.routes.jobs import event_router as job_event_router
from mfq.server.api.routes.jobs import router as job_router
from mfq.server.api.routes.mcp import router as mcp_router
from mfq.server.api.routes.media import router as media_router
from mfq.server.api.routes.models import hub_router
from mfq.server.api.routes.models import router as model_router
from mfq.server.api.routes.openai import router as openai_router
from mfq.server.api.routes.runtime import profile_router
from mfq.server.api.routes.runtime import router as runtime_router
from mfq.server.api.routes.sessions import initial_router as initial_session_router
from mfq.server.api.routes.sessions import router as session_router
from mfq.server.protocol.models import ErrorDetail, ErrorResponse
from mfq.server.services.service import ServerService, ServiceError


def create_app(
    service: ServerService | None = None,
    *,
    web_root: str | Path | None = None,
    api_key: str = "",
    api_keys: ApiKeyManager | None = None,
) -> FastAPI:
    """Create an executable app, or a route-only contract app when service is omitted."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            if service is not None:
                await service.start()
            yield
        finally:
            if service is not None:
                await service.aclose()

    app = FastAPI(
        title="MFQ Server API",
        summary="Persistent sessions, media, model workers, and realtime inference",
        version="1.0.0",
        openapi_version="3.1.0",
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.api_key = api_key
    app.state.api_keys = api_keys
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "tauri://localhost",
            "http://tauri.localhost",
            "https://tauri.localhost",
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Accept", "Authorization", "Content-Type", "X-Content-SHA256"],
    )

    @app.middleware("http")
    async def protect_and_harden(request: Request, call_next: Any) -> Response:
        if (api_key or api_keys is not None) and request.url.path.startswith(("/api/", "/v1")):
            authorization = request.headers.get("authorization", "")
            supplied = authorization[7:] if authorization.startswith("Bearer ") else ""
            authenticated = api_keys.authenticate(supplied) if api_keys is not None else None
            legacy = api_key and secrets.compare_digest(supplied, api_key)
            scope = required_scope(request.method, request.url.path)
            if not legacy and authenticated is None:
                detail = ErrorDetail(code="unauthorized", message="invalid API credential")
                return JSONResponse(
                    status_code=401,
                    content=ErrorResponse(error=detail).model_dump(mode="json"),
                    headers={"WWW-Authenticate": "Bearer"},
                )
            if (
                not legacy
                and authenticated is not None
                and api_keys is not None
                and not api_keys.permits(authenticated, scope)
            ):
                detail = ErrorDetail(
                    code="insufficient_scope",
                    message=f"API credential requires the {scope} scope",
                )
                return JSONResponse(
                    status_code=403,
                    content=ErrorResponse(error=detail).model_dump(mode="json"),
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; connect-src 'self' ws: wss:; img-src 'self' data: blob:; "
            "media-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; font-src 'self' data:"
        )
        return response

    @app.exception_handler(ServiceError)
    async def handle_service_error(_: Any, error: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content=ErrorResponse(error=error.detail).model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Any, error: RequestValidationError) -> JSONResponse:
        detail = ErrorDetail(
            code="invalid_request",
            message="request validation failed",
            details={"errors": jsonable_encoder(error.errors())},
        )
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(error=detail).model_dump(mode="json"),
        )

    app.include_router(openai_router)
    app.include_router(auth_router)
    app.include_router(initial_session_router)
    app.include_router(profile_router)
    app.include_router(job_router)
    app.include_router(hub_router)
    app.include_router(job_event_router)
    app.include_router(session_router)
    app.include_router(media_router)
    app.include_router(mcp_router)
    app.include_router(model_router)
    app.include_router(runtime_router)

    if web_root is not None:
        root = Path(web_root)
        if not root.is_dir():
            raise ValueError(f"web root is not a directory: {root}")
        app.mount("/", StaticFiles(directory=root, html=True), name="web")

    return app


def create_contract_app() -> FastAPI:
    """Create the route-only application used to generate the public OpenAPI contract."""

    return create_app()
