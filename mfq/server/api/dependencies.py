"""FastAPI dependencies shared by public API route modules."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Request, WebSocket

from mfq.server.api.auth import ApiKeyManager
from mfq.server.services.service import ServerService, ServiceError


def require_service(request: Request) -> ServerService:
    service = request.app.state.service
    if service is None:
        raise ServiceError(
            501,
            "contract_only",
            "the protocol contract application does not execute requests",
        )
    return service


def require_websocket_service(websocket: WebSocket) -> ServerService:
    service = websocket.app.state.service
    if service is None:
        raise ServiceError(
            501,
            "contract_only",
            "the protocol contract application does not execute requests",
        )
    return service


def require_api_keys(request: Request) -> ApiKeyManager:
    api_keys = request.app.state.api_keys
    if api_keys is None:
        raise ServiceError(
            501,
            "api_key_management_unavailable",
            "scoped API key management requires a configured root credential",
        )
    return api_keys


async def authorize_websocket(websocket: WebSocket) -> bool:
    api_key: str = websocket.app.state.api_key
    api_keys: ApiKeyManager | None = websocket.app.state.api_keys
    if not api_key and api_keys is None:
        return True
    authorization = websocket.headers.get("authorization", "")
    supplied = authorization[7:] if authorization.startswith("Bearer ") else ""
    if not supplied:
        supplied = websocket.query_params.get("access_token", "")
    authenticated = api_keys.authenticate(supplied) if api_keys is not None else None
    if (api_key and secrets.compare_digest(supplied, api_key)) or (
        authenticated is not None and api_keys and api_keys.permits(authenticated, "inference")
    ):
        return True
    await websocket.close(code=1008, reason="invalid API credential")
    return False


ServiceDependency = Annotated[ServerService, Depends(require_service)]
WebSocketServiceDependency = Annotated[
    ServerService,
    Depends(require_websocket_service),
]
ApiKeysDependency = Annotated[ApiKeyManager, Depends(require_api_keys)]
