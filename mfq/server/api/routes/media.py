"""Media upload, download, and document routes."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Header
from fastapi.responses import FileResponse

from mfq.server.api.dependencies import ServiceDependency
from mfq.server.api.routes import ERROR_RESPONSES
from mfq.server.protocol.models import (
    SHA256_PATTERN,
    CreateDocumentRequest,
    DocumentResource,
    MediaResource,
)

router = APIRouter()


@router.post(
    "/api/v1/media",
    response_model=MediaResource,
    status_code=201,
    responses=ERROR_RESPONSES,
    tags=["media"],
)
async def upload_media(
    service: ServiceDependency,
    body: Annotated[bytes, Body(media_type="application/octet-stream")],
    content_type: Annotated[str, Header(alias="Content-Type")],
    content_sha256: Annotated[
        str,
        Header(alias="X-Content-SHA256", pattern=SHA256_PATTERN),
    ],
) -> MediaResource:
    return await service.upload_media(body, content_type, content_sha256)


@router.get(
    "/api/v1/media/{media_id}",
    responses=ERROR_RESPONSES,
    tags=["media"],
)
async def get_media(service: ServiceDependency, media_id: UUID) -> FileResponse:
    resource, path = await service.get_media(media_id)
    return FileResponse(
        path,
        media_type=resource.media.mime_type,
        headers={
            "Cache-Control": "private, immutable, max-age=31536000",
            "ETag": f'"{resource.media.sha256}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post(
    "/api/v1/documents",
    response_model=DocumentResource,
    status_code=201,
    responses=ERROR_RESPONSES,
    tags=["media"],
)
async def create_document(
    service: ServiceDependency, body: CreateDocumentRequest
) -> DocumentResource:
    return await service.create_document(body)


@router.get(
    "/api/v1/documents/{media_id}",
    response_model=DocumentResource,
    responses=ERROR_RESPONSES,
    tags=["media"],
)
async def get_document(service: ServiceDependency, media_id: UUID) -> DocumentResource:
    return await service.get_document(media_id)
