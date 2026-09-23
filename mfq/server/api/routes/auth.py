"""Scoped API-key management routes."""

from __future__ import annotations

import asyncio
from uuid import UUID

from fastapi import APIRouter

from mfq.server.api.dependencies import ApiKeysDependency
from mfq.server.api.routes import ERROR_RESPONSES
from mfq.server.protocol.models import (
    ApiKeyList,
    ApiKeyResource,
    ApiKeySecretResource,
    CreateApiKeyRequest,
)
from mfq.server.services.service import ServiceError
from mfq.server.state.storage import ApiKeyNotFoundError, StorageError

router = APIRouter()


@router.post(
    "/api/v1/auth/keys",
    response_model=ApiKeySecretResource,
    status_code=201,
    responses=ERROR_RESPONSES,
    tags=["auth"],
)
async def create_api_key(
    api_keys: ApiKeysDependency, body: CreateApiKeyRequest
) -> ApiKeySecretResource:
    try:
        return await asyncio.to_thread(api_keys.create, body)
    except StorageError as error:
        raise ServiceError(409, "api_key_conflict", str(error)) from error


@router.get(
    "/api/v1/auth/keys",
    response_model=ApiKeyList,
    responses=ERROR_RESPONSES,
    tags=["auth"],
)
async def list_api_keys(
    api_keys: ApiKeysDependency,
) -> ApiKeyList:
    data = await asyncio.to_thread(api_keys.store.list_api_keys)
    return ApiKeyList(data=data)


@router.post(
    "/api/v1/auth/keys/{key_id}/revoke",
    response_model=ApiKeyResource,
    responses=ERROR_RESPONSES,
    tags=["auth"],
)
async def revoke_api_key(api_keys: ApiKeysDependency, key_id: UUID) -> ApiKeyResource:
    try:
        return await asyncio.to_thread(api_keys.store.revoke_api_key, key_id)
    except ApiKeyNotFoundError as error:
        raise ServiceError(404, "api_key_not_found", str(error)) from error


@router.post(
    "/api/v1/auth/keys/{key_id}/rotate",
    response_model=ApiKeySecretResource,
    responses=ERROR_RESPONSES,
    tags=["auth"],
)
async def rotate_api_key(api_keys: ApiKeysDependency, key_id: UUID) -> ApiKeySecretResource:
    try:
        return await asyncio.to_thread(api_keys.rotate, key_id)
    except ApiKeyNotFoundError as error:
        raise ServiceError(404, "api_key_not_found", str(error)) from error
