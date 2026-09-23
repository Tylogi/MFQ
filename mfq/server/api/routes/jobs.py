"""Background-job routes and event streams."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, Query, Response
from fastapi.responses import StreamingResponse

from mfq.server.api.dependencies import ServiceDependency
from mfq.server.api.routes import ERROR_RESPONSES
from mfq.server.protocol.models import (
    CreateJobRequest,
    JobEventList,
    JobKindList,
    JobList,
    JobResource,
    JobStatus,
)

router = APIRouter()
event_router = APIRouter()


@router.post(
    "/api/v1/jobs",
    response_model=JobResource,
    status_code=202,
    responses=ERROR_RESPONSES,
    tags=["jobs"],
)
async def create_job(service: ServiceDependency, body: CreateJobRequest) -> JobResource:
    return await service.create_job(body)


@router.get(
    "/api/v1/jobs/kinds",
    response_model=JobKindList,
    responses=ERROR_RESPONSES,
    tags=["jobs"],
)
async def job_kinds(
    service: ServiceDependency,
) -> JobKindList:
    return await service.job_kinds()


@router.get(
    "/api/v1/jobs",
    response_model=JobList,
    responses=ERROR_RESPONSES,
    tags=["jobs"],
)
async def list_jobs(
    service: ServiceDependency,
    status: JobStatus | None = None,
    kind: Annotated[str | None, Query(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> JobList:
    return await service.list_jobs(
        status=status,
        kind=kind,
        limit=limit,
        offset=offset,
    )


@router.delete(
    "/api/v1/jobs/completed",
    status_code=204,
    responses=ERROR_RESPONSES,
    tags=["jobs"],
)
async def clear_completed_jobs(
    service: ServiceDependency,
) -> Response:
    await service.clear_completed_jobs()
    return Response(status_code=204)


@router.get(
    "/api/v1/jobs/{job_id}",
    response_model=JobResource,
    responses=ERROR_RESPONSES,
    tags=["jobs"],
)
async def get_job(service: ServiceDependency, job_id: UUID) -> JobResource:
    return await service.get_job(job_id)


@router.delete(
    "/api/v1/jobs/{job_id}",
    status_code=204,
    responses=ERROR_RESPONSES,
    tags=["jobs"],
)
async def delete_job(service: ServiceDependency, job_id: UUID) -> Response:
    await service.delete_job(job_id)
    return Response(status_code=204)


@router.post(
    "/api/v1/jobs/{job_id}/cancel",
    response_model=JobResource,
    responses=ERROR_RESPONSES,
    tags=["jobs"],
)
async def cancel_job(service: ServiceDependency, job_id: UUID) -> JobResource:
    return await service.cancel_job(job_id)


@router.post(
    "/api/v1/jobs/{job_id}/retry",
    response_model=JobResource,
    status_code=202,
    responses=ERROR_RESPONSES,
    tags=["jobs"],
)
async def retry_job(service: ServiceDependency, job_id: UUID) -> JobResource:
    return await service.retry_job(job_id)


@event_router.get(
    "/api/v1/jobs/{job_id}/events",
    response_model=JobEventList,
    responses=ERROR_RESPONSES,
    tags=["jobs"],
)
async def list_job_events(
    service: ServiceDependency,
    job_id: UUID,
    after: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> JobEventList:
    return await service.list_job_events(job_id, after=after, limit=limit)


@event_router.get(
    "/api/v1/jobs/{job_id}/events/stream",
    responses=ERROR_RESPONSES,
    tags=["jobs"],
)
async def stream_job_events(
    service: ServiceDependency,
    job_id: UUID,
    after: Annotated[int, Query(ge=0)] = 0,
    last_event_id: Annotated[int | None, Header(alias="Last-Event-ID", ge=0)] = None,
) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        cursor = max(after, last_event_id or 0)
        while True:
            result = await service.list_job_events(
                job_id,
                after=cursor,
                limit=200,
            )
            for event in result.data:
                cursor = event.sequence
                payload = event.model_dump_json()
                yield f"id: {event.sequence}\nevent: {event.type.value}\ndata: {payload}\n\n"
            job = await service.get_job(job_id)
            if job.status in {
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.INTERRUPTED,
            }:
                return
            if not result.data:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.25)

    await service.get_job(job_id)
    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
