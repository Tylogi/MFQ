from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from mfq.server.api.dependencies import ServiceDependency
from mfq.server.api.routes import ERROR_RESPONSES
from mfq.server.protocol.quantization import (
    QuantizationDirectories, QuantizationFiles, QuantizationSource, QuantizationSourceRequest, QuantizationWorkspace,
    SourceLoadingRequest, SourceLoadingPlan,
)
from mfq.server.services.jobs import JobExecutionError
from mfq.server.services.service import ServiceError

router = APIRouter(tags=['quantization'], responses=ERROR_RESPONSES)


def workbench(service):
    if service.tool_handlers is None:
        raise ServiceError(501, 'quantization_unavailable', 'Python quantization tools are not configured')
    return service.tool_handlers.quantization_workbench


async def invoke(function, *args):
    try:
        return await asyncio.to_thread(function, *args)
    except JobExecutionError as error:
        raise ServiceError(422, error.detail.code, error.detail.message, details=error.detail.details) from error
    except (ValueError, OSError, KeyError) as error:
        raise ServiceError(422, 'quantization_validation_failed', str(error)) from error


@router.get('/api/v1/quantization/workspace', response_model=QuantizationWorkspace)
async def workspace(service: ServiceDependency):
    return await invoke(workbench(service).workspace)


@router.put('/api/v1/quantization/workspace', response_model=QuantizationWorkspace)
async def configure_workspace(service: ServiceDependency, body: QuantizationDirectories):
    return await invoke(workbench(service).configure, body)


@router.get('/api/v1/quantization/files', response_model=QuantizationFiles)
async def files(service: ServiceDependency, path: str | None = Query(default=None, max_length=4096)):
    return await invoke(workbench(service).files, path)


@router.post('/api/v1/quantization/source', response_model=QuantizationSource)
async def source(service: ServiceDependency, body: QuantizationSourceRequest):
    return await invoke(workbench(service).source, body.path)


@router.get('/api/v1/quantization/jobs/{job_id}/recipe')
async def export_recipe(service: ServiceDependency, job_id: UUID):
    job = await service.get_job(job_id)
    if job.status.value != 'succeeded' or not job.result or job.result.get('artifact_kind') != 'recipe':
        raise ServiceError(409, 'recipe_not_ready', 'this task has no completed recipe')
    path = Path(job.result['output'])
    if not path.is_file():
        raise ServiceError(404, 'recipe_not_found', 'recipe file is no longer available')
    return FileResponse(path, media_type='application/json', filename=path.name)


@router.post('/api/v1/quantization/loading-plan', response_model=SourceLoadingPlan)
async def loading_plan(service: ServiceDependency, body: SourceLoadingRequest):
    return await invoke(workbench(service).loading_plan, body)
