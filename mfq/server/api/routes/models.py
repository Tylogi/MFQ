"""Model, artifact, dataset, evaluation, hub, and cluster routes."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Query, Request, Response

from mfq.server.api.dependencies import ServiceDependency
from mfq.server.api.routes import ERROR_RESPONSES
from mfq.server.protocol.analysis import CheckpointAnalysis
from mfq.server.protocol.models import (
    ArtifactLineageList,
    CompareEvaluationsRequest,
    CreateDatasetRequest,
    CreateRemoteNodeRequest,
    DatasetList,
    DatasetResource,
    DeleteWorkspaceArtifactRequest,
    DeleteModelRequest,
    ModelDeletionPlan,
    EvaluationComparisonResource,
    EvaluationKind,
    EvaluationResultList,
    EvaluationToolsResource,
    HubModelInfo,
    HubModelSearchResult,
    HubReferenceRequest,
    HubDownloadProbeRequest,
    HubDownloadReuseRequest,
    HubDownloadReuseResponse,
    ModelArtifactList,
    ModelCacheProfile,
    ModelDirectoryList,
    ModelLoadRequest,
    ModelUnloadRequest,
    OfficialDatasetList,
    OfficialDatasetResource,
    OfficialModelList,
    OpenModelDirectoryRequest,
    OpenModelDirectoryResult,
    OperationAccepted,
    ProbabilityScoreRequest,
    ProbabilityScoreResult,
    RegisterModelDirectoryRequest,
    ConfigureDownloadDirectoryRequest,
    RemoteNodeList,
    RemoteNodeResource,
    UpdateRemoteNodeRequest,
)
from mfq.server.services.service import ServiceError

hub_router = APIRouter()
router = APIRouter()


@hub_router.post('/api/v1/hub/download-sources/probe', responses=ERROR_RESPONSES, tags=['models'])
async def probe_download_sources(service: ServiceDependency, body: HubDownloadProbeRequest) -> dict[str, Any]:
    return await service.hub_catalog.probe_download_sources(body.sources)


@hub_router.post('/api/v1/hub/download-reuse', response_model=HubDownloadReuseResponse,
                 responses=ERROR_RESPONSES, tags=['models'])
async def download_reuse(service: ServiceDependency, body: HubDownloadReuseRequest) -> HubDownloadReuseResponse:
    from mfq.server.services.hub import HubError
    from mfq.server.services.jobs import JobExecutionError
    if service.tool_handlers is None:
        raise ServiceError(422, 'download_unavailable', 'Model downloads are unavailable on this server.')
    try:
        info = await service.hub_catalog.download_files(body.provider, body.repo_id, body.revision)
        reusable = await asyncio.to_thread(service.tool_handlers.download_reuse,
                                          body.destination, body.download_directory, info.files)
        return HubDownloadReuseResponse(reusable_files=reusable)
    except JobExecutionError as error:
        raise ServiceError(422, error.detail.code, error.detail.message) from error
    except (HubError, asyncio.TimeoutError) as error:
        raise ServiceError(502, 'model_hub_error', 'Could not check reusable model files.', retryable=True) from error

@hub_router.get("/api/v1/hub/download-directory", responses=ERROR_RESPONSES, tags=["models"])
async def download_directory(service: ServiceDependency) -> dict[str, str | int | bool | None]:
    return await asyncio.to_thread(service.tool_handlers.download_directory_status)


@hub_router.put("/api/v1/hub/download-directory", responses=ERROR_RESPONSES, tags=["models"])
async def configure_download_directory(service: ServiceDependency, body: ConfigureDownloadDirectoryRequest) -> dict[str, str | int | bool | None]:
    from mfq.server.state.catalog import ModelDirectoryNotFoundError
    try:
        settings = service.tool_handlers.download_directories
        directory = (service.catalog.directory_path(body.directory_id) if body.directory_id else None) if 'directory_id' in body.model_fields_set else settings.selected
        await settings.configure(directory, evict_prefix_cache=body.evict_prefix_cache)
        return await asyncio.to_thread(service.tool_handlers.download_directory_status)
    except (ModelDirectoryNotFoundError, ValueError) as error:
        raise ServiceError(400, "invalid_download_directory", str(error)) from error
    except OSError:
        raise ServiceError(500, "download_directory_save_failed", "could not save download directory") from None


@hub_router.get(
    "/api/v1/hub/official",
    response_model=OfficialModelList,
    responses=ERROR_RESPONSES,
    tags=["models"],
)
async def official_hub_models(
    service: ServiceDependency,
    refresh: bool = False,
) -> OfficialModelList:
    return await service.official_hub_models(refresh=refresh)


@hub_router.post(
    "/api/v1/hub/resolve",
    response_model=HubModelInfo,
    responses=ERROR_RESPONSES,
    tags=["models"],
)
async def resolve_hub_model(
    service: ServiceDependency,
    body: HubReferenceRequest,
) -> HubModelInfo:
    return await service.resolve_hub_model(body)


@hub_router.get(
    "/api/v1/hub/models",
    response_model=HubModelSearchResult,
    responses=ERROR_RESPONSES,
    tags=["models"],
)
async def search_hub_models(
    service: ServiceDependency,
    provider: Annotated[str, Query(pattern="^(huggingface|modelscope)$")],
    query: Annotated[str, Query(min_length=1, max_length=255)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> HubModelSearchResult:
    return await service.search_hub_models(provider, query, limit)


@hub_router.get(
    "/api/v1/hub/models/{provider}/{owner}/{name}",
    response_model=HubModelInfo,
    responses=ERROR_RESPONSES,
    tags=["models"],
)
async def hub_model_info(
    service: ServiceDependency,
    provider: str,
    owner: str,
    name: str,
    revision: Annotated[str | None, Query(max_length=255)] = None,
) -> HubModelInfo:
    if provider not in {"huggingface", "modelscope"}:
        raise ServiceError(404, "model_hub_not_found", provider)
    return await service.hub_model_info(provider, f"{owner}/{name}", revision)


@router.post(
    "/api/v1/models/load",
    response_model=OperationAccepted,
    status_code=202,
    responses=ERROR_RESPONSES,
    tags=["models"],
)
async def load_model(service: ServiceDependency, body: ModelLoadRequest) -> OperationAccepted:
    return await service.load_model(body)


@router.get(
    "/api/v1/models",
    response_model=ModelArtifactList,
    responses=ERROR_RESPONSES,
    tags=["models"],
)
async def model_artifacts(service: ServiceDependency, refresh: bool = False) -> ModelArtifactList:
    return await service.model_artifacts(refresh=refresh)


@router.get(
    "/api/v1/models/directories",
    response_model=ModelDirectoryList,
    responses=ERROR_RESPONSES,
    tags=["models"],
)
async def model_directories(
    service: ServiceDependency,
    directory_id: Annotated[
        str | None,
        Query(pattern=r"^[0-9a-f]{32}$"),
    ] = None,
    path: Annotated[
        str | None,
        Query(min_length=1, max_length=4096),
    ] = None,
) -> ModelDirectoryList:
    return await service.model_directories(directory_id=directory_id, path=path)


@router.post(
    "/api/v1/models/directories/open",
    response_model=OpenModelDirectoryResult,
    responses=ERROR_RESPONSES,
    tags=["models"],
)
async def open_model_directory(service: ServiceDependency, body: OpenModelDirectoryRequest) -> OpenModelDirectoryResult:
    return await service.open_model_directory(body)


@router.post(
    "/api/v1/models/directories/register",
    response_model=ModelArtifactList,
    responses=ERROR_RESPONSES,
    tags=["models"],
)
async def register_model_directory(
    service: ServiceDependency,
    body: RegisterModelDirectoryRequest,
) -> ModelArtifactList:
    return await service.register_model_directory(body)


@router.get(
    "/api/v1/models/{model_id}/directory",
    response_model=ModelDirectoryList,
    responses=ERROR_RESPONSES,
    tags=["models"],
)
async def model_artifact_directory(service: ServiceDependency, model_id: str) -> ModelDirectoryList:
    return await service.model_artifact_directory(model_id)


@router.get("/api/v1/models/{model_id}/deletion", response_model=ModelDeletionPlan, responses=ERROR_RESPONSES, tags=["models"])
async def model_deletion_plan(service: ServiceDependency, model_id: str) -> ModelDeletionPlan:
    return await service.model_deletion_plan(model_id)


@router.post("/api/v1/models/{model_id}/delete", responses=ERROR_RESPONSES, tags=["models"])
async def delete_model_artifact(service: ServiceDependency, model_id: str, body: DeleteModelRequest) -> dict[str, int]:
    return await service.delete_model_artifact(model_id, body.token)


@router.get("/api/v1/models/{model_id}/cache-profile", response_model=ModelCacheProfile | None,
            responses=ERROR_RESPONSES, tags=["models"])
async def model_cache_profile(service: ServiceDependency, model_id: str,
    include_recurrent: bool = False) -> ModelCacheProfile | None:
    return await service.model_cache_profile(model_id, include_recurrent=include_recurrent)


@router.get("/api/v1/models/{model_id}/analysis", response_model=CheckpointAnalysis,
            responses=ERROR_RESPONSES, tags=["models"])
async def checkpoint_analysis(service: ServiceDependency, model_id: str) -> CheckpointAnalysis:
    return await service.checkpoint_analysis(model_id)


@router.get(
    "/api/v1/artifacts/lineage",
    response_model=ArtifactLineageList,
    responses=ERROR_RESPONSES,
    tags=["models"],
)
async def artifact_lineage(
    service: ServiceDependency,
    artifact_uri: Annotated[str | None, Query(max_length=2048)] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> ArtifactLineageList:
    return await service.artifact_lineage(
        artifact_uri=artifact_uri,
        limit=limit,
    )


@router.post(
    "/api/v1/artifacts/remove",
    response_model=dict[str, Any],
    responses=ERROR_RESPONSES,
    tags=["models"],
)
async def delete_workspace_artifact(
    service: ServiceDependency,
    body: DeleteWorkspaceArtifactRequest,
) -> dict[str, Any]:
    return await service.delete_workspace_artifact(body)


@router.post(
    "/api/v1/datasets",
    response_model=DatasetResource,
    status_code=201,
    responses=ERROR_RESPONSES,
    tags=["evaluation"],
)
async def create_dataset(service: ServiceDependency, body: CreateDatasetRequest) -> DatasetResource:
    return await service.create_dataset(body)


@router.get(
    "/api/v1/datasets",
    response_model=DatasetList,
    responses=ERROR_RESPONSES,
    tags=["evaluation"],
)
async def list_datasets(
    service: ServiceDependency,
) -> DatasetList:
    return await service.list_datasets()


@router.get("/api/v1/datasets/catalog", response_model=OfficialDatasetList, responses=ERROR_RESPONSES, tags=["evaluation"])
async def official_dataset_catalog(service: ServiceDependency) -> OfficialDatasetList:
    from dataclasses import asdict

    from mfq.server.services.evaluation_datasets import OFFICIAL_DATASETS
    return OfficialDatasetList(data=[OfficialDatasetResource.model_validate(asdict(spec)) for spec in OFFICIAL_DATASETS.values()])


@router.delete(
    "/api/v1/datasets/{dataset_id}",
    status_code=204,
    responses=ERROR_RESPONSES,
    tags=["evaluation"],
)
async def delete_dataset(service: ServiceDependency, dataset_id: UUID) -> Response:
    await service.delete_dataset(dataset_id)
    return Response(status_code=204)


@router.get("/api/v1/evaluations/tools", response_model=EvaluationToolsResource, responses=ERROR_RESPONSES, tags=["evaluation"])
async def evaluation_tools(service: ServiceDependency, request: Request) -> EvaluationToolsResource:
    from mfq.server.services.official_benchmarks import benchmark_readiness
    handlers = service.jobs.handlers
    root = getattr(service.tool_handlers, "root", None)
    return EvaluationToolsResource(workspace_root=str(root) if root else None,
        api_base=str(request.base_url).rstrip("/") + "/v1",
        quality_available="evaluate.wikitext2" in handlers,
        reference_available="reference.wikitext2" in handlers,
        benchmark_available="benchmark.inference" in handlers,
        accuracy_available="evaluate.accuracy" in handlers,
        task_benchmarks=benchmark_readiness(root) if "evaluate.accuracy" in handlers else {})


@router.post("/api/v1/evaluations/probabilities", response_model=ProbabilityScoreResult, responses=ERROR_RESPONSES, tags=["evaluation"])
async def evaluation_probabilities(service: ServiceDependency, body: ProbabilityScoreRequest) -> ProbabilityScoreResult:
    from mfq.server.runtime.client import BackendError
    pool = getattr(service.tool_handlers, "runtime_manager", None)
    if pool is None:
        raise ServiceError(501, "probability_unavailable", "no managed runtime is configured")
    try:
        async with pool.benchmark_runtime(body.instance_id) as (_, backend):
            if not callable(getattr(backend, "score", None)) or not await backend.probability_available():
                raise ServiceError(501, "probability_unavailable", "the loaded runtime does not support probability scoring")
            result = await backend.score(body.model_dump(mode="json", exclude={"instance_id"}))
            return ProbabilityScoreResult.model_validate(result)
    except BackendError as error:
        raise ServiceError(error.status_code or 502, error.code, str(error), retryable=error.retryable) from error


@router.get(
    "/api/v1/evaluations",
    response_model=EvaluationResultList,
    responses=ERROR_RESPONSES,
    tags=["evaluation"],
)
async def evaluations(
    service: ServiceDependency,
    kind: EvaluationKind | None = None,
    model_id: Annotated[str | None, Query(max_length=255)] = None,
    limit: Annotated[int, Query(ge=1, le=2000)] = 200,
) -> EvaluationResultList:
    return await service.evaluations(kind=kind, model_id=model_id, limit=limit)


@router.post(
    "/api/v1/evaluations/compare",
    response_model=EvaluationComparisonResource,
    responses=ERROR_RESPONSES,
    tags=["evaluation"],
)
async def compare_evaluations(
    service: ServiceDependency,
    body: CompareEvaluationsRequest,
) -> EvaluationComparisonResource:
    return await service.compare_evaluations(body)


@router.post(
    "/api/v1/cluster/nodes",
    response_model=RemoteNodeResource,
    status_code=201,
    responses=ERROR_RESPONSES,
    tags=["cluster"],
)
async def create_remote_node(
    service: ServiceDependency, body: CreateRemoteNodeRequest
) -> RemoteNodeResource:
    return await service.create_remote_node(body)


@router.get(
    "/api/v1/cluster/nodes",
    response_model=RemoteNodeList,
    responses=ERROR_RESPONSES,
    tags=["cluster"],
)
async def remote_nodes(service: ServiceDependency, refresh: bool = False) -> RemoteNodeList:
    return await service.remote_nodes(refresh=refresh)


@router.put(
    "/api/v1/cluster/nodes/{node_id}",
    response_model=RemoteNodeResource,
    responses=ERROR_RESPONSES,
    tags=["cluster"],
)
async def update_remote_node(
    service: ServiceDependency, node_id: UUID, body: UpdateRemoteNodeRequest
) -> RemoteNodeResource:
    return await service.update_remote_node(node_id, body)


@router.delete(
    "/api/v1/cluster/nodes/{node_id}",
    status_code=204,
    responses=ERROR_RESPONSES,
    tags=["cluster"],
)
async def delete_remote_node(service: ServiceDependency, node_id: UUID) -> Response:
    await service.delete_remote_node(node_id)
    return Response(status_code=204)


@router.post(
    "/api/v1/models/unload",
    response_model=OperationAccepted,
    status_code=202,
    responses=ERROR_RESPONSES,
    tags=["models"],
)
async def unload_model(service: ServiceDependency, body: ModelUnloadRequest) -> OperationAccepted:
    return await service.unload_model(body)
