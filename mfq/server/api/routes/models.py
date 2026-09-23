"""Model, artifact, dataset, evaluation, hub, and cluster routes."""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Query, Response

from mfq.server.api.dependencies import ServiceDependency
from mfq.server.api.routes import ERROR_RESPONSES
from mfq.server.protocol.models import (
    ArtifactLineageList,
    CompareEvaluationsRequest,
    CreateDatasetRequest,
    CreateRemoteNodeRequest,
    DatasetList,
    DatasetResource,
    DeleteWorkspaceArtifactRequest,
    EvaluationComparisonResource,
    EvaluationKind,
    EvaluationResultList,
    HubModelInfo,
    HubModelSearchResult,
    ModelArtifactList,
    ModelDirectoryList,
    ModelLoadRequest,
    ModelUnloadRequest,
    OperationAccepted,
    RegisterModelDirectoryRequest,
    RemoteNodeList,
    RemoteNodeResource,
    UpdateRemoteNodeRequest,
)
from mfq.server.services.service import ServiceError

hub_router = APIRouter()
router = APIRouter()


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


@router.delete(
    "/api/v1/datasets/{dataset_id}",
    status_code=204,
    responses=ERROR_RESPONSES,
    tags=["evaluation"],
)
async def delete_dataset(service: ServiceDependency, dataset_id: UUID) -> Response:
    await service.delete_dataset(dataset_id)
    return Response(status_code=204)


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
