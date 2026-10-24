"""Prometheus scrape endpoint for the serving runtimes."""

from __future__ import annotations

from fastapi import APIRouter, Response

from mfq.server.api.dependencies import ServiceDependency
from mfq.server.services.metrics import PROMETHEUS_CONTENT_TYPE

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics(service: ServiceDependency) -> Response:
    # The content type of the text format carries the format version, so it is
    # set exactly instead of letting the framework append a charset parameter.
    return Response(
        content=await service.prometheus_metrics(),
        headers={"content-type": PROMETHEUS_CONTENT_TYPE},
    )
