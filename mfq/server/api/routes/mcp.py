"""MCP server and tool routes."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Response

from mfq.server.api.dependencies import ServiceDependency
from mfq.server.api.routes import ERROR_RESPONSES
from mfq.server.protocol.models import (
    CreateMcpServerRequest,
    McpServerList,
    McpServerResource,
    McpToolCallRequest,
    McpToolCallResult,
    McpToolList,
    UpdateMcpServerRequest,
)

router = APIRouter()


@router.post(
    "/api/v1/mcp/servers",
    response_model=McpServerResource,
    status_code=201,
    responses=ERROR_RESPONSES,
    tags=["mcp"],
)
async def create_mcp_server(
    service: ServiceDependency, body: CreateMcpServerRequest
) -> McpServerResource:
    return await service.create_mcp_server(body)


@router.get(
    "/api/v1/mcp/servers",
    response_model=McpServerList,
    responses=ERROR_RESPONSES,
    tags=["mcp"],
)
async def list_mcp_servers(
    service: ServiceDependency,
) -> McpServerList:
    return await service.list_mcp_servers()


@router.patch(
    "/api/v1/mcp/servers/{server_id}",
    response_model=McpServerResource,
    responses=ERROR_RESPONSES,
    tags=["mcp"],
)
async def update_mcp_server(
    service: ServiceDependency, server_id: UUID, body: UpdateMcpServerRequest
) -> McpServerResource:
    return await service.update_mcp_server(server_id, body)


@router.delete(
    "/api/v1/mcp/servers/{server_id}",
    status_code=204,
    responses=ERROR_RESPONSES,
    tags=["mcp"],
)
async def delete_mcp_server(service: ServiceDependency, server_id: UUID) -> Response:
    await service.delete_mcp_server(server_id)
    return Response(status_code=204)


@router.get(
    "/api/v1/mcp/tools",
    response_model=McpToolList,
    responses=ERROR_RESPONSES,
    tags=["mcp"],
)
async def list_mcp_tools(
    service: ServiceDependency,
) -> McpToolList:
    return await service.list_mcp_tools()


@router.post(
    "/api/v1/mcp/tools/call",
    response_model=McpToolCallResult,
    responses=ERROR_RESPONSES,
    tags=["mcp"],
)
async def call_mcp_tool(service: ServiceDependency, body: McpToolCallRequest) -> McpToolCallResult:
    return await service.call_mcp_tool(body)
