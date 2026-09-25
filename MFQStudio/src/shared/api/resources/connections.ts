/** 封装 connections 领域资源请求，不保存组件状态。 */
import type { RemoteNode, McpServerResource, McpToolResource, McpToolCallResult } from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';

export const connectionsApi = {
  /** 获取远程节点状态，可要求服务刷新健康信息。 */
  async remoteNodes(refresh = false): Promise<RemoteNode[]> {
    return (
      await request<{ data: RemoteNode[] }>(
        `/api/v1/cluster/nodes${refresh ? '?refresh=true' : ''}`,
      )
    ).data;
  },

  /** 注册远程服务节点及其凭据环境变量名称。 */
  createRemoteNode(body: {
    name: string;
    url: string;
    api_key_env?: string | null;
    enabled: boolean;
  }): Promise<RemoteNode> {
    return request('/api/v1/cluster/nodes', {
      method: 'POST',
      body: JSON.stringify(body),
    });
  },

  /** 移除远程节点注册。 */
  async deleteRemoteNode(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/cluster/nodes/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  /** 获取 MCP 服务配置列表。 */
  async mcpServers(): Promise<McpServerResource[]> {
    return (await request<{ data: McpServerResource[] }>('/api/v1/mcp/servers')).data;
  },

  /** 聚合 MCP 工具及各服务的发现错误。 */
  async mcpTools(): Promise<{ data: McpToolResource[]; errors: Record<string, string> }> {
    return request('/api/v1/mcp/tools');
  },

  /** 注册 HTTP 或 stdio MCP 服务。 */
  createMcpServer(body: {
    name: string;
    transport: 'stdio' | 'streamable_http';
    enabled: boolean;
    url?: string | null;
    command?: string | null;
    args?: string[];
    header_env?: Record<string, string>;
  }): Promise<McpServerResource> {
    return request('/api/v1/mcp/servers', { method: 'POST', body: JSON.stringify(body) });
  },

  /** 启用或禁用已注册的 MCP 服务。 */
  updateMcpServer(id: string, enabled: boolean): Promise<McpServerResource> {
    return request(`/api/v1/mcp/servers/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ enabled }),
    });
  },

  /** 删除 MCP 服务配置。 */
  async deleteMcpServer(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/mcp/servers/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  /** 执行已获用户确认的工具调用，返回工具输出。 */
  callMcpTool(name: string, arguments_: Record<string, unknown>): Promise<McpToolCallResult> {
    return request('/api/v1/mcp/tools/call', {
      method: 'POST',
      body: JSON.stringify({ name, arguments: arguments_, confirm: true }),
    });
  },
};
