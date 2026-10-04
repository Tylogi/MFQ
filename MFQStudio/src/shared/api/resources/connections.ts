/** Wrap resource requests for the connections domain without storing component state. */
import type { RemoteNode, McpServerResource, McpToolResource, McpToolCallResult } from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';

export const connectionsApi = {
  /** Get remote node status, optionally asking the server to refresh health information. */
  async remoteNodes(refresh = false): Promise<RemoteNode[]> {
    return (
      await request<{ data: RemoteNode[] }>(
        `/api/v1/cluster/nodes${refresh ? '?refresh=true' : ''}`,
      )
    ).data;
  },

  /** Register a remote service node and the name of its credential environment variable. */
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

  /** Remove a remote node registration. */
  async deleteRemoteNode(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/cluster/nodes/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  /** Get the list of MCP service configurations. */
  async mcpServers(): Promise<McpServerResource[]> {
    return (await request<{ data: McpServerResource[] }>('/api/v1/mcp/servers')).data;
  },

  /** Aggregate MCP tools and discovery errors from each service. */
  async mcpTools(): Promise<{ data: McpToolResource[]; errors: Record<string, string> }> {
    return request('/api/v1/mcp/tools');
  },

  /** Register an HTTP or stdio MCP service. */
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

  /** Enable or disable a registered MCP service. */
  updateMcpServer(id: string, enabled: boolean): Promise<McpServerResource> {
    return request(`/api/v1/mcp/servers/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ enabled }),
    });
  },

  /** Delete an MCP service configuration. */
  async deleteMcpServer(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/mcp/servers/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  /** Execute a tool call confirmed by the user and return its output. */
  callMcpTool(name: string, arguments_: Record<string, unknown>): Promise<McpToolCallResult> {
    return request('/api/v1/mcp/tools/call', {
      method: 'POST',
      body: JSON.stringify({ name, arguments: arguments_, confirm: true }),
    });
  },
};
