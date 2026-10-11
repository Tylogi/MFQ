/** 封装 connections 领域资源请求，不保存组件状态。 */
import type { RemoteNode } from '../types';
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

};
