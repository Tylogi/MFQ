/** 封装 sessions 领域资源请求，不保存组件状态。 */
import type {
  SessionMode,
  MessageRole,
  ContentPart,
  Session,
  SessionArchive,
  Message,
  ResponseResource,
} from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';

export const sessionsApi = {
  /** 获取最近的会话列表，供侧栏与会话选择使用。 */
  async listSessions(): Promise<Session[]> {
    return (await request<{ data: Session[] }>('/api/v1/sessions?limit=200')).data;
  },

  /** 按模型和交互模式创建会话，返回服务端分配的标识与版本。 */
  createSession(
    model: string,
    mode: SessionMode,
    title?: string,
    metadata?: Record<string, unknown>,
  ): Promise<Session> {
    return request('/api/v1/sessions', {
      method: 'POST',
      body: JSON.stringify({ model, mode, title, metadata: metadata ?? {} }),
    });
  },

  /** 读取会话的权威状态和版本号，可随页面或请求取消。 */
  getSession(id: string, signal?: AbortSignal): Promise<Session> {
    return request(`/api/v1/sessions/${id}`, { signal });
  },

  /** 更新会话标题、模式或元数据，不修改历史消息。 */
  updateSession(
    id: string,
    update: { title?: string | null; mode?: SessionMode; metadata?: Record<string, unknown> },
  ): Promise<Session> {
    return request(`/api/v1/sessions/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(update),
    });
  },

  /** 获取持久化消息，供首次加载及生成完成后的最终同步使用。 */
  async listMessages(id: string, signal?: AbortSignal): Promise<Message[]> {
    return (await request<{ data: Message[] }>(`/api/v1/sessions/${id}/messages`, { signal })).data;
  },

  /** 获取响应状态与性能记录，用于核对请求标识和最终消息。 */
  async listResponses(id: string, signal?: AbortSignal): Promise<ResponseResource[]> {
    return (
      await request<{ data: ResponseResource[] }>(`/api/v1/sessions/${id}/responses?limit=1000`, {
        signal,
      })
    ).data;
  },

  /** 请求服务端取消该会话正在运行的生成，调用方负责确认最终状态。 */
  cancelResponse(id: string, signal?: AbortSignal): Promise<ResponseResource> {
    return request(`/api/v1/sessions/${id}/responses/cancel`, { method: 'POST', signal });
  },

  /** 使用预期版本追加消息，版本冲突由服务端返回错误。 */
  appendMessage(
    id: string,
    expectedRevision: number,
    role: MessageRole,
    parts: ContentPart[],
  ): Promise<{ session: Session; message: Message }> {
    return request(`/api/v1/sessions/${id}/messages`, {
      method: 'POST',
      body: JSON.stringify({ expected_revision: expectedRevision, role, parts }),
    });
  },

  /** 从指定历史位置派生会话，可同时指定新的模型。 */
  forkSession(
    id: string,
    atMessageId: string | null,
    includeMessage = true,
    title?: string | null,
    model?: string,
  ): Promise<Session> {
    return request(`/api/v1/sessions/${id}/fork`, {
      method: 'POST',
      body: JSON.stringify({
        at_message_id: atMessageId,
        include_message: includeMessage,
        title,
        model,
      }),
    });
  },

  /** 按版本回退会话历史，供编辑和重新生成操作使用。 */
  rewindSession(
    id: string,
    expectedRevision: number,
    atMessageId: string,
    includeMessage = true,
  ): Promise<Session> {
    return request(`/api/v1/sessions/${id}/rewind`, {
      method: 'POST',
      body: JSON.stringify({
        expected_revision: expectedRevision,
        at_message_id: atMessageId,
        include_message: includeMessage,
      }),
    });
  },

  /** 删除指定服务端会话，成功后由调用方清理界面状态。 */
  async deleteSession(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/sessions/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  /** 导出会话及媒体归档，供用户下载或备份。 */
  exportSession(id: string): Promise<SessionArchive> {
    return request(`/api/v1/sessions/${id}/export`);
  },

  /** 导入会话归档并返回新会话及媒体导入数量。 */
  importSession(archive: SessionArchive): Promise<{
    session: Session;
    messages_imported: number;
    media_imported: number;
  }> {
    return request('/api/v1/sessions/import', {
      method: 'POST',
      body: JSON.stringify(archive),
    });
  },
};
