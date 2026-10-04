/** Wrap resource requests for the sessions domain without storing component state. */
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
  /** Get recent sessions for the sidebar and session selector. */
  async listSessions(): Promise<Session[]> {
    return (await request<{ data: Session[] }>('/api/v1/sessions?limit=200')).data;
  },

  /** Create a session for a model and interaction mode, returning the server-assigned ID and revision. */
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

  /** Read the authoritative session state and revision, cancellable with the page or request. */
  getSession(id: string, signal?: AbortSignal): Promise<Session> {
    return request(`/api/v1/sessions/${id}`, { signal });
  },

  /** Update session title, mode, or metadata without changing message history. */
  updateSession(
    id: string,
    update: { title?: string | null; mode?: SessionMode; metadata?: Record<string, unknown> },
  ): Promise<Session> {
    return request(`/api/v1/sessions/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(update),
    });
  },

  /** Get persisted messages for initial loading and final synchronization after generation. */
  async listMessages(id: string, signal?: AbortSignal): Promise<Message[]> {
    return (await request<{ data: Message[] }>(`/api/v1/sessions/${id}/messages`, { signal })).data;
  },

  /** Get response status and performance records to verify request IDs and final messages. */
  async listResponses(id: string, signal?: AbortSignal): Promise<ResponseResource[]> {
    return (
      await request<{ data: ResponseResource[] }>(`/api/v1/sessions/${id}/responses?limit=1000`, {
        signal,
      })
    ).data;
  },

  /** Ask the server to cancel generation for this session; the caller verifies the final state. */
  cancelResponse(id: string, signal?: AbortSignal): Promise<ResponseResource> {
    return request(`/api/v1/sessions/${id}/responses/cancel`, { method: 'POST', signal });
  },

  /** Append messages using the expected revision; the server reports revision conflicts as errors. */
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

  /** Fork a session from a specified history position, optionally selecting a new model. */
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

  /** Rewind session history to a specified revision for editing and regeneration. */
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

  /** Delete a server-side session; the caller clears interface state on success. */
  async deleteSession(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/sessions/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  /** Export a session and media archive for user download or backup. */
  exportSession(id: string): Promise<SessionArchive> {
    return request(`/api/v1/sessions/${id}/export`);
  },

  /** Import a session archive and return the new sessions and media import count. */
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
