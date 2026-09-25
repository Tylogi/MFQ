/** 封装 media 领域资源请求，不保存组件状态。 */
import type { MediaResource, DocumentResource } from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders, getApiToken } from '../client';

export const mediaApi = {
  /** 计算文件摘要并上传原始媒体，返回可用于消息的媒体引用。 */
  async uploadMedia(file: File, mimeType?: string): Promise<MediaResource> {
    const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
    const sha256 = Array.from(new Uint8Array(digest), (value) =>
      value.toString(16).padStart(2, '0'),
    ).join('');
    const response = await fetch(apiUrl('/api/v1/media'), {
      method: 'POST',
      headers: {
        'Content-Type': mimeType || file.type || 'application/octet-stream',
        'X-Content-SHA256': sha256,
        ...(getApiToken() ? { Authorization: `Bearer ${getApiToken()}` } : {}),
      },
      body: file,
    });
    if (!response.ok) throw await errorFromResponse(response);
    return (await response.json()) as MediaResource;
  },

  /** 返回媒体资源地址；需要鉴权读取时应调用 fetchMedia。 */
  mediaUrl(id: string): string {
    return apiUrl(`/api/v1/media/${id}`);
  },

  /** 带鉴权读取媒体二进制，支持组件卸载时取消。 */
  async fetchMedia(id: string, signal?: AbortSignal): Promise<Blob> {
    const response = await fetch(apiUrl(`/api/v1/media/${id}`), {
      headers: authorizedHeaders(),
      signal,
    });
    if (!response.ok) throw await errorFromResponse(response);
    return response.blob();
  },

  /** 请求服务端提取已上传文档，返回文档文本与媒体引用。 */
  createDocument(mediaId: string, name: string): Promise<DocumentResource> {
    return request('/api/v1/documents', {
      method: 'POST',
      body: JSON.stringify({ media_id: mediaId, name }),
    });
  },
};
