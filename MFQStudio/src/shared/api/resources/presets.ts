/** 封装 presets 领域资源请求，不保存组件状态。 */
import type { GenerationPresetResource } from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';

export const presetsApi = {
  /** 获取服务端保存的生成预设列表。 */
  async generationPresets(): Promise<GenerationPresetResource[]> {
    return (await request<{ data: GenerationPresetResource[] }>('/api/v1/presets')).data;
  },

  /** 保存新的生成预设，返回服务端资源。 */
  createGenerationPreset(
    body: Omit<GenerationPresetResource, 'id' | 'created_at' | 'updated_at'>,
  ): Promise<GenerationPresetResource> {
    return request('/api/v1/presets', { method: 'POST', body: JSON.stringify(body) });
  },

  /** 覆盖指定预设的推理配置。 */
  updateGenerationPreset(
    id: string,
    body: Omit<GenerationPresetResource, 'id' | 'created_at' | 'updated_at'>,
  ): Promise<GenerationPresetResource> {
    return request(`/api/v1/presets/${id}`, {
      method: 'PUT',
      body: JSON.stringify(body),
    });
  },

  /** 删除指定预设，不修改已经创建的会话。 */
  async deleteGenerationPreset(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/presets/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },
};
