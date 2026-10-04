/** Wrap resource requests for the presets domain without storing component state. */
import type { GenerationPresetResource } from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';

export const presetsApi = {
  /** Get the list of generation presets saved on the server. */
  async generationPresets(): Promise<GenerationPresetResource[]> {
    return (await request<{ data: GenerationPresetResource[] }>('/api/v1/presets')).data;
  },

  /** Save a new generation preset and return the server resource. */
  createGenerationPreset(
    body: Omit<GenerationPresetResource, 'id' | 'created_at' | 'updated_at'>,
  ): Promise<GenerationPresetResource> {
    return request('/api/v1/presets', { method: 'POST', body: JSON.stringify(body) });
  },

  /** Replace the inference configuration of the specified preset. */
  updateGenerationPreset(
    id: string,
    body: Omit<GenerationPresetResource, 'id' | 'created_at' | 'updated_at'>,
  ): Promise<GenerationPresetResource> {
    return request(`/api/v1/presets/${id}`, {
      method: 'PUT',
      body: JSON.stringify(body),
    });
  },

  /** Delete the specified preset without changing existing sessions. */
  async deleteGenerationPreset(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/presets/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },
};
