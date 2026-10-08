import { useState } from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { useGenerationPresets } from './useGenerationPresets';
import { DEFAULT_SETTINGS } from './configuration';
import { presetResourceBody, presetSnapshot, STORED_PRESETS_KEY, type StoredPreset } from './presets';
import { presetsApi } from '../../shared/api/resources/presets';
import { setApiBaseUrl } from '../../shared/api/client';
import { studioConfirm } from '../../studio';
import type { GenerationPresetResource } from '../../shared/api/types';

vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ connectionRevision: 1 }) }));
vi.mock('./SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en,
  contextSize: 32768, setContextSize: vi.fn() }) }));
vi.mock('../../shared/api/resources/presets', () => ({ presetsApi: {
  generationPresets: vi.fn(), createGenerationPreset: vi.fn(), updateGenerationPreset: vi.fn(), deleteGenerationPreset: vi.fn(),
} }));
vi.mock('../../studio', () => ({ studioConfirm: vi.fn() }));

const preset: StoredPreset = { id: 'old-id', name: 'Old server preset', contextSize: 32768,
  inheritGlobalSettings: false, settings: presetSnapshot(DEFAULT_SETTINGS), updatedAt: '2026-10-06' };
const resource = { ...presetResourceBody(preset, 'model', 'text'), id: preset.id,
  created_at: preset.updatedAt, updated_at: preset.updatedAt } as GenerationPresetResource;
function Manager() {
  const [draft, setDraft] = useState({ ...DEFAULT_SETTINGS, inheritModelDefaults: false });
  return useGenerationPresets(draft, setDraft, 'model', 'text', true).manager;
}
beforeEach(() => {
  localStorage.clear(); setApiBaseUrl('http://server-a');
  vi.mocked(presetsApi.generationPresets).mockResolvedValue([resource]);
  vi.mocked(studioConfirm).mockResolvedValue(true);
});
afterEach(() => { vi.clearAllMocks(); setApiBaseUrl(''); });

it('keeps cached server resource IDs separate between connections even if the new server is offline', async () => {
  const first = render(<Manager />);
  await screen.findByRole('option', { name: preset.name });
  await waitFor(() => expect(localStorage.getItem(`${STORED_PRESETS_KEY}:http://server-a`)).toContain('old-id'));
  first.unmount();
  setApiBaseUrl('http://server-b');
  vi.mocked(presetsApi.generationPresets).mockRejectedValue(new Error('new server offline'));
  render(<Manager />);
  await screen.findByText('new server offline');
  expect(screen.queryByRole('option', { name: preset.name })).not.toBeInTheDocument();
  expect(presetsApi.updateGenerationPreset).not.toHaveBeenCalled();
});

it('does not delete a preset after its confirmation outlives the page', async () => {
  let confirm!: (value: boolean) => void;
  vi.mocked(studioConfirm).mockReturnValue(new Promise((resolve) => { confirm = resolve; }));
  const view = render(<Manager />);
  await screen.findByRole('option', { name: preset.name });
  fireEvent.change(screen.getByLabelText('Saved presets'), { target: { value: preset.name } });
  fireEvent.click(screen.getByRole('button', { name: 'Delete preset' }));
  view.unmount(); setApiBaseUrl('http://server-b');
  await act(async () => { confirm(true); });
  expect(presetsApi.deleteGenerationPreset).not.toHaveBeenCalled();
});

it('does not overwrite the new server cache with a late save from the old page', async () => {
  let finish!: (value: GenerationPresetResource) => void;
  vi.mocked(presetsApi.updateGenerationPreset).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
  const view = render(<Manager />);
  await screen.findByRole('option', { name: preset.name });
  fireEvent.change(screen.getByLabelText('Saved presets'), { target: { value: preset.name } });
  fireEvent.click(screen.getByRole('button', { name: 'Update' }));
  view.unmount(); setApiBaseUrl('http://server-b');
  await act(async () => { finish({ ...resource, name: 'Late old response' }); });
  expect(localStorage.getItem(`${STORED_PRESETS_KEY}:http://server-b`)).toBeNull();
  expect(localStorage.getItem(`${STORED_PRESETS_KEY}:http://server-a`)).not.toContain('Late old response');
});
