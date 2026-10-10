import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { beforeEach, expect, it, vi } from 'vitest';
import { useRuntime } from '../src/app/RuntimeProvider';
import { useSettings } from '../src/features/settings/SettingsProvider';
import { DEFAULT_SETTINGS } from '../src/features/settings/configuration';
import { SettingsRoute } from '../src/features/settings/SettingsRoute';
import { runtimeApi } from '../src/shared/api/resources/runtime';
import { studioConfirm } from '../src/studio';

vi.mock('../src/app/RuntimeProvider', () => ({ useRuntime: vi.fn() }));
vi.mock('../src/features/settings/SettingsProvider', () => ({ useSettings: vi.fn() }));
vi.mock('../src/shared/api/resources/runtime', () => ({ runtimeApi: { reloadRuntime: vi.fn() } }));
vi.mock('../src/studio', () => ({ studioConfirm: vi.fn() }));
vi.mock('../src/features/chat/hooks/useActiveSessionMode', () => ({ useActiveSessionMode: () => 'text' }));
vi.mock('../src/features/settings/useGenerationPresets', () => ({
  useGenerationPresets: () => ({ presets: [], setPresets: vi.fn(), clearSelection: vi.fn(), manager: null }),
}));

const refreshRuntime = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  refreshRuntime.mockResolvedValue(true);
  vi.mocked(useSettings).mockReturnValue({
    settings: { ...DEFAULT_SETTINGS, inheritModelDefaults: false },
    contextSize: 8192, setContextSize: vi.fn(), replaceSettings: vi.fn(),
    tr: (_zh: string, en: string) => en,
  } as unknown as ReturnType<typeof useSettings>);
  vi.mocked(useRuntime).mockReturnValue({
    runtime: { instance_id: 'chosen-instance', context_capacity: 32768 },
    realtime: null, selectedModel: 'chosen-model', studio: null,
    ready: true, refreshRuntime, instances: [], capabilities: null,
  } as unknown as ReturnType<typeof useRuntime>);
});

it('keeps context editing in service management instead of duplicating it in settings', () => {
  render(<MemoryRouter><SettingsRoute /></MemoryRouter>);
  expect(screen.queryByText('Context', { exact: true })).not.toBeInTheDocument();
  expect(screen.queryByRole('spinbutton', { name: 'Global context cap' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Reload model with this context' })).not.toBeInTheDocument();
  expect(screen.getByText('Appearance', { exact: true })).toBeInTheDocument();
  expect(studioConfirm).not.toHaveBeenCalled();
  expect(runtimeApi.reloadRuntime).not.toHaveBeenCalled();
  expect(refreshRuntime).not.toHaveBeenCalled();
});
