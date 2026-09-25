/** 验证设置页面确认重载时传递上下文和实例标识，取消时不请求。 */
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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

it('确认重载后传递 contextSize 和 instance_id，完成后刷新运行时', async () => {
  const user = userEvent.setup();
  let confirm!: (accepted: boolean) => void;
  let complete!: (result: Awaited<ReturnType<typeof runtimeApi.reloadRuntime>>) => void;
  vi.mocked(studioConfirm).mockReturnValue(new Promise((resolve) => { confirm = resolve; }));
  vi.mocked(runtimeApi.reloadRuntime).mockReturnValue(new Promise((resolve) => { complete = resolve; }));
  render(<MemoryRouter><SettingsRoute /></MemoryRouter>);
  const button = screen.getByRole('button', { name: 'Reload model with this context' });
  await user.click(button);
  expect(studioConfirm).toHaveBeenCalledOnce();
  expect(runtimeApi.reloadRuntime).not.toHaveBeenCalled();
  expect(refreshRuntime).not.toHaveBeenCalled();
  await act(async () => { confirm(true); });
  expect(runtimeApi.reloadRuntime).toHaveBeenCalledExactlyOnceWith(8192, 'chosen-instance');
  expect(button).toBeDisabled();
  expect(refreshRuntime).not.toHaveBeenCalled();
  await act(async () => { complete({} as Awaited<ReturnType<typeof runtimeApi.reloadRuntime>>); });
  expect(refreshRuntime).toHaveBeenCalledExactlyOnceWith(true);
  await waitFor(() => expect(button).toBeEnabled());
});

it('取消重载确认时不请求也不刷新运行时', async () => {
  vi.mocked(studioConfirm).mockResolvedValue(false);
  const user = userEvent.setup();
  render(<MemoryRouter><SettingsRoute /></MemoryRouter>);
  await user.click(screen.getByRole('button', { name: 'Reload model with this context' }));
  expect(studioConfirm).toHaveBeenCalledOnce();
  expect(runtimeApi.reloadRuntime).not.toHaveBeenCalled();
  expect(refreshRuntime).not.toHaveBeenCalled();
  expect(screen.getByRole('button', { name: 'Reload model with this context' })).toBeEnabled();
});
