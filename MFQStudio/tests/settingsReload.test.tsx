/** Verify settingsReload behavior and integration contracts. */
import { i18n } from '../src/i18n';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, expect, it, vi } from 'vitest';
import { useRuntime } from '../src/app/RuntimeProvider';
import { ModelContextSettings } from '../src/features/runtime/ModelContextSettings';
import { MemorySettingsPanel } from '../src/features/connections/MemorySettingsPanel';
import type { RuntimeInstance, RuntimeStatus } from '../src/shared/api/types';

vi.mock('../src/app/RuntimeProvider', () => ({ useRuntime: vi.fn() }));
vi.mock('../src/features/settings/SettingsProvider', () => ({ useSettings: () => ({
  contextSize: 32768, setContextSize: vi.fn(), t: i18n.getFixedT('en'),
}) }));

const reloadModelContext = vi.fn();
const instances = [
  { id: 'flash', model: 'Qwen3.8-Flash-S4-L', state: 'ready', context_size: 32768, context_capacity: 131072,
    memory: { resident_weight_bytes: 78.6 * 2 ** 30 } },
  { id: 'dense', model: 'Qwen3.8-27B-S4-M', state: 'ready', context_size: 16384, context_capacity: 262144,
    memory: { resident_weight_bytes: 17.7 * 2 ** 30 } },
] as RuntimeInstance[];
const state = (patch = {}) => ({
  runtime: { instance_id: 'dense', model: instances[1].model, mlx_active_bytes: 17.7 * 2 ** 30 },
  instances, reloadingInstances: {}, reloadModelContext, ...patch,
} as unknown as ReturnType<typeof useRuntime>);

beforeEach(() => {
  vi.clearAllMocks();
  reloadModelContext.mockResolvedValue({ max_context: 8192 });
  vi.mocked(useRuntime).mockReturnValue(state());
});

it('verifies settingsReload test behavior 1', () => {
  render(<ModelContextSettings />);
  expect(screen.getByRole('spinbutton', { name: 'Qwen3.8-Flash-S4-L maximum context' })).toHaveValue(32768);
  expect(screen.getByRole('spinbutton', { name: 'Qwen3.8-27B-S4-M maximum context' })).toHaveValue(16384);
  expect(screen.getByRole('spinbutton', { name: 'Qwen3.8-Flash-S4-L maximum context' })).toHaveAttribute('max', '131072');
  expect(screen.getByRole('spinbutton', { name: 'Qwen3.8-27B-S4-M maximum context' })).toHaveAttribute('max', '262144');
});

it('verifies settingsReload test behavior 2', async () => {
  const user = userEvent.setup();
  const view = render(<ModelContextSettings />);
  const flash = screen.getByRole('spinbutton', { name: 'Qwen3.8-Flash-S4-L maximum context' });
  fireEvent.change(flash, { target: { value: '8192' } });
  fireEvent.change(screen.getByRole('spinbutton', { name: 'Qwen3.8-27B-S4-M maximum context' }), { target: { value: '65536' } });
  await user.click(screen.getByRole('button', { name: 'Reload Qwen3.8-Flash-S4-L' }));
  expect(reloadModelContext).toHaveBeenCalledExactlyOnceWith('flash', 8192);
  vi.mocked(useRuntime).mockReturnValue(state({ runtime: { instance_id: 'flash', max_context: 8192 } }));
  view.rerender(<ModelContextSettings />);
  expect(screen.getByRole('spinbutton', { name: 'Qwen3.8-27B-S4-M maximum context' })).toHaveValue(65536);
});

it('verifies settingsReload test behavior 3', async () => {
  let complete!: (result: RuntimeStatus) => void;
  reloadModelContext.mockReturnValue(new Promise((resolve) => { complete = resolve; }));
  const view = render(<ModelContextSettings />);
  fireEvent.click(screen.getByRole('button', { name: 'Reload Qwen3.8-Flash-S4-L' }));
  vi.mocked(useRuntime).mockReturnValue(state({ reloadingInstances: { flash: 32768 } }));
  view.rerender(<ModelContextSettings />);
  expect(screen.getByRole('button', { name: 'Reload Qwen3.8-Flash-S4-L' })).toBeDisabled();
  expect(screen.getByText('Reloading…')).toBeVisible();
  expect(screen.getByRole('button', { name: 'Reload Qwen3.8-27B-S4-M' })).toBeEnabled();
  await act(async () => { complete({ max_context: 16384 }); });
  await waitFor(() => expect(screen.getByRole('spinbutton', { name: 'Qwen3.8-Flash-S4-L maximum context' })).toHaveValue(16384));
});

it.each(['', '0', '512.5', '131073'])('verifies settingsReload test behavior 4', (value) => {
  render(<ModelContextSettings />);
  fireEvent.change(screen.getByRole('spinbutton', { name: 'Qwen3.8-Flash-S4-L maximum context' }), { target: { value } });
  expect(screen.getByRole('button', { name: 'Reload Qwen3.8-Flash-S4-L' })).toBeDisabled();
  expect(reloadModelContext).not.toHaveBeenCalled();
});

it('verifies settingsReload test behavior 5', () => {
  render(<MemorySettingsPanel />);
  const row = screen.getByText(/Current weight residency/).closest('.memory-budget-actions')!;
  expect(row).toHaveTextContent('96.3');
  expect(row).not.toHaveTextContent('17.7');
});
