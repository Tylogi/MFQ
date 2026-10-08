import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import type { RuntimeStatus } from '../../shared/api/types';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { InferencePolicyPanel } from './InferencePolicyPanel';

const context = vi.hoisted(() => ({
  runtime: null as RuntimeStatus | null,
  ready: true,
  connectionRevision: 1,
  refreshRuntime: vi.fn(),
}));
const notify = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => context }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('../../stores/toastStore', () => ({ toast: notify }));

beforeEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
  context.runtime = { mtp_service_enabled: true };
  context.ready = true;
  context.connectionRevision = 1;
  context.refreshRuntime.mockResolvedValue(true);
});

it('saves the service switch remotely without reloading a model', async () => {
  const save = vi.spyOn(runtimeApi, 'configureInferencePolicy').mockResolvedValue({ mtp_enabled: false });
  const reload = vi.spyOn(runtimeApi, 'reloadRuntime');
  context.refreshRuntime.mockImplementation(async () => {
    context.runtime = { mtp_service_enabled: false };
    return true;
  });
  render(<InferencePolicyPanel />);
  const control = screen.getByRole('switch', { name: 'Service MTP' });
  expect(control).toHaveAttribute('aria-checked', 'true');
  fireEvent.click(control);
  await waitFor(() => expect(save).toHaveBeenCalledExactlyOnceWith(false));
  await waitFor(() => expect(control).toHaveAttribute('aria-checked', 'false'));
  expect(context.refreshRuntime).toHaveBeenCalledOnce();
  expect(reload).not.toHaveBeenCalled();
  expect(notify.success).toHaveBeenCalledOnce();
});

it('does not fake a changed switch when the server rejects the save', async () => {
  vi.spyOn(runtimeApi, 'configureInferencePolicy').mockRejectedValue(new Error('save failed'));
  render(<InferencePolicyPanel />);
  const control = screen.getByRole('switch', { name: 'Service MTP' });
  fireEvent.click(control);
  await waitFor(() => expect(notify.error).toHaveBeenCalledWith('save failed'));
  expect(control).toHaveAttribute('aria-checked', 'true');
  expect(notify.success).not.toHaveBeenCalled();
});

it('requires an actual server-reported value and follows later updates', () => {
  context.runtime = null;
  const view = render(<InferencePolicyPanel />);
  expect(screen.getByRole('switch', { name: 'Service MTP' })).toBeDisabled();
  context.runtime = { mtp_service_enabled: false };
  view.rerender(<InferencePolicyPanel />);
  expect(screen.getByRole('switch', { name: 'Service MTP' })).toHaveAttribute('aria-checked', 'false');
  context.runtime = { mtp_service_enabled: true };
  view.rerender(<InferencePolicyPanel />);
  expect(screen.getByRole('switch', { name: 'Service MTP' })).toHaveAttribute('aria-checked', 'true');
});

it('ignores a late save reply after switching servers', async () => {
  let finish!: (value: { mtp_enabled: boolean }) => void;
  vi.spyOn(runtimeApi, 'configureInferencePolicy').mockReturnValue(new Promise((resolve) => { finish = resolve; }));
  const view = render(<InferencePolicyPanel />);
  fireEvent.click(screen.getByRole('switch', { name: 'Service MTP' }));
  context.connectionRevision = 2;
  view.rerender(<InferencePolicyPanel />);
  finish({ mtp_enabled: false });
  await waitFor(() => expect(screen.getByRole('switch', { name: 'Service MTP' })).not.toBeDisabled());
  expect(context.refreshRuntime).not.toHaveBeenCalled();
  expect(notify.success).not.toHaveBeenCalled();
});
