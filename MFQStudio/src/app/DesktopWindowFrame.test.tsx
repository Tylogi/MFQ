import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useEffect } from 'react';
import { beforeEach, expect, it, vi } from 'vitest';
import { SettingsProvider, useSettings } from '../features/settings/SettingsProvider';
import { DesktopWindowFrame } from './DesktopWindowFrame';

const native = vi.hoisted(() => ({
  custom: vi.fn(), isMaximized: vi.fn(), isFocused: vi.fn(),
  minimize: vi.fn(), toggleMaximize: vi.fn(), close: vi.fn(), setTheme: vi.fn(),
  onResized: vi.fn(), onFocusChanged: vi.fn(),
}));
vi.mock('../shared/platform/studio', () => ({ studioCustomTitlebar: native.custom }));
vi.mock('@tauri-apps/api/window', () => ({ getCurrentWindow: () => native }));

beforeEach(() => {
  vi.clearAllMocks();
  native.custom.mockResolvedValue(true);
  native.isMaximized.mockResolvedValue(false);
  native.isFocused.mockResolvedValue(true);
  native.minimize.mockResolvedValue(undefined);
  native.toggleMaximize.mockResolvedValue(undefined);
  native.close.mockResolvedValue(undefined);
  native.setTheme.mockResolvedValue(undefined);
  native.onResized.mockResolvedValue(vi.fn());
  native.onFocusChanged.mockResolvedValue(vi.fn());
});

function mount() {
  return render(<SettingsProvider><DesktopWindowFrame><main>Workspace</main></DesktopWindowFrame></SettingsProvider>);
}

it('keeps browser and other desktop platforms free of custom controls', async () => {
  native.custom.mockResolvedValue(false);
  mount();
  await waitFor(() => expect(native.custom).toHaveBeenCalled());
  expect(screen.getByRole('main')).toHaveTextContent('Workspace');
  expect(screen.queryByRole('banner')).not.toBeInTheDocument();
  expect(native.onResized).not.toHaveBeenCalled();
});

it('minimizes, maximizes, restores and closes through the native window', async () => {
  const user = userEvent.setup();
  mount();
  await user.click(await screen.findByRole('button', { name: 'Minimize' }));
  expect(native.minimize).toHaveBeenCalledOnce();
  native.isMaximized.mockResolvedValue(true);
  await user.click(screen.getByRole('button', { name: 'Maximize' }));
  expect(native.toggleMaximize).toHaveBeenCalledOnce();
  await screen.findByRole('button', { name: 'Restore window' });
  native.isMaximized.mockResolvedValue(false);
  await user.click(screen.getByRole('button', { name: 'Restore window' }));
  await screen.findByRole('button', { name: 'Maximize' });
  await user.click(screen.getByRole('button', { name: 'Close window' }));
  expect(native.close).toHaveBeenCalledOnce();
});

it('does not restart the workspace when native window detection completes', async () => {
  let resolve: (value: boolean) => void = () => {};
  native.custom.mockImplementation(() => new Promise(done => { resolve = done; }));
  const started = vi.fn();
  function Workspace() { useEffect(started, []); return <main>Workspace</main>; }
  render(<SettingsProvider><DesktopWindowFrame><Workspace /></DesktopWindowFrame></SettingsProvider>);
  expect(started).toHaveBeenCalledOnce();
  await act(async () => { resolve(true); });
  await screen.findByRole('button', { name: 'Minimize' });
  expect(started).toHaveBeenCalledOnce();
});

it('tracks native maximize and focus changes and releases its listeners', async () => {
  const stopResize = vi.fn(), stopFocus = vi.fn();
  native.onResized.mockResolvedValue(stopResize);
  native.onFocusChanged.mockResolvedValue(stopFocus);
  const view = mount();
  await waitFor(() => expect(native.onFocusChanged).toHaveBeenCalledOnce());
  native.isMaximized.mockResolvedValue(true);
  await act(async () => { native.onResized.mock.calls[0][0](); });
  await screen.findByRole('button', { name: 'Restore window' });
  act(() => { native.onFocusChanged.mock.calls[0][0]({ payload: false }); });
  expect(view.container.querySelector('.desktop-window')).toHaveAttribute('data-window-focused', 'false');
  view.unmount();
  expect(stopResize).toHaveBeenCalledOnce();
  expect(stopFocus).toHaveBeenCalledOnce();
});

it('synchronizes the native window theme with the application preference', async () => {
  function Preference() {
    const { updateSettings } = useSettings();
    return <button onClick={() => updateSettings({ theme: 'dark' })}>Dark theme</button>;
  }
  render(<SettingsProvider><DesktopWindowFrame><Preference /></DesktopWindowFrame></SettingsProvider>);
  await waitFor(() => expect(native.setTheme).toHaveBeenCalledWith(null));
  await userEvent.click(screen.getByRole('button', { name: 'Dark theme' }));
  await waitFor(() => expect(native.setTheme).toHaveBeenLastCalledWith('dark'));
});

it('releases listeners that finish registering after the frame unmounts', async () => {
  let register: (stop: () => void) => void = () => {};
  native.onResized.mockImplementation(() => new Promise(resolve => { register = resolve; }));
  const view = mount();
  await waitFor(() => expect(native.onResized).toHaveBeenCalledOnce());
  view.unmount();
  const stop = vi.fn();
  await act(async () => { register(stop); });
  expect(stop).toHaveBeenCalledOnce();
});
