import { act, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { StudioUpdateStatus } from '../../shared/platform/studio';
import { UpdateManager, useStudioUpdates } from './UpdateManager';

const tr = (_zh: string, en: string) => en;
const release = { version: '0.3.3', tag: 'v0.3.3', name: 'Release title', notes: '# Improvements\n\n- Faster inference\n<script>alert(1)</script><img src=x onerror=alert(1)>',
  published_at: '2026-10-10T00:00:00Z', page_url: 'https://github.com/Tylogi/TyloQuant/releases/tag/v0.3.3', prerelease: false,
  asset: { name: 'MFQ.dmg', byte_size: 128, download_url: 'https://github.com/Tylogi/TyloQuant/releases/download/v0.3.3/MFQ.dmg', sha256: 'a'.repeat(64) } };
const status: StudioUpdateStatus = { current_version: '0.3.2', current_release: true, automatic_check: true, automatic_download: false,
  update_available: true, platform_supported: true, releases_page: 'https://github.com/Tylogi/TyloQuant/releases', latest: release, releases: [release],
  installed_versions: [{ version: '0.3.2', tag: 'v0.3.2', current: true, ready: true, byte_size: 0, installed_at_epoch_seconds: 0 }] };

beforeEach(() => { localStorage.clear(); });
afterEach(() => { delete window.__TAURI_INTERNALS__; });
function bridge(snapshot = status) {
  const invoke = vi.fn(async (command: string) => command === 'studio_update_status' ? snapshot : command === 'studio_confirm' ? false : null);
  window.__TAURI_INTERNALS__ = { invoke: invoke as NonNullable<Window['__TAURI_INTERNALS__']>['invoke'] };
  return invoke;
}
const actions = () => ({ busy: null, progress: null, download: vi.fn(), install: vi.fn(), refresh: vi.fn(), remove: vi.fn(), setAutomatic: vi.fn(), setAutomaticDownload: vi.fn() });

it.each(['zh-CN', 'en'])('localizes Release labels without changing English (%s)', language => {
  bridge();
  const en = language === 'en';
  const translate = (zh: string, english: string) => en ? english : zh;
  const snapshot = { ...status, releases: [{ ...release, name: 'MFQ Studio 0.3.3', notes: 'Version notes' }] };
  const { container, rerender } = render(<UpdateManager {...actions()} status={snapshot} tr={translate} />);
  expect(screen.getByRole('heading', { name: en ? /^Release$/ : /^正式版$/ })).toBeInTheDocument();
  expect(container.querySelector('.update-current-row small')).toHaveTextContent(en ? 'Release' : '正式版');
  expect(container.querySelector('.local-version.current small')).toHaveTextContent(en ? 'Running · Release' : '正在运行 · 正式版');
  expect(screen.getByRole('switch', { name: en ? 'Automatically download Release updates' : '自动下载正式版更新' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: en ? 'View v0.3.3 Release' : '查看 v0.3.3 正式版' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: en ? 'View all Releases' : '查看全部正式版' })).toBeInTheDocument();
  if (!en) expect(container.textContent).not.toContain('Release');
  rerender(<UpdateManager {...actions()} status={{ ...snapshot, current_release: false, releases: [] }} tr={translate} />);
  expect(screen.getByText(en ? 'Development builds do not auto-download updates; choose a Release manually.' : '开发构建不参与自动下载；可手动选择正式版。')).toBeInTheDocument();
  expect(screen.getByText(en ? 'No Release information yet. Check for updates when online.' : '暂无正式版信息，联网后检查更新。')).toBeInTheDocument();
  if (!en) expect(container.textContent).not.toContain('Release');
});

it('shows Release notes without executable markup, and filters experimental entries', () => {
  bridge();
  const callbacks = actions();
  const { container } = render(<UpdateManager {...callbacks} status={{ ...status, releases: [release, { ...release, tag: 'v0.4.0rc1', version: '0.4.0', prerelease: true }] }} tr={tr} />);
  expect(screen.getByRole('heading', { name: 'Improvements' })).toBeInTheDocument();
  expect(screen.getByText('Faster inference')).toBeInTheDocument();
  expect(screen.queryByText('v0.4.0')).not.toBeInTheDocument();
  expect(container.querySelectorAll('script,img')).toHaveLength(0);
  fireEvent.click(screen.getByRole('button', { name: 'Download' }));
  expect(callbacks.download).toHaveBeenCalledWith('v0.3.3');
});

it('browsers can read notes without offering installation or downloads', () => {
  render(<UpdateManager {...actions()} status={{ ...status, platform_supported: false }} tr={tr} />);
  expect(screen.getByText('Faster inference')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Download' })).not.toBeInTheDocument();
  expect(screen.queryByRole('switch', { name: 'Automatically download Release updates' })).not.toBeInTheDocument();
});

it('auto-downloads a newer Release once but never silently installs or restarts', async () => {
  const invoke = bridge({ ...status, automatic_download: true });
  const onError = vi.fn();
  renderHook(() => useStudioUpdates(onError));
  await waitFor(() => expect(invoke.mock.calls.filter(([command]) => command === 'studio_update_download')).toHaveLength(1));
  await waitFor(() => expect(invoke.mock.calls.filter(([command]) => command === 'studio_update_status')).toHaveLength(2));
  expect(invoke.mock.calls.some(([command]) => command === 'studio_update_install')).toBe(false);
});

it('development builds never auto-download even when the preference was saved', async () => {
  const invoke = bridge({ ...status, current_version: '0.3.2+dev.12345678.modified', current_release: false, automatic_download: true });
  const onError = vi.fn();
  const { result } = renderHook(() => useStudioUpdates(onError));
  await waitFor(() => expect(result.current.status).not.toBeNull());
  expect(invoke.mock.calls.some(([command]) => command === 'studio_update_download')).toBe(false);
});

it('does not download an already-cached update', async () => {
  const invoke = bridge({ ...status, automatic_download: true, installed_versions: [...status.installed_versions,
    { version: '0.3.3', tag: 'v0.3.3', current: false, ready: true, byte_size: 128, installed_at_epoch_seconds: 0 }] });
  const onError = vi.fn();
  const { result } = renderHook(() => useStudioUpdates(onError));
  await waitFor(() => expect(result.current.status).not.toBeNull());
  expect(invoke.mock.calls.some(([command]) => command === 'studio_update_download')).toBe(false);
});

it('rechecks immediately when the connection comes back online', async () => {
  const invoke = bridge();
  const onError = vi.fn();
  const { result } = renderHook(() => useStudioUpdates(onError));
  await waitFor(() => expect(result.current.status).not.toBeNull());
  act(() => window.dispatchEvent(new Event('online')));
  await waitFor(() => expect(invoke).toHaveBeenLastCalledWith('studio_update_status', { force: true }));
});

it('declining installation never switches a version', async () => {
  const invoke = bridge();
  const onError = vi.fn();
  const { result } = renderHook(() => useStudioUpdates(onError));
  await waitFor(() => expect(result.current.status).not.toBeNull());
  await act(() => result.current.install('0.3.3', tr));
  expect(invoke.mock.calls.some(([command]) => command === 'studio_update_install')).toBe(false);
});

it('disabled automatic checks do not retry on online events', async () => {
  const invoke = bridge({ ...status, automatic_check: false });
  const onError = vi.fn();
  const { result } = renderHook(() => useStudioUpdates(onError));
  await waitFor(() => expect(result.current.status).not.toBeNull());
  act(() => window.dispatchEvent(new Event('online')));
  expect(invoke).toHaveBeenCalledOnce();
});
