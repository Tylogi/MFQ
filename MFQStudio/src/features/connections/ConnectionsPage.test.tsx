import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { ConnectionsPage } from './ConnectionsPage';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { browserServiceUrl, setApiBaseUrl } from '../../shared/api/client';
import { setBrowserServiceUrl } from '../../shared/api/client';

const { reloadService } = vi.hoisted(() => ({ reloadService: vi.fn() }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({
  runtime: { model: 'model' }, models: [], instances: [], studio: null,
  selectedModel: '', setSelectedModel: vi.fn(), reloadService,
}) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('./MemorySettingsPanel', () => ({ MemorySettingsPanel: () => null }));
vi.mock('./InferenceDefaultsPanel', () => ({ InferenceDefaultsPanel: () => null }));
vi.mock('./ToolsRoutingPanel', () => ({ ToolsRoutingPanel: () => null }));
vi.mock('../runtime/RuntimeProfilesPanel', () => ({ RuntimeProfilesPanel: () => <h2>Runtime profiles</h2> }));
vi.mock('../../studio', () => ({ isStudio: () => false, studioCredential: async () => '' }));

beforeEach(() => {
  localStorage.clear();
  setBrowserServiceUrl('');
  setApiBaseUrl('');
  reloadService.mockResolvedValue(true);
  vi.spyOn(runtimeApi, 'runtimeListener').mockResolvedValue({ host: '127.0.0.1', port: 8090, configurable: true });
});
afterEach(() => { vi.restoreAllMocks(); localStorage.clear(); setApiBaseUrl(''); });

it('网页端可编辑端口，保存实际更改监听并持久化连接地址', async () => {
  const change = vi.spyOn(runtimeApi, 'configureRuntimeListener').mockResolvedValue({ host: '127.0.0.1', port: 8091, configurable: true });
  render(<MemoryRouter><ConnectionsPage /></MemoryRouter>);
  expect(screen.getByRole('heading', { name: /^Service$/ })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Runtime profiles' })).toBeInTheDocument();
  const port = screen.getByRole('spinbutton', { name: 'Port' });
  await waitFor(() => expect(runtimeApi.runtimeListener).toHaveBeenCalled());
  fireEvent.change(port, { target: { value: '8091' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save server settings' }));
  await waitFor(() => expect(change).toHaveBeenCalledExactlyOnceWith(8091));
  expect(browserServiceUrl()).toBe('http://127.0.0.1:8091');
  expect(reloadService).toHaveBeenCalledOnce();
});

it.each([true, false])('unlocks saving after following the newly accepted port (reachable: %s)', async (reachable) => {
  vi.spyOn(runtimeApi, 'configureRuntimeListener').mockResolvedValue({ host: '127.0.0.1', port: 8091, configurable: true });
  reloadService.mockImplementation(async () => {
    setApiBaseUrl(browserServiceUrl());
    return reachable;
  });
  render(<MemoryRouter><ConnectionsPage /></MemoryRouter>);
  await waitFor(() => expect(runtimeApi.runtimeListener).toHaveBeenCalled());
  fireEvent.change(screen.getByRole('spinbutton', { name: 'Port' }), { target: { value: '8091' } });
  const save = screen.getByRole('button', { name: 'Save server settings' });
  fireEvent.click(save);
  await waitFor(() => expect(reloadService).toHaveBeenCalledOnce());
  await waitFor(() => expect(save).toBeEnabled());
  expect(screen.getByRole('spinbutton', { name: 'Port' })).toBeEnabled();
});

it('端口占用时保留原连接，不假报保存成功', async () => {
  vi.spyOn(runtimeApi, 'configureRuntimeListener').mockRejectedValue(new Error('port in use'));
  render(<MemoryRouter><ConnectionsPage /></MemoryRouter>);
  await waitFor(() => expect(runtimeApi.runtimeListener).toHaveBeenCalled());
  fireEvent.change(screen.getByRole('spinbutton', { name: 'Port' }), { target: { value: '8091' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save server settings' }));
  await waitFor(() => expect(runtimeApi.configureRuntimeListener).toHaveBeenCalled());
  expect(browserServiceUrl()).toBe('');
  expect(reloadService).not.toHaveBeenCalled();
});

it('does not overwrite another service connection while a listener change is pending', async () => {
  let resolve!: (value: Awaited<ReturnType<typeof runtimeApi.configureRuntimeListener>>) => void;
  vi.spyOn(runtimeApi, 'configureRuntimeListener').mockReturnValue(new Promise((done) => { resolve = done; }));
  const view = render(<MemoryRouter><ConnectionsPage /></MemoryRouter>);
  await waitFor(() => expect(runtimeApi.runtimeListener).toHaveBeenCalled());
  fireEvent.change(screen.getByRole('spinbutton', { name: 'Port' }), { target: { value: '8091' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save server settings' }));
  await waitFor(() => expect(runtimeApi.configureRuntimeListener).toHaveBeenCalled());
  view.unmount();
  setApiBaseUrl('https://second.invalid');
  resolve({ host: '127.0.0.1', port: 8091, configurable: true });
  await waitFor(() => expect(browserServiceUrl()).toBe(''));
  expect(reloadService).not.toHaveBeenCalled();
});

it('finishes following an accepted port change even when navigating away', async () => {
  let resolve!: (value: Awaited<ReturnType<typeof runtimeApi.configureRuntimeListener>>) => void;
  vi.spyOn(runtimeApi, 'configureRuntimeListener').mockReturnValue(new Promise((done) => { resolve = done; }));
  const view = render(<MemoryRouter><ConnectionsPage /></MemoryRouter>);
  await waitFor(() => expect(runtimeApi.runtimeListener).toHaveBeenCalled());
  fireEvent.change(screen.getByRole('spinbutton', { name: 'Port' }), { target: { value: '8091' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save server settings' }));
  await waitFor(() => expect(runtimeApi.configureRuntimeListener).toHaveBeenCalled());
  view.unmount();
  resolve({ host: '127.0.0.1', port: 8091, configurable: true });
  await waitFor(() => expect(browserServiceUrl()).toBe('http://127.0.0.1:8091'));
  expect(reloadService).toHaveBeenCalledOnce();
});
