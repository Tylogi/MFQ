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
vi.mock('./MemorySettingsPanel', () => ({ MemorySettingsPanel: () => <div className="section-label">Memory plan</div> }));
vi.mock('./ContextManagementPanel', () => ({ ContextManagementPanel: () => <div className="section-label">Context management</div> }));
vi.mock('./PrefixCacheSettingsPanel', () => ({ PrefixCacheSettingsPanel: () => <div className="section-label">Persistent prefix cache</div> }));
vi.mock('./InferenceDefaultsPanel', () => ({ InferenceDefaultsPanel: () => null }));
vi.mock('./RemoteRoutingPanel', () => ({ RemoteRoutingPanel: () => null }));
vi.mock('../runtime/RuntimeProfilesPanel', () => ({ RuntimeProfilesPanel: () => <h2>Runtime profiles</h2> }));
vi.mock('../../studio', () => ({ isStudio: () => false, studioCredential: async () => '' }));

beforeEach(() => {
  localStorage.clear();
  setBrowserServiceUrl('');
  setApiBaseUrl('');
  reloadService.mockResolvedValue(true);
  vi.spyOn(runtimeApi, 'runtimeListener').mockResolvedValue({ host: '127.0.0.1', port: 8090, anthropic_port: 8091, configurable: true });
});
afterEach(() => { vi.restoreAllMocks(); localStorage.clear(); setApiBaseUrl(''); });

it('keeps API, memory, context and persistent cache in independent ordered sections', () => {
  const { container } = render(<MemoryRouter><ConnectionsPage /></MemoryRouter>);
  expect([...container.querySelectorAll('.server-page > .section-label')].map(item => item.textContent)).toEqual([
    'API parameters', 'Memory plan', 'Context management', 'Persistent prefix cache', 'Automation',
  ]);
  expect(container.querySelector('.server-api-panel')).toContainElement(screen.getByRole('spinbutton', { name: 'OpenAI port' }));
  expect(container.querySelector('.server-api-panel')).toContainElement(screen.getByRole('spinbutton', { name: 'Anthropic port' }));
  expect(screen.queryByRole('button', { name: 'Save server settings' })).not.toBeInTheDocument();
});

it('网页端可编辑端口，保存实际更改监听并持久化连接地址', async () => {
  const change = vi.spyOn(runtimeApi, 'configureRuntimeListener').mockResolvedValue({ host: '127.0.0.1', port: 8091, configurable: true });
  render(<MemoryRouter><ConnectionsPage /></MemoryRouter>);
  expect(screen.getByRole('heading', { name: /^Service$/ })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Runtime profiles' })).toBeInTheDocument();
  const port = screen.getByRole('spinbutton', { name: 'OpenAI port' });
  await waitFor(() => expect(runtimeApi.runtimeListener).toHaveBeenCalled());
  fireEvent.change(port, { target: { value: '8091' } });
  fireEvent.blur(port);
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
  const port = screen.getByRole('spinbutton', { name: 'OpenAI port' });
  fireEvent.change(port, { target: { value: '8091' } });
  fireEvent.keyDown(port, { key: 'Enter' });
  await waitFor(() => expect(reloadService).toHaveBeenCalledOnce());
  await waitFor(() => expect(port).toBeEnabled());
});

it('端口占用时保留原连接，不假报保存成功', async () => {
  vi.spyOn(runtimeApi, 'configureRuntimeListener').mockRejectedValue(new Error('port in use'));
  render(<MemoryRouter><ConnectionsPage /></MemoryRouter>);
  await waitFor(() => expect(runtimeApi.runtimeListener).toHaveBeenCalled());
  const port = screen.getByRole('spinbutton', { name: 'OpenAI port' });
  fireEvent.change(port, { target: { value: '8091' } });
  fireEvent.blur(port);
  await waitFor(() => expect(runtimeApi.configureRuntimeListener).toHaveBeenCalled());
  expect(browserServiceUrl()).toBe('');
  expect(reloadService).not.toHaveBeenCalled();
  expect(await screen.findByRole('alert')).toHaveTextContent('port in use');
});

it('does not overwrite another service connection while a listener change is pending', async () => {
  let resolve!: (value: Awaited<ReturnType<typeof runtimeApi.configureRuntimeListener>>) => void;
  vi.spyOn(runtimeApi, 'configureRuntimeListener').mockReturnValue(new Promise((done) => { resolve = done; }));
  const view = render(<MemoryRouter><ConnectionsPage /></MemoryRouter>);
  await waitFor(() => expect(runtimeApi.runtimeListener).toHaveBeenCalled());
  const port = screen.getByRole('spinbutton', { name: 'OpenAI port' });
  fireEvent.change(port, { target: { value: '8091' } });
  fireEvent.blur(port);
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
  const port = screen.getByRole('spinbutton', { name: 'OpenAI port' });
  fireEvent.change(port, { target: { value: '8091' } });
  fireEvent.blur(port);
  await waitFor(() => expect(runtimeApi.configureRuntimeListener).toHaveBeenCalled());
  view.unmount();
  resolve({ host: '127.0.0.1', port: 8091, configurable: true });
  await waitFor(() => expect(browserServiceUrl()).toBe('http://127.0.0.1:8091'));
  expect(reloadService).toHaveBeenCalledOnce();
});

it('applies the Anthropic port independently without changing the OpenAI connection', async () => {
  const change = vi.spyOn(runtimeApi, 'configureRuntimeListener').mockResolvedValue({
    host: '127.0.0.1', port: 8090, anthropic_port: 8092, configurable: true,
  });
  render(<MemoryRouter><ConnectionsPage /></MemoryRouter>);
  const port = screen.getByRole('spinbutton', { name: 'Anthropic port' });
  await waitFor(() => expect(port).toHaveValue(8091));
  fireEvent.change(port, { target: { value: '8092' } });
  expect(change).not.toHaveBeenCalled();
  fireEvent.keyDown(port, { key: 'Enter' });
  fireEvent.blur(port);
  await waitFor(() => expect(change).toHaveBeenCalledExactlyOnceWith(8092, 'anthropic'));
  expect(browserServiceUrl()).toBe('');
  expect(reloadService).not.toHaveBeenCalled();
  expect(await screen.findByRole('status')).toHaveTextContent('Applied without reloading models');
  expect(screen.getByText('http://127.0.0.1:8092/v1/messages')).toBeInTheDocument();
});

it('initial render and unchanged blur never write settings', async () => {
  const change = vi.spyOn(runtimeApi, 'configureRuntimeListener');
  render(<MemoryRouter><ConnectionsPage /></MemoryRouter>);
  const port = screen.getByRole('spinbutton', { name: 'Anthropic port' });
  await waitFor(() => expect(port).toHaveValue(8091));
  fireEvent.blur(port);
  fireEvent.blur(screen.getByRole('spinbutton', { name: 'OpenAI port' }));
  expect(change).not.toHaveBeenCalled();
  expect(reloadService).not.toHaveBeenCalled();
});

it.each(['OpenAI port', 'Anthropic port'])('rejects invalid %s without sending a write', async (name) => {
  const change = vi.spyOn(runtimeApi, 'configureRuntimeListener');
  render(<MemoryRouter><ConnectionsPage /></MemoryRouter>);
  await waitFor(() => expect(screen.getByRole('spinbutton', { name: 'Anthropic port' })).toHaveValue(8091));
  const port = screen.getByRole('spinbutton', { name });
  fireEvent.change(port, { target: { value: '65536' } });
  fireEvent.blur(port);
  expect(await screen.findByRole('alert')).toHaveTextContent('Port must be an integer between 1 and 65535');
  expect(change).not.toHaveBeenCalled();
});
