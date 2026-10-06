/** Verify browser service settings preserve the actual service origin and report listener failures. */
import { i18n } from '../../i18n';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { ConnectionsPage } from './ConnectionsPage';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { browserServiceUrl, setApiBaseUrl } from '../../shared/api/client';

const { reloadService } = vi.hoisted(() => ({ reloadService: vi.fn() }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({
  runtime: { model: 'model' }, models: [], instances: [], studio: null,
  selectedModel: '', setSelectedModel: vi.fn(), reloadService,
}) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ t: i18n.getFixedT('en') }) }));
vi.mock('./MemorySettingsPanel', () => ({ MemorySettingsPanel: () => null }));
vi.mock('./InferenceDefaultsPanel', () => ({ InferenceDefaultsPanel: () => null }));
vi.mock('./ToolsRoutingPanel', () => ({ ToolsRoutingPanel: () => null }));
vi.mock('../runtime/RuntimeProfilesPanel', () => ({ RuntimeProfilesPanel: () => <h2>Runtime profiles</h2> }));
vi.mock('../../studio', () => ({ isStudio: () => false, studioCredential: async () => '' }));

beforeEach(() => {
  localStorage.clear();
  setApiBaseUrl('');
  reloadService.mockResolvedValue(true);
  vi.spyOn(runtimeApi, 'runtimeListener').mockResolvedValue({ host: '127.0.0.1', port: 8090, configurable: true });
});
afterEach(() => { vi.restoreAllMocks(); localStorage.clear(); setApiBaseUrl(''); });

it('verifies ConnectionsPage test behavior 1', async () => {
  const change = vi.spyOn(runtimeApi, 'configureRuntimeListener').mockResolvedValue({ host: '127.0.0.1', port: 8091, configurable: true });
  render(<MemoryRouter><ConnectionsPage /></MemoryRouter>);
  expect(screen.getByRole('heading', { name: /^Service$/ })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Runtime profiles' })).toBeInTheDocument();
  const port = screen.getByRole('spinbutton', { name: 'Port' });
  await waitFor(() => expect(runtimeApi.runtimeListener).toHaveBeenCalled());
  fireEvent.change(port, { target: { value: '8091' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save server settings' }));
  await waitFor(() => expect(change).toHaveBeenCalledExactlyOnceWith(8091));
  expect(browserServiceUrl()).toBe('http://localhost:8091');
  expect(reloadService).toHaveBeenCalledOnce();
});

it('verifies ConnectionsPage test behavior 2', async () => {
  vi.spyOn(runtimeApi, 'configureRuntimeListener').mockRejectedValue(new Error('port in use'));
  render(<MemoryRouter><ConnectionsPage /></MemoryRouter>);
  await waitFor(() => expect(runtimeApi.runtimeListener).toHaveBeenCalled());
  fireEvent.change(screen.getByRole('spinbutton', { name: 'Port' }), { target: { value: '8091' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save server settings' }));
  await waitFor(() => expect(runtimeApi.configureRuntimeListener).toHaveBeenCalled());
  expect(browserServiceUrl()).toBe('');
  expect(reloadService).not.toHaveBeenCalled();
});

it('keeps a same-origin remote service in remote mode when saving', async () => {
  vi.stubGlobal('window', new Proxy(window, {
    get(target, key) {
      if (key === 'location') return new URL('https://studio.example:9443/chat');
      const value = Reflect.get(target, key);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  }));
  try {
    const change = vi.spyOn(runtimeApi, 'configureRuntimeListener');
    render(<MemoryRouter><ConnectionsPage /></MemoryRouter>);
    await waitFor(() => expect(screen.getByRole('combobox', { name: 'Bind address' })).toHaveValue('remote'));
    expect(screen.getByRole('textbox', { name: 'Remote endpoint' })).toHaveValue('https://studio.example:9443');
    fireEvent.click(screen.getByRole('button', { name: 'Save server settings' }));
    await waitFor(() => expect(reloadService).toHaveBeenCalledOnce());
    expect(browserServiceUrl()).toBe('https://studio.example:9443');
    expect(change).not.toHaveBeenCalled();
  } finally {
    vi.unstubAllGlobals();
  }
});
