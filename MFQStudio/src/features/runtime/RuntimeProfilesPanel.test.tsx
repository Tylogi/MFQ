import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { RuntimeProfilesPanel } from './RuntimeProfilesPanel';
import { DEFAULT_SETTINGS } from '../settings/configuration';
import { setApiBaseUrl } from '../../shared/api/client';
import type { ModelArtifact, RuntimeProfile } from '../../shared/api/types';

const api = vi.hoisted(() => ({ artifacts: vi.fn(), profiles: vi.fn(), create: vi.fn(), load: vi.fn(), remove: vi.fn(),
  confirm: vi.fn(), refresh: vi.fn(), select: vi.fn(), navigate: vi.fn(), success: vi.fn(), error: vi.fn() }));
const state = vi.hoisted(() => ({ ready: true, connectionRevision: 0 }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ ...state,
  runtime: { instance_id: 'a', model: 'model' }, instances: [{ id: 'a', model: 'model', context_size: 512 }],
  realtime: null, refreshRuntime: api.refresh, setSelectedModel: api.select }) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ settings: { ...DEFAULT_SETTINGS, inheritModelDefaults: false },
  contextSize: 2048, tr: (_zh: string, en: string) => en }) }));
vi.mock('../../shared/api/resources/runtime', () => ({ runtimeApi: { runtimeProfiles: api.profiles, createRuntimeProfile: api.create,
  loadRuntimeProfile: api.load, deleteRuntimeProfile: api.remove } }));
vi.mock('../../shared/api/resources/models', () => ({ modelsApi: { modelArtifacts: api.artifacts } }));
vi.mock('../../studio', () => ({ studioConfirm: api.confirm }));
vi.mock('react-router', () => ({ useNavigate: () => api.navigate }));
vi.mock('../../stores/toastStore', () => ({ toast: { success: api.success, error: api.error } }));

const profile = { id: 'profile-a', name: 'Saved profile', artifact_id: 'artifact-a', drifted: false,
  load: { model: 'model', context_size: 512, prefill_chunk_size: 2048 } } as RuntimeProfile;
beforeEach(() => {
  vi.clearAllMocks();
  setApiBaseUrl('');
  state.connectionRevision = 0;
  api.artifacts.mockResolvedValue([{ id: 'artifact-a', name: 'model' } as ModelArtifact]);
  api.profiles.mockResolvedValue([profile]);
  api.create.mockResolvedValue(profile);
  api.load.mockResolvedValue({ operation_id: 'load-a' });
  api.remove.mockResolvedValue(undefined);
  api.confirm.mockResolvedValue(true);
  api.refresh.mockResolvedValue(true);
});
afterEach(() => setApiBaseUrl(''));

it('saves the loaded model context instead of the next-load global default', async () => {
  render(<RuntimeProfilesPanel />);
  await screen.findByText('Saved profile');
  fireEvent.change(screen.getByPlaceholderText('Current configuration name'), { target: { value: 'My profile' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save current' }));
  await waitFor(() => expect(api.create).toHaveBeenCalledOnce());
  expect(api.create.mock.calls[0][0]).toMatchObject({ name: 'My profile', load: { model: 'model', context_size: 512 } });
});

it('does not load a drifted profile after an old confirmation outlives its service', async () => {
  let confirm!: (accepted: boolean) => void;
  api.profiles.mockResolvedValue([{ ...profile, drifted: true }]);
  api.confirm.mockReturnValueOnce(new Promise((done) => { confirm = done; }));
  render(<RuntimeProfilesPanel />);
  await screen.findByText('Saved profile');
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  setApiBaseUrl('https://second.invalid');
  await act(async () => { confirm(true); });
  expect(api.load).not.toHaveBeenCalled();
  expect(api.navigate).not.toHaveBeenCalled();
});

it('locks profile actions while a drift confirmation is pending', async () => {
  let confirm!: (accepted: boolean) => void;
  api.profiles.mockResolvedValue([{ ...profile, drifted: true }]);
  api.confirm.mockReturnValueOnce(new Promise((done) => { confirm = done; }));
  render(<RuntimeProfilesPanel />);
  await screen.findByText('Saved profile');
  const load = screen.getByRole('button', { name: 'Load' });
  fireEvent.click(load);
  expect(load).toBeDisabled();
  fireEvent.click(load);
  expect(api.confirm).toHaveBeenCalledOnce();
  await act(async () => { confirm(false); });
  expect(load).toBeEnabled();
  expect(api.load).not.toHaveBeenCalled();
});

it('does not navigate after an accepted load response arrives off-page', async () => {
  let resolve!: (value: { operation_id: string }) => void;
  api.load.mockReturnValueOnce(new Promise((done) => { resolve = done; }));
  const view = render(<RuntimeProfilesPanel />);
  await screen.findByText('Saved profile');
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(api.load).toHaveBeenCalledOnce());
  view.unmount();
  await act(async () => { resolve({ operation_id: 'old-load' }); });
  expect(api.refresh).not.toHaveBeenCalled();
  expect(api.select).not.toHaveBeenCalled();
  expect(api.navigate).not.toHaveBeenCalled();
});

it('finishes refresh and selection before navigating away on a valid load', async () => {
  let resolve!: (value: boolean) => void;
  api.refresh.mockReturnValueOnce(new Promise((done) => { resolve = done; }));
  render(<RuntimeProfilesPanel />);
  await screen.findByText('Saved profile');
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(api.refresh).toHaveBeenCalledOnce());
  expect(api.navigate).not.toHaveBeenCalled();
  await act(async () => { resolve(true); });
  expect(api.select).toHaveBeenCalledExactlyOnceWith('model');
  expect(api.navigate).toHaveBeenCalledOnce();
});

it('does not refresh another service after a pending profile save', async () => {
  let resolve!: (value: RuntimeProfile) => void;
  api.create.mockReturnValueOnce(new Promise((done) => { resolve = done; }));
  const view = render(<RuntimeProfilesPanel />);
  await screen.findByText('Saved profile');
  fireEvent.change(screen.getByPlaceholderText('Current configuration name'), { target: { value: 'My profile' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save current' }));
  await waitFor(() => expect(api.create).toHaveBeenCalledOnce());
  view.unmount();
  await act(async () => { resolve(profile); });
  expect(api.profiles).toHaveBeenCalledOnce();
  expect(api.refresh).not.toHaveBeenCalled();
  expect(api.success).not.toHaveBeenCalled();
});

it('does not show old deletion errors after leaving the panel', async () => {
  let reject!: (cause: Error) => void;
  api.remove.mockReturnValueOnce(new Promise((_, fail) => { reject = fail; }));
  const view = render(<RuntimeProfilesPanel />);
  await screen.findByText('Saved profile');
  fireEvent.click(screen.getByRole('button', { name: 'Delete profile' }));
  view.unmount();
  await act(async () => { reject(new Error('Old service failed')); });
  expect(api.error).not.toHaveBeenCalled();
});
