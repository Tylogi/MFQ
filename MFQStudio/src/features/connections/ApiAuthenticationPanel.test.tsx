import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { ApiAuthenticationPanel } from './ApiAuthenticationPanel';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { getApiToken, setApiBaseUrl, setApiToken } from '../../shared/api/client';
import { saveStudioCredential } from '../../studio';

vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ connectionRevision: 0 }) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('../../studio', () => ({ saveStudioCredential: vi.fn().mockResolvedValue(undefined) }));

beforeEach(() => {
  vi.restoreAllMocks();
  vi.mocked(saveStudioCredential).mockReset().mockResolvedValue(undefined);
  setApiBaseUrl(''); setApiToken('');
  vi.spyOn(runtimeApi, 'authenticationPolicy').mockResolvedValue({ enabled: false, has_key: false });
  vi.spyOn(runtimeApi, 'configureAuthenticationPolicy').mockResolvedValue({ enabled: true, has_key: true });
});

async function mount() {
  render(<ApiAuthenticationPanel />);
  await waitFor(() => expect(screen.getByLabelText('Authentication')).toBeEnabled());
}

it('defaults to no authentication without writing server settings', async () => {
  await mount();
  expect(screen.getByLabelText('Authentication')).toHaveValue('none');
  expect(screen.queryByLabelText('API Key')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Apply authentication' })).toBeDisabled();
  expect(runtimeApi.configureAuthenticationPolicy).not.toHaveBeenCalled();
});

it('rejects enabling without a key', async () => {
  await mount();
  fireEvent.change(screen.getByLabelText('Authentication'), { target: { value: 'api-key' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply authentication' }));
  expect(screen.getByRole('alert')).toHaveTextContent('Enter an API key');
  expect(screen.getByLabelText('Authentication')).toHaveValue('none');
  expect(runtimeApi.configureAuthenticationPolicy).not.toHaveBeenCalled();
});

it('enables authentication and follows the accepted credential without double submission', async () => {
  await mount();
  fireEvent.change(screen.getByLabelText('Authentication'), { target: { value: 'api-key' } });
  fireEvent.change(screen.getByLabelText('API Key'), { target: { value: 'example-test-key' } });
  expect(screen.getByLabelText('API Key')).toHaveAttribute('type', 'password');
  const applyButton = screen.getByRole('button', { name: 'Apply authentication' });
  fireEvent.keyDown(screen.getByLabelText('API Key'), { key: 'Enter' });
  fireEvent.click(applyButton);
  await waitFor(() => expect(runtimeApi.configureAuthenticationPolicy).toHaveBeenCalledExactlyOnceWith(true, 'example-test-key'));
  expect(getApiToken()).toBe('example-test-key');
  expect(saveStudioCredential).toHaveBeenCalledWith('example-test-key');
  expect(screen.getByLabelText('API Key')).toHaveValue('');
});

it('disables authentication using the previous key and clears the client credential', async () => {
  setApiToken('example-old-key');
  vi.mocked(runtimeApi.authenticationPolicy).mockResolvedValue({ enabled: true, has_key: true });
  vi.mocked(runtimeApi.configureAuthenticationPolicy).mockImplementation(async () => {
    expect(getApiToken()).toBe('example-old-key');
    return { enabled: false, has_key: true };
  });
  await mount();
  fireEvent.change(screen.getByLabelText('Authentication'), { target: { value: 'none' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply authentication' }));
  await waitFor(() => expect(getApiToken()).toBe(''));
  expect(runtimeApi.configureAuthenticationPolicy).toHaveBeenCalledWith(false, undefined);
  expect(saveStudioCredential).toHaveBeenCalledWith('');
});

it('does not rotate an active key on unchanged Enter', async () => {
  vi.mocked(runtimeApi.authenticationPolicy).mockResolvedValue({ enabled: true, has_key: true });
  await mount();
  fireEvent.keyDown(screen.getByLabelText('API Key'), { key: 'Enter' });
  expect(runtimeApi.configureAuthenticationPolicy).not.toHaveBeenCalled();
});

it('restores the client vault after the server rejects a change', async () => {
  setApiToken('example-old-key');
  vi.mocked(runtimeApi.authenticationPolicy).mockResolvedValue({ enabled: true, has_key: true });
  vi.mocked(runtimeApi.configureAuthenticationPolicy).mockRejectedValue(new Error('save failed'));
  await mount();
  fireEvent.change(screen.getByLabelText('API Key'), { target: { value: 'example-new-key' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply authentication' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('save failed');
  expect(getApiToken()).toBe('example-old-key');
  expect(screen.getByLabelText('API Key')).toHaveValue('');
  expect(vi.mocked(saveStudioCredential).mock.calls).toEqual([['example-new-key'], ['example-old-key']]);
});

it('does not overwrite a different service after a delayed response', async () => {
  let finish!: (value: { enabled: boolean; has_key: boolean }) => void;
  vi.mocked(runtimeApi.configureAuthenticationPolicy).mockReturnValue(new Promise(resolve => { finish = resolve; }));
  await mount();
  fireEvent.change(screen.getByLabelText('Authentication'), { target: { value: 'api-key' } });
  fireEvent.change(screen.getByLabelText('API Key'), { target: { value: 'example-test-key' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply authentication' }));
  await waitFor(() => expect(runtimeApi.configureAuthenticationPolicy).toHaveBeenCalled());
  setApiBaseUrl('https://other.invalid'); setApiToken('example-other-key');
  await act(async () => { finish({ enabled: true, has_key: true }); });
  expect(getApiToken()).toBe('example-other-key');
});
