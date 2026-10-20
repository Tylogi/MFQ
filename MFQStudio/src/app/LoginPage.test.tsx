import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { ApiError, getApiToken, request, setApiBaseUrl, setApiToken, setBrowserServiceUrl } from '../shared/api/client';
import { runtimeApi } from '../shared/api/resources/runtime';
import { saveStudioCredential } from '../studio';
import { useJobStore } from '../stores/jobStore';
import { RuntimeProvider } from './RuntimeProvider';
import { StudioShell } from './StudioShell';

vi.mock('../studio', () => ({
  studioStatus: vi.fn().mockResolvedValue(null),
  studioCredential: vi.fn(),
  saveStudioCredential: vi.fn().mockResolvedValue(undefined),
  startLocalStudio: vi.fn(),
}));
vi.mock('../features/settings/SettingsProvider', () => ({
  useSettings: () => ({ tr: (_zh: string, en: string) => en }),
}));
vi.mock('./StudioSidebar', () => ({ StudioSidebar: () => <nav>Studio navigation</nav> }));

let enabled: boolean;
let fetchMock: ReturnType<typeof vi.fn>;
const validKey = 'fixture-login-key';

function response(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status });
}

function mount(path = '/') {
  return render(<MemoryRouter initialEntries={[path]}><RuntimeProvider><Routes>
    <Route element={<StudioShell />}>
      <Route index element={<p>Overview content</p>} />
      <Route path="runtime" element={<p>Service content</p>} />
      <Route path="chat" element={<p>Chat content</p>} />
      <Route path="settings" element={<p>Settings content</p>} />
    </Route>
  </Routes></RuntimeProvider></MemoryRouter>);
}

function submit(key = validKey) {
  fireEvent.change(screen.getByLabelText('API Key'), { target: { value: key } });
  fireEvent.click(screen.getByRole('button', { name: 'Sign in' }));
}

beforeEach(() => {
  enabled = true;
  setApiBaseUrl(''); setApiToken(''); setBrowserServiceUrl('');
  useJobStore.getState().setJobs([]);
  vi.mocked(saveStudioCredential).mockResolvedValue(undefined);
  fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (enabled && new Headers(init?.headers).get('Authorization') !== `Bearer ${validKey}`) {
      return response({ error: { code: 'unauthorized', message: 'invalid API credential', retryable: false, details: {} } }, 401);
    }
    if (url === '/api/v1/runtime/status') return response({ runtime_state: 'idle' });
    return response({ data: [] });
  });
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  setApiBaseUrl(''); setApiToken(''); setBrowserServiceUrl('');
});

it.each(['/', '/runtime', '/chat', '/settings'])('shows sign-in instead of protected content or a connection failure at %s', async (path) => {
  mount(path);
  await screen.findByRole('heading', { name: 'Sign in' });
  expect(screen.queryByText('Studio navigation')).not.toBeInTheDocument();
  expect(screen.queryByText(/content$/)).not.toBeInTheDocument();
  expect(screen.queryByText('Service unavailable')).not.toBeInTheDocument();
  expect(screen.queryByText('invalid API credential')).not.toBeInTheDocument();
  await waitFor(() => expect(document.title).toBe('Sign in · MFQ Studio'));
});

it('skips sign-in when authentication is disabled', async () => {
  enabled = false;
  mount();
  await screen.findByText('Overview content');
  expect(screen.queryByRole('heading', { name: 'Sign in' })).not.toBeInTheDocument();
  expect(saveStudioCredential).not.toHaveBeenCalled();
});

it('validates a key, restores the original page, and does not change authentication policy or persist the browser key', async () => {
  mount('/chat');
  await screen.findByRole('heading', { name: 'Sign in' });
  submit(`  ${validKey}  `);
  await screen.findByText('Chat content');
  expect(getApiToken()).toBe(validKey);
  expect(saveStudioCredential).toHaveBeenCalledExactlyOnceWith(validKey);
  expect(localStorage.length).toBe(0);
  expect(sessionStorage.length).toBe(0);
  expect(fetchMock.mock.calls.every(([url, init]) => !String(url).includes(validKey) && !init?.method)).toBe(true);
  expect(fetchMock.mock.calls.some(([url]) => String(url).includes('/auth/'))).toBe(false);
});

it('shows a friendly invalid-key error and permits a second attempt', async () => {
  mount();
  await screen.findByRole('heading', { name: 'Sign in' });
  submit('fixture-wrong-key');
  expect(await screen.findByRole('alert')).toHaveTextContent('The API key is incorrect or expired.');
  expect(getApiToken()).toBe('');
  expect(saveStudioCredential).not.toHaveBeenCalled();
  submit();
  await screen.findByText('Overview content');
});

it('keeps network failures separate from authentication failures', async () => {
  fetchMock.mockRejectedValue(new TypeError('Failed to fetch'));
  mount();
  await screen.findByRole('heading', { name: 'Service unavailable' });
  expect(screen.queryByLabelText('API Key')).not.toBeInTheDocument();
});

it('does not mistake insufficient scope for an invalid or missing key', async () => {
  fetchMock.mockImplementation(async () => response({ error: {
    code: 'insufficient_scope', message: 'Missing jobs scope', retryable: false, details: {},
  } }, 403));
  mount('/runtime');
  await screen.findByText(/Missing jobs scope/);
  expect(screen.queryByRole('heading', { name: 'Sign in' })).not.toBeInTheDocument();
});

it('returns to sign-in when a currently open page receives a 401', async () => {
  enabled = false;
  mount();
  await screen.findByText('Overview content');
  enabled = true;
  await act(async () => { await expect(request('/api/v1/models')).rejects.toBeInstanceOf(ApiError); });
  await screen.findByRole('heading', { name: 'Sign in' });
  submit();
  await screen.findByText('Overview content');
});

it('does not submit duplicate sign-in requests', async () => {
  mount();
  await screen.findByRole('heading', { name: 'Sign in' });
  let finish!: (value: Response) => void;
  fetchMock.mockImplementationOnce(() => new Promise<Response>((resolve) => { finish = resolve; }));
  submit();
  const field = screen.getByLabelText('API Key');
  const calls = fetchMock.mock.calls.length;
  fireEvent.submit(field.closest('form')!);
  expect(fetchMock).toHaveBeenCalledTimes(calls);
  expect(screen.getByRole('button', { name: 'Signing in…' })).toBeDisabled();
  await act(async () => { finish(response({ data: [] })); });
  await screen.findByText('Overview content');
});

it('does not apply a credential if the connection changes during verification', async () => {
  mount();
  await screen.findByRole('heading', { name: 'Sign in' });
  let finish!: (value: Response) => void;
  fetchMock.mockImplementationOnce(() => new Promise<Response>((resolve) => { finish = resolve; }));
  submit();
  setApiBaseUrl('https://another.invalid');
  await act(async () => { finish(response({ data: [] })); });
  await waitFor(() => expect(screen.getByRole('button', { name: 'Sign in' })).toBeEnabled());
  expect(getApiToken()).toBe('');
  expect(saveStudioCredential).not.toHaveBeenCalled();
});

it('keeps a failed credential-vault write on the sign-in page without applying the key', async () => {
  vi.mocked(saveStudioCredential).mockRejectedValue(new Error('vault locked'));
  mount();
  await screen.findByRole('heading', { name: 'Sign in' });
  submit();
  expect(await screen.findByRole('alert')).toHaveTextContent('Could not save the sign-in credential.');
  expect(getApiToken()).toBe('');
});

it('handles structured 401 errors from runtime refreshes too', async () => {
  vi.spyOn(runtimeApi, 'runtimeInstances').mockRejectedValue(new ApiError(401, { error: {
    code: 'unauthorized', message: 'invalid API credential', retryable: false, details: {},
  } }));
  mount();
  await screen.findByRole('heading', { name: 'Sign in' });
});
