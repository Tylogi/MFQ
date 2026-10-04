/** Verify persistent failures and correct retry actions on ready and directly accessible pages. */
import { fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useRuntime } from './RuntimeProvider';
import { StudioShell } from './StudioShell';
import { NotFoundPage } from './NotFoundPage';

vi.mock('./RuntimeProvider', () => ({ useRuntime: vi.fn() }));
vi.mock('../features/settings/SettingsProvider', () => ({
  useSettings: () => ({ tr: (_chinese: string, english: string) => english }),
}));
/** Mount directly accessible routes and mock shared runtime states. */
function renderShell(overrides: Partial<ReturnType<typeof useRuntime>>, path = '/settings') {
  const runtime = {
    runtime: null,
    selectedModel: '',
    models: [],
    instances: [],
    loading: false,
    connectionError: null,
    refreshError: null,
    jobStreamErrors: {},
    ready: true,
    reloadService: vi.fn(),
    refreshRuntime: vi.fn(),
    retryJobStreams: vi.fn(),
    ...overrides,
  } as unknown as ReturnType<typeof useRuntime>;
  vi.mocked(useRuntime).mockReturnValue(runtime);
  render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route element={<StudioShell />}>
          <Route path="settings" element={<p>Settings content</p>} />
          <Route path="runtime" element={<p>Connection settings content</p>} />
          <Route path="chat" element={<p>Chat content</p>} />
          <Route index element={<p>Overview content</p>} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
  return runtime;
}

describe('describes StudioShell test behavior 1', () => {
  beforeEach(() => vi.clearAllMocks());

  it('verifies StudioShell test behavior 2', () => {
    const runtime = renderShell({ ready: false, connectionError: 'server offline' });
    expect(screen.getByText('Settings content')).toBeTruthy();
    expect(within(screen.getByRole('alert')).getByText('server offline')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Retry connection' }));
    expect(runtime.reloadService).toHaveBeenCalledOnce();
  });

  it('verifies StudioShell test behavior 3', () => {
    const runtime = renderShell({
      refreshError: 'refresh offline',
      jobStreamErrors: { 'job-1': 'stream offline' },
    });
    expect(screen.getAllByRole('alert')).toHaveLength(2);
    expect(screen.getByText('refresh offline')).toBeTruthy();
    expect(screen.getByText('job-1: stream offline')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Refresh again' }));
    expect(runtime.refreshRuntime).toHaveBeenCalledWith(false);
    fireEvent.click(screen.getByRole('button', { name: 'Reconnect task streams' }));
    expect(runtime.retryJobStreams).toHaveBeenCalledOnce();
  });

  it('verifies StudioShell test behavior 4', () => {
    const runtime = renderShell({ ready: false, connectionError: 'server offline' }, '/chat');
    expect(screen.getByRole('heading', { name: 'Service unavailable' })).toBeTruthy();
    expect(screen.queryByText('Chat content')).toBeNull();
    fireEvent.click(screen.getByText('View error details'));
    expect(screen.getByText('server offline')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Reconnect' }));
    expect(runtime.reloadService).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole('button', { name: 'Connection settings' }));
    expect(screen.getByText('Connection settings content')).toBeTruthy();
  });

  it('verifies StudioShell test behavior 5', () => {
    renderShell({ ready: false }, '/chat');
    expect(screen.getByRole('status')).toHaveTextContent('Loading…');
    expect(screen.getByRole('heading', { name: 'Loading…' })).toBeTruthy();
    expect(screen.queryByRole('heading', { name: 'Service unavailable' })).toBeNull();
  });

  it('verifies StudioShell test behavior 6', () => {
    renderShell({ ready: false, connectionError: 'server offline' }, '/missing');
    expect(screen.getByRole('heading', { name: 'Page not found' })).toBeTruthy();
    expect(screen.queryByRole('heading', { name: 'Service unavailable' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Back to overview' }));
    expect(screen.getByRole('heading', { name: 'Service unavailable' })).toBeTruthy();
  });

  it('verifies StudioShell test behavior 7', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    function CrashingPage(): never {
      throw new Error('page render failed');
    }
    vi.mocked(useRuntime).mockReturnValue({
      runtime: null, selectedModel: '', models: [], instances: [], loading: false,
      connectionError: null, refreshError: null, jobStreamErrors: {}, ready: true,
      reloadService: vi.fn(), refreshRuntime: vi.fn(), retryJobStreams: vi.fn(),
    } as unknown as ReturnType<typeof useRuntime>);
    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Routes>
          <Route element={<StudioShell />}>
            <Route path="chat" element={<CrashingPage />} />
            <Route index element={<p>Overview content</p>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByRole('heading', { name: 'Page unavailable' })).toBeTruthy();
    fireEvent.click(screen.getByText('View error details'));
    expect(screen.getByText('page render failed')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Retry page' })).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Back to overview' }));
    expect(screen.getByText('Overview content')).toBeTruthy();
    vi.restoreAllMocks();
  });
});
