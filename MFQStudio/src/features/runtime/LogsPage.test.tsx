/** Verify independent SSE subscriptions and cursor-based log and request history. */
import { i18n } from '../../i18n';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { runtimeApi } from '../../shared/api/resources/runtime';
import type { RuntimeLogEntry, RuntimeMetricSnapshot } from '../../shared/api/types';
import { LogsPage } from './LogsPage';

const runtime = vi.hoisted(() => ({
  ready: true,
  connectionRevision: 1,
  instances: [],
  refreshRuntime: vi.fn(),
}));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => runtime }));
vi.mock('../settings/SettingsProvider', () => ({
  useSettings: () => ({ t: i18n.getFixedT('en') }),
}));

/** Build log rows with stable sequence identities for pagination tests. */
function logs(from: number, count: number): RuntimeLogEntry[] {
  return Array.from({ length: count }, (_, index) => ({
    sequence: from + index,
    created_at: '2026-10-06T00:00:00Z',
    level: 'info',
    message: `event-${from + index}`,
    fields: {},
  }));
}
/** Build one distinct sampled request per row. */
function requests(from: number, count: number): RuntimeMetricSnapshot[] {
  return logs(from, count).map(
    ({ sequence }) =>
      ({
        sequence,
        captured_at: '2026-10-06T00:00:00Z',
        values: { last_request: { id: `request-${sequence}` } },
      }) as RuntimeMetricSnapshot,
  );
}

let pushLog: (entry: RuntimeLogEntry) => void;
let pushRequest: (entry: RuntimeMetricSnapshot) => void;

beforeEach(() => {
  runtime.connectionRevision = 1;
  vi.useFakeTimers();
  vi.spyOn(runtimeApi, 'streamRuntimeLogs').mockImplementation((_after, callback, signal) => {
    pushLog = callback;
    return new Promise((resolve) => signal.addEventListener('abort', () => resolve(), { once: true }));
  });
  vi.spyOn(runtimeApi, 'streamRuntimeRequests').mockImplementation((_after, callback, signal) => {
    pushRequest = callback;
    return new Promise((resolve) => signal.addEventListener('abort', () => resolve(), { once: true }));
  });
  vi.spyOn(runtimeApi, 'runtimeStatus').mockResolvedValue({});
  vi.spyOn(runtimeApi, 'runtimeLogPage').mockResolvedValue([]);
  vi.spyOn(runtimeApi, 'runtimeRequestPage').mockResolvedValue([]);
});
afterEach(() => vi.useRealTimers());

it('shows calendar dates and seconds so events from different days remain distinguishable', async () => {
  const yesterday = new Date(2026, 9, 5, 14, 5, 9).toISOString();
  const today = new Date(2026, 9, 6, 14, 5, 9).toISOString();
  vi.mocked(runtimeApi.runtimeLogPage).mockResolvedValue([
    { ...logs(1, 1)[0], created_at: yesterday },
    { ...logs(2, 1)[0], created_at: today },
  ]);
  render(<LogsPage />);
  await act(async () => {});
  const times = screen.getByRole('region', { name: 'Runtime logs' }).querySelectorAll('time');
  expect(Array.from(times, (time) => time.textContent)).toEqual([
    '10/06/2026, 14:05:09',
    '10/05/2026, 14:05:09',
  ]);
  expect(times[0]).toHaveAttribute('datetime', today);
});

it('loads fifty latest rows per panel and appends older history on bottom scroll', async () => {
  vi.mocked(runtimeApi.runtimeLogPage)
    .mockResolvedValueOnce(logs(51, 50))
    .mockResolvedValueOnce(logs(1, 50));
  vi.mocked(runtimeApi.runtimeRequestPage)
    .mockResolvedValueOnce(requests(51, 50))
    .mockResolvedValueOnce(requests(1, 50));
  render(<LogsPage />);
  await act(async () => {});
  const logPanel = screen.getByRole('region', { name: 'Runtime logs' });
  const requestPanel = screen.getByRole('region', { name: 'Recent requests' });
  expect(logPanel.querySelectorAll('[data-history-id]')).toHaveLength(50);
  expect(requestPanel.querySelectorAll('[data-history-id]')).toHaveLength(50);
  expect(logPanel.querySelector('[data-history-id]')).toHaveTextContent('event-100');
  expect(screen.queryByText('event-1')).not.toBeInTheDocument();
  await act(async () => {
    fireEvent.scroll(logPanel);
    fireEvent.scroll(requestPanel);
  });
  expect(runtimeApi.runtimeLogPage).toHaveBeenLastCalledWith(
    50,
    { before: 51, order: 'desc' },
    expect.any(AbortSignal),
  );
  expect(runtimeApi.runtimeRequestPage).toHaveBeenLastCalledWith(
    50,
    { before: 51, order: 'desc' },
    expect.any(AbortSignal),
  );
  expect(screen.getByText('event-1')).toBeInTheDocument();
  expect(screen.getByText('request-1')).toBeInTheDocument();
  expect(logPanel.querySelectorAll('[data-history-id]')).toHaveLength(100);
});

it('keeps loaded history when SSE rows arrive and cancels subscriptions on unmount', async () => {
  vi.mocked(runtimeApi.runtimeLogPage)
    .mockResolvedValueOnce(logs(1, 50))
    .mockResolvedValueOnce(logs(51, 2));
  vi.mocked(runtimeApi.runtimeRequestPage)
    .mockResolvedValueOnce(requests(1, 50))
    .mockResolvedValueOnce(requests(51, 2));
  const view = render(<LogsPage />);
  await act(async () => {});
  await act(async () => {
    logs(51, 2).forEach(pushLog);
    requests(51, 2).forEach(pushRequest);
  });
  expect(runtimeApi.streamRuntimeLogs).toHaveBeenCalledWith(50, expect.any(Function), expect.any(AbortSignal));
  expect(screen.getByText('event-1')).toBeInTheDocument();
  expect(screen.getByText('event-52')).toBeInTheDocument();
  expect(screen.getByText('request-1')).toBeInTheDocument();
  const signal = vi.mocked(runtimeApi.runtimeLogPage).mock.calls[0][2]!;
  view.unmount();
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2000);
  });
  expect(runtimeApi.runtimeLogPage).toHaveBeenCalledTimes(1);
  expect(runtimeApi.runtimeStatus).not.toHaveBeenCalled();
  expect(signal.aborted).toBe(true);
});

it('keeps logs live without status sampling and retries failed older pages without gaps', async () => {
  vi.mocked(runtimeApi.runtimeStatus).mockReturnValue(new Promise(() => {}));
  vi.mocked(runtimeApi.runtimeLogPage)
    .mockResolvedValueOnce(logs(51, 50))
    .mockRejectedValueOnce(new Error('temporary failure'))
    .mockResolvedValueOnce(logs(1, 1));
  render(<LogsPage />);
  await act(async () => {});
  const panel = screen.getByRole('region', { name: 'Runtime logs' });
  await act(async () => {
    fireEvent.click(within(panel).getByText('Load older records'));
  });
  expect(within(panel).getByRole('alert')).toBeInTheDocument();
  await act(async () => {
    fireEvent.click(within(panel).getByText('Retry'));
  });
  expect(runtimeApi.runtimeLogPage).toHaveBeenLastCalledWith(
    50,
    { before: 51, order: 'desc' },
    expect.any(AbortSignal),
  );
  expect(within(panel).getByText('event-1')).toBeInTheDocument();
  expect(within(panel).getByText('No older records')).toBeInTheDocument();
});

it('ignores history responses from a previous connection', async () => {
  let resolveOld!: (entries: RuntimeLogEntry[]) => void;
  vi.mocked(runtimeApi.runtimeLogPage)
    .mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveOld = resolve;
        }),
    )
    .mockResolvedValueOnce(logs(20, 1));
  const view = render(<LogsPage />);
  await act(async () => {});
  runtime.connectionRevision = 2;
  view.rerender(<LogsPage />);
  await act(async () => {
    resolveOld(logs(1, 1));
  });
  expect(screen.getByText('event-20')).toBeInTheDocument();
  expect(screen.queryByText('event-1')).not.toBeInTheDocument();
});

it('deduplicates replayed request events', async () => {
  const first = requests(1, 1);
  vi.mocked(runtimeApi.runtimeRequestPage)
    .mockResolvedValueOnce(first)
    .mockResolvedValueOnce([{ ...first[0], sequence: 5 }]);
  render(<LogsPage />);
  await act(async () => {});
  await act(async () => {
    pushRequest(first[0]);
  });
  expect(screen.getAllByText('request-1')).toHaveLength(1);
});

it('preserves the visible history row when live events prepend', async () => {
  vi.mocked(runtimeApi.runtimeLogPage)
    .mockResolvedValueOnce(logs(1, 3))
    .mockResolvedValueOnce(logs(4, 1));
  render(<LogsPage />);
  await act(async () => {});
  const panel = screen.getByRole('region', { name: 'Runtime logs' });
  panel.scrollTop = 100;
  const row = panel.querySelector<HTMLElement>('[data-history-id="3"]')!;
  vi.spyOn(row, 'getBoundingClientRect')
    .mockReturnValueOnce({ top: 10, bottom: 50 } as DOMRect)
    .mockReturnValueOnce({ top: 10, bottom: 50 } as DOMRect)
    .mockReturnValue({ top: 40, bottom: 80 } as DOMRect);
  await act(async () => {
    pushLog(logs(4, 1)[0]);
  });
  expect(panel.scrollTop).toBe(130);
});


it('reconnects from the last applied cursor and does not refetch live pages', async () => {
  vi.mocked(runtimeApi.runtimeLogPage).mockResolvedValueOnce(logs(1, 1));
  vi.mocked(runtimeApi.streamRuntimeLogs).mockImplementationOnce(async (_after, callback) => {
    callback(logs(2, 1)[0]);
    throw new Error('connection lost');
  });
  render(<LogsPage />);
  await act(async () => {});
  await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
  expect(runtimeApi.streamRuntimeLogs).toHaveBeenLastCalledWith(2, expect.any(Function), expect.any(AbortSignal));
  expect(runtimeApi.runtimeLogPage).toHaveBeenCalledTimes(1);
  await act(async () => { pushLog(logs(3, 1)[0]); });
  expect(screen.getByText('event-3')).toBeInTheDocument();
});
