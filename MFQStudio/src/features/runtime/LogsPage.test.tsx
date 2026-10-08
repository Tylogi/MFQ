import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { LogsPage } from './LogsPage';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { jobsApi } from '../../shared/api/resources/jobs';
import { useJobStore } from '../../stores/jobStore';
import { studioConfirm } from '../../studio';
import { mergeRequestRecords, requestRecords } from './useLogHistory';
import type { JobResource, RuntimeInstance, RuntimeLogEntry, RuntimeMetricSnapshot } from '../../shared/api/types';

const state = vi.hoisted(() => ({ ready: true, connectionRevision: 0, refreshRuntime: vi.fn().mockResolvedValue(true),
  refreshError: null as string | null,
  instances: [] as RuntimeInstance[] }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => state }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('../../studio', () => ({ studioConfirm: vi.fn().mockResolvedValue(true) }));

const stamp = () => new Date().toISOString();
function event(sequence: number, level: RuntimeLogEntry['level'], message: string, instance = 'a'): RuntimeLogEntry {
  return { sequence, level, message, instance_id: instance, fields: { source: 'runtime.stdout' }, created_at: stamp() };
}
function snapshot(sequence: number, instance = 'a'): RuntimeMetricSnapshot {
  return { sequence, instance_id: instance, model: instance === 'a' ? 'Qwen3.8' : 'DeepSeek', captured_at: stamp(),
    values: { last_request: { id: 'request-one', completed_at: Date.now() / 1000, stream: true,
      prompt_tokens: 1024, completion_tokens: 128, prefill_tps: 2800, prefill_ms: 366, ttft_ms: 500,
      decode_tps: 60, decode_ms: 2133, finish_reason: 'length', complete_generation_ms: 2633 } } };
}
function job(id = 'job-a'): JobResource {
  return { id, kind: 'model.load', status: 'succeeded', payload: { model: 'Qwen3.8' }, progress: 1,
    cancel_requested: false, result: { instance_id: 'a' }, created_at: stamp(), updated_at: stamp() };
}
beforeEach(() => {
  state.ready = true; state.connectionRevision = 0;
  state.refreshError = null; state.refreshRuntime.mockReset().mockResolvedValue(true);
  state.instances = [{ id: 'a', model: 'Qwen3.8', state: 'ready', devices: ['metal'], active_sessions: 0, queued_requests: 0 },
    { id: 'b', model: 'DeepSeek', state: 'ready', devices: ['metal'], active_sessions: 0, queued_requests: 0 }];
  useJobStore.getState().setJobs([job()]);
  vi.spyOn(runtimeApi, 'runtimeLogs').mockResolvedValue([event(1, 'info', 'Model loaded'), event(2, 'warning', 'Memory pressure'), event(3, 'error', 'PLE failed', 'b')]);
  vi.spyOn(runtimeApi, 'runtimeMetrics').mockResolvedValue([snapshot(1), snapshot(2)]);
  vi.spyOn(jobsApi, 'jobEvents').mockResolvedValue([event(1, 'info', 'Weights ready')]);
  vi.mocked(studioConfirm).mockResolvedValue(true);
});
afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers(); });

it('keeps a known model name when subsequent request snapshots omit it', () => {
  const first = requestRecords([snapshot(1)])[0];
  const next = { ...first, model: null, request: { id: first.request.id, finish_reason: 'stop' } };
  expect(mergeRequestRecords([first, next])[0]).toMatchObject({ model: 'Qwen3.8',
    request: { decode_tps: 60, finish_reason: 'stop' } });
});

it('uses one console with three selectable tabs and displays more than eight events, newest first', async () => {
  vi.mocked(runtimeApi.runtimeLogs).mockResolvedValue(Array.from({ length: 12 }, (_, i) => event(i + 1, 'info', `message-${i + 1}`)));
  const { container } = render(<LogsPage />);
  await screen.findByText('message-12');
  expect(container.querySelectorAll('.logs-console')).toHaveLength(1);
  expect(container.querySelectorAll('.log-event-record')).toHaveLength(12);
  expect(container.querySelector('.log-record')).toHaveTextContent('message-12');
  expect(screen.getAllByRole('tab')).toHaveLength(3);
  fireEvent.click(screen.getByRole('tab', { name: /Requests/ }));
  expect(screen.getByRole('tabpanel')).toHaveTextContent('60 tok/s');
  expect(container.querySelectorAll('.log-request-record')).toHaveLength(1);
});

it('combines search, severity, source and time filters and can reset them', async () => {
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.change(screen.getByLabelText('Level'), { target: { value: 'warning' } });
  expect(screen.getByRole('tabpanel')).toHaveTextContent('Memory pressure');
  expect(screen.getByRole('tabpanel')).not.toHaveTextContent('PLE failed');
  fireEvent.change(screen.getByRole('searchbox'), { target: { value: 'MEMORY' } });
  fireEvent.change(screen.getByLabelText('Source'), { target: { value: 'b' } });
  expect(screen.getByText('No matching records')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Reset filters' }));
  expect(screen.getByText('PLE failed')).toBeInTheDocument();
  expect(screen.getByRole('searchbox')).toHaveValue('');
});

it('defaults to recent records and includes older ones only when the time filter is widened', async () => {
  const old = event(1, 'info', 'Old startup'); old.created_at = '2020-01-01T00:00:00Z';
  vi.mocked(runtimeApi.runtimeLogs).mockResolvedValue([old, event(2, 'info', 'Current startup')]);
  render(<LogsPage />);
  await screen.findByText('Current startup');
  expect(screen.queryByText('Old startup')).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('Time'), { target: { value: '' } });
  expect(screen.getByText('Old startup')).toBeInTheDocument();
});

it('expands raw fields and copies the actual selected event', async () => {
  const copy = vi.fn().mockResolvedValue(undefined);
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: copy } });
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.click(screen.getByRole('button', { name: /PLE failed/ }));
  expect(screen.getByText('Raw record')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Copy' }));
  await waitFor(() => expect(copy).toHaveBeenCalled());
  expect(JSON.parse(copy.mock.calls[0][0])).toMatchObject({ sequence: 3, level: 'error', message: 'PLE failed' });
});

it('deduplicates request snapshots per instance and keeps the latest measurements', () => {
  const old = snapshot(1), latest = snapshot(2), other = snapshot(3, 'b');
  old.values.last_request!.decode_tps = 40;
  expect(requestRecords([other, latest, old])).toHaveLength(2);
  expect(requestRecords([other, latest, old]).find((item) => item.instance_id === 'a')?.request.decode_tps).toBe(60);
});

it('preserves native measurements when a later request snapshot is partial', () => {
  const old = snapshot(1), latest = snapshot(2);
  latest.values.last_request = { id: 'request-one', finish_reason: 'stop' };
  const combined = requestRecords([latest, old]);
  expect(combined).toHaveLength(1);
  expect(combined[0].request).toMatchObject({ decode_tps: 60, completion_tokens: 128, finish_reason: 'stop' });
  const previous = requestRecords([old]);
  const next = requestRecords([latest]);
  expect(mergeRequestRecords([...previous, ...next])[0].request).toMatchObject({ decode_tps: 60, finish_reason: 'stop' });
});

it('keeps the precise completion time when a later native status rounds it down to seconds', () => {
  const older = snapshot(1);
  older.values.last_request = { id: 'older', completed_at: 100.8, decode_tps: 40 };
  const latest = snapshot(2);
  latest.values.last_request = { id: 'newer', completed_at: 100.9, decode_tps: 60 };
  const status = snapshot(3);
  status.values.last_request = { id: 'newer', completed_at: 100, decode_tps: 60 };
  const records = requestRecords([older, latest, status]);
  expect(records[0].request).toMatchObject({ id: 'newer', completed_at: 100.9, decode_tps: 60 });
  expect(mergeRequestRecords([...requestRecords([older, latest]), ...requestRecords([status])])[0].request.id).toBe('newer');
});

it('does not infer a non-streaming request from unavailable transport metadata', async () => {
  const value = snapshot(1);
  delete value.values.last_request!.stream;
  vi.mocked(runtimeApi.runtimeMetrics).mockResolvedValue([value]);
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.click(screen.getByRole('tab', { name: /Requests/ }));
  expect(screen.getByRole('tabpanel')).not.toHaveTextContent('Non-streaming');
});

it.each([true, false])('does not present native Runtime stream=%s as the client transport mode', async (stream) => {
  const value = snapshot(1);
  value.values.last_request!.stream = stream;
  vi.mocked(runtimeApi.runtimeMetrics).mockResolvedValue([value]);
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.click(screen.getByRole('tab', { name: /Requests/ }));
  expect(screen.getByRole('tabpanel')).not.toHaveTextContent(' · Streaming');
  expect(screen.getByRole('tabpanel')).not.toHaveTextContent(' · Non-streaming');
  fireEvent.click(within(screen.getByRole('tabpanel')).getByRole('button'));
  expect(document.querySelector('.log-record-detail > pre')).toHaveTextContent(`"stream": ${stream}`);
});

it('shows native prefill/decode and TTFT separately in expandable request details', async () => {
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.click(screen.getByRole('tab', { name: /Requests/ }));
  fireEvent.click(within(screen.getByRole('tabpanel')).getByRole('button'));
  expect(screen.getByText('2,800 tok/s · 366 ms')).toBeInTheDocument();
  expect(screen.getByText('60 tok/s · 2,133 ms')).toBeInTheDocument();
  expect(screen.getByText('500 ms')).toBeInTheDocument();
});

it('uses native generation time when the optional complete timing is a zero placeholder', async () => {
  const value = snapshot(1);
  value.values.last_request!.complete_generation_ms = 0;
  value.values.last_request!.generation_ms = 4649.4;
  vi.mocked(runtimeApi.runtimeMetrics).mockResolvedValue([value]);
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.click(screen.getByRole('tab', { name: /Requests/ }));
  fireEvent.click(within(screen.getByRole('tabpanel')).getByRole('button'));
  const total = screen.getByText('Total request time').parentElement!;
  expect(total).toHaveTextContent('4,649.4 ms');
  expect(total).not.toHaveTextContent('0 ms');
});

it('pauses network polling, supports a manual refresh while paused and resumes immediately', async () => {
  vi.useFakeTimers();
  render(<LogsPage />);
  await act(async () => {});
  expect(runtimeApi.runtimeLogs).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole('button', { name: 'Pause updates' }));
  await act(async () => { await vi.advanceTimersByTimeAsync(12000); });
  expect(runtimeApi.runtimeLogs).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  await act(async () => {});
  expect(runtimeApi.runtimeLogs).toHaveBeenCalledTimes(2);
  expect(screen.getByText('Paused')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Resume updates' }));
  await act(async () => {});
  expect(runtimeApi.runtimeLogs).toHaveBeenCalledTimes(3);
  await act(async () => { await vi.advanceTimersByTimeAsync(4000); });
  expect(runtimeApi.runtimeLogs).toHaveBeenCalledTimes(4);
});

it('stops polling on navigation and aborts owned requests', async () => {
  vi.useFakeTimers();
  const view = render(<LogsPage />);
  await act(async () => {});
  const signal = vi.mocked(runtimeApi.runtimeLogs).mock.calls[0][1];
  expect(signal?.aborted).toBe(false);
  view.unmount();
  expect(signal?.aborted).toBe(true);
  await act(async () => { await vi.advanceTimersByTimeAsync(12000); });
  expect(runtimeApi.runtimeLogs).toHaveBeenCalledTimes(1);
});

it('defers hidden-page polling and immediately catches up when the page becomes visible', async () => {
  vi.useFakeTimers();
  const visibility = vi.spyOn(document, 'hidden', 'get').mockReturnValue(true);
  const view = render(<LogsPage />);
  await act(async () => { await vi.advanceTimersByTimeAsync(12000); });
  expect(runtimeApi.runtimeLogs).not.toHaveBeenCalled();
  expect(runtimeApi.runtimeMetrics).not.toHaveBeenCalled();
  visibility.mockReturnValue(false);
  fireEvent(document, new Event('visibilitychange'));
  await act(async () => {});
  expect(runtimeApi.runtimeLogs).toHaveBeenCalledOnce();
  expect(screen.getByText('PLE failed')).toBeInTheDocument();
  view.unmount();
  fireEvent(document, new Event('visibilitychange'));
  await act(async () => { await vi.advanceTimersByTimeAsync(12000); });
  expect(runtimeApi.runtimeLogs).toHaveBeenCalledOnce();
});

it('does not duplicate a pending history request when visibility changes repeatedly', async () => {
  vi.useFakeTimers();
  let resolve!: (entries: RuntimeLogEntry[]) => void;
  vi.mocked(runtimeApi.runtimeLogs).mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
  render(<LogsPage />);
  fireEvent(document, new Event('visibilitychange'));
  fireEvent(document, new Event('visibilitychange'));
  await act(async () => { await vi.advanceTimersByTimeAsync(12000); });
  expect(runtimeApi.runtimeLogs).toHaveBeenCalledOnce();
  expect(runtimeApi.runtimeMetrics).toHaveBeenCalledOnce();
  await act(async () => { resolve([event(4, 'info', 'Pending history completed')]); });
  expect(screen.getByText('Pending history completed')).toBeInTheDocument();
  await act(async () => { await vi.advanceTimersByTimeAsync(4000); });
  expect(runtimeApi.runtimeLogs).toHaveBeenCalledTimes(2);
});

it('keeps a paused manual refresh pending if the page is hidden before its history request', async () => {
  vi.useFakeTimers();
  const visibility = vi.spyOn(document, 'hidden', 'get').mockReturnValue(false);
  render(<LogsPage />);
  await act(async () => {});
  fireEvent.click(screen.getByRole('button', { name: 'Pause updates' }));
  let resolve!: (success: boolean) => void;
  state.refreshRuntime.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  visibility.mockReturnValue(true);
  await act(async () => { resolve(true); });
  expect(runtimeApi.runtimeLogs).toHaveBeenCalledOnce();
  vi.mocked(runtimeApi.runtimeLogs).mockResolvedValue([event(4, 'info', 'Manually retrieved after return')]);
  visibility.mockReturnValue(false);
  fireEvent(document, new Event('visibilitychange'));
  await act(async () => {});
  expect(screen.getByText('Manually retrieved after return')).toBeInTheDocument();
  expect(screen.getByText('Paused')).toBeInTheDocument();
  expect(runtimeApi.runtimeLogs).toHaveBeenCalledTimes(2);
  fireEvent(document, new Event('visibilitychange'));
  await act(async () => { await vi.advanceTimersByTimeAsync(12000); });
  expect(runtimeApi.runtimeLogs).toHaveBeenCalledTimes(2);
});

it('keeps one successful stream usable if the other query fails and visibly reports the failure', async () => {
  vi.mocked(runtimeApi.runtimeMetrics).mockRejectedValue(new Error('metrics offline'));
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  expect(screen.getByRole('status')).toHaveTextContent('metrics offline');
  expect(screen.getByText('Refresh failed')).toBeInTheDocument();
});

it('does not mix a disconnected server’s late response into the new connection', async () => {
  let resolve: (entries: RuntimeLogEntry[]) => void = () => {};
  vi.mocked(runtimeApi.runtimeLogs).mockReturnValueOnce(new Promise((done) => { resolve = done; }));
  const view = render(<LogsPage />);
  state.connectionRevision = 1;
  vi.mocked(runtimeApi.runtimeLogs).mockResolvedValue([event(10, 'info', 'New service')]);
  view.rerender(<LogsPage />);
  await screen.findByText('New service');
  await act(async () => { resolve([event(9, 'info', 'Old service')]); });
  expect(screen.queryByText('Old service')).not.toBeInTheDocument();
});

it('shows task events and deletes only the selected completed record after confirmation', async () => {
  const remove = vi.spyOn(jobsApi, 'deleteJob').mockResolvedValue(undefined);
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.click(screen.getByRole('tab', { name: /Jobs/ }));
  fireEvent.click(screen.getByRole('button', { name: /Load model/ }));
  await screen.findByText('Weights ready');
  fireEvent.click(screen.getByRole('button', { name: 'Delete this record' }));
  await waitFor(() => expect(remove).toHaveBeenCalledWith('job-a'));
  expect(studioConfirm).toHaveBeenCalled();
  expect(useJobStore.getState().jobs).toHaveLength(0);
});

it('retains job events while a newer progress snapshot is loading or fails', async () => {
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.click(screen.getByRole('tab', { name: /Jobs/ }));
  fireEvent.click(screen.getByRole('button', { name: /Load model/ }));
  await screen.findByText('Weights ready');
  let reject!: (error: Error) => void;
  vi.mocked(jobsApi.jobEvents).mockReturnValueOnce(new Promise((_resolve, fail) => { reject = fail; }));
  await act(async () => { useJobStore.getState().setJobs([{ ...job(), updated_at: new Date(Date.now() + 1000).toISOString() }]); });
  expect(screen.getByText('Weights ready')).toBeInTheDocument();
  await act(async () => { reject(new Error('job events offline')); });
  expect(screen.getByText('Weights ready')).toBeInTheDocument();
  expect(screen.getByRole('status')).toHaveTextContent('job events offline');
});

it('refreshes expanded job events even when the job progress timestamp has not changed', async () => {
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.click(screen.getByRole('tab', { name: /Jobs/ }));
  fireEvent.click(screen.getByRole('button', { name: /Load model/ }));
  await screen.findByText('Weights ready');
  fireEvent.click(screen.getByRole('button', { name: 'Pause updates' }));
  vi.useFakeTimers();
  await act(async () => { await vi.advanceTimersByTimeAsync(1); });
  vi.mocked(jobsApi.jobEvents).mockResolvedValue([event(1, 'info', 'Weights ready'), event(2, 'info', 'New task output')]);
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  await act(async () => {});
  expect(screen.getByText('New task output')).toBeInTheDocument();
  expect(screen.getByText('Paused')).toBeInTheDocument();
});

it('does not describe a failed initial job-event query as an empty event history', async () => {
  vi.mocked(jobsApi.jobEvents).mockRejectedValue(new Error('job events offline'));
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.click(screen.getByRole('tab', { name: /Jobs/ }));
  fireEvent.click(screen.getByRole('button', { name: /Load model/ }));
  await screen.findByText('job events offline');
  expect(screen.queryByText('No additional events')).not.toBeInTheDocument();
});

it('filters unload and evaluation jobs by their target instance, not only by model names', async () => {
  useJobStore.getState().setJobs([
    { ...job('unload'), kind: 'model.unload', payload: { instance_id: 'a' }, result: null },
    { ...job('evaluation'), kind: 'evaluate.accuracy', payload: { instance_id: 'a' }, result: null },
    { ...job('other-instance'), kind: 'benchmark.inference', payload: { instance_id: 'b' }, result: null },
  ]);
  const { container } = render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.click(screen.getByRole('tab', { name: /Jobs/ }));
  fireEvent.change(screen.getByLabelText('Source'), { target: { value: 'a' } });
  expect(container.querySelectorAll('.log-job-record')).toHaveLength(2);
  expect(screen.getByRole('tabpanel')).toHaveTextContent('Unload model');
  expect(screen.getByRole('tabpanel')).toHaveTextContent('Task evaluation');
  expect(screen.getByRole('tabpanel')).not.toHaveTextContent('Inference benchmark');
  expect(within(screen.getByRole('tabpanel')).getAllByText('Qwen3.8')).toHaveLength(2);
});

it('keeps unloaded job targets available in the source filter without request snapshots', async () => {
  state.instances = [];
  vi.mocked(runtimeApi.runtimeLogs).mockResolvedValue([]);
  vi.mocked(runtimeApi.runtimeMetrics).mockResolvedValue([]);
  useJobStore.getState().setJobs([job(),
    { ...job('unloaded'), kind: 'model.unload', payload: { instance_id: 'a' }, result: null }]);
  const { container } = render(<LogsPage />);
  await waitFor(() => expect(screen.getByRole('option', { name: 'Qwen3.8 · a' })).toBeInTheDocument());
  fireEvent.click(screen.getByRole('tab', { name: /Jobs/ }));
  fireEvent.change(screen.getByLabelText('Source'), { target: { value: 'a' } });
  expect(container.querySelectorAll('.log-job-record')).toHaveLength(2);
  expect(screen.getByRole('tabpanel')).toHaveTextContent('Unload model');
});

it('uses the resolved model name rather than the submitted asset ID in historical tasks', async () => {
  state.instances = [];
  vi.mocked(runtimeApi.runtimeLogs).mockResolvedValue([]);
  vi.mocked(runtimeApi.runtimeMetrics).mockResolvedValue([]);
  useJobStore.getState().setJobs([{ ...job(), payload: { model: 'submitted-asset-id' },
    result: { instance_id: 'past-a', model: 'Qwen3.8' } }]);
  render(<LogsPage />);
  await waitFor(() => expect(screen.getByRole('option', { name: 'Qwen3.8 · past-a' })).toBeInTheDocument());
  fireEvent.click(screen.getByRole('tab', { name: /Jobs/ }));
  fireEvent.change(screen.getByLabelText('Source'), { target: { value: 'past-a' } });
  expect(screen.getByRole('tabpanel')).toHaveTextContent('Qwen3.8');
  expect(screen.getByRole('tabpanel')).not.toHaveTextContent('submitted-asset-id');
  fireEvent.click(screen.getByRole('button', { name: /Load model/ }));
  await screen.findByText('Weights ready');
  expect(document.querySelector('.log-record-detail > pre')).toHaveTextContent('submitted-asset-id');
});

it.each(['runtime.admission', 'runtime.stderr', 'runtime.lifecycle', 'runtime.log-drain'])('uses explicitly reported %s model names after the instance and job history are gone', async (source) => {
  state.instances = [];
  useJobStore.getState().setJobs([]);
  vi.mocked(runtimeApi.runtimeMetrics).mockResolvedValue([]);
  vi.mocked(runtimeApi.runtimeLogs).mockResolvedValue([{ ...event(1, 'warning', 'Request rejected', 'gone'),
    fields: { source, model: 'Qwen3.8' } }]);
  render(<LogsPage />);
  await screen.findByText('Request rejected');
  expect(screen.getByRole('option', { name: 'Qwen3.8 · gone' })).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('Source'), { target: { value: 'gone' } });
  expect(screen.getByRole('tabpanel')).toHaveTextContent(`Qwen3.8 · ${source}`);
});

it('keeps known task instance identities separate when the same model was loaded more than once', async () => {
  useJobStore.getState().setJobs([job('current'), { ...job('previous'), result: { instance_id: 'past-a' } },
    { ...job('not-started'), status: 'failed', result: null }]);
  const { container } = render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.click(screen.getByRole('tab', { name: /Jobs/ }));
  fireEvent.change(screen.getByLabelText('Source'), { target: { value: 'a' } });
  expect(container.querySelectorAll('.log-job-record')).toHaveLength(2);
  fireEvent.change(screen.getByLabelText('Source'), { target: { value: 'past-a' } });
  expect(container.querySelectorAll('.log-job-record')).toHaveLength(2);
});

it('never clears task history when confirmation is cancelled', async () => {
  const clear = vi.spyOn(jobsApi, 'clearCompletedJobs').mockResolvedValue(undefined);
  vi.mocked(studioConfirm).mockResolvedValue(false);
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.click(screen.getByRole('tab', { name: /Jobs/ }));
  fireEvent.click(screen.getByRole('button', { name: 'Clear finished records' }));
  await act(async () => {});
  expect(clear).not.toHaveBeenCalled();
});

it('exports the filtered records without changing any server state', async () => {
  const create = vi.fn().mockReturnValue('blob:test');
  const revoke = vi.fn();
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: create });
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revoke });
  const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.change(screen.getByLabelText('Level'), { target: { value: 'error' } });
  fireEvent.click(screen.getByRole('button', { name: 'Export' }));
  expect(create).toHaveBeenCalledWith(expect.any(Blob));
  expect(click).toHaveBeenCalledTimes(1);
});

it('does not clear the new server when an old confirmation is accepted after switching connections', async () => {
  let confirm!: (accepted: boolean) => void;
  vi.mocked(studioConfirm).mockReturnValue(new Promise((resolve) => { confirm = resolve; }));
  const clear = vi.spyOn(jobsApi, 'clearCompletedJobs').mockResolvedValue(undefined);
  const view = render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.click(screen.getByRole('tab', { name: /Jobs/ }));
  fireEvent.click(screen.getByRole('button', { name: 'Clear finished records' }));
  state.connectionRevision = 1;
  view.rerender(<LogsPage />);
  await act(async () => { confirm(true); });
  expect(clear).not.toHaveBeenCalled();
  expect(useJobStore.getState().jobs).toHaveLength(1);
});

it('fetches only new records after the initial window and merges them without duplicates', async () => {
  vi.useFakeTimers();
  render(<LogsPage />);
  await act(async () => {});
  vi.mocked(runtimeApi.runtimeLogs).mockResolvedValue([event(4, 'info', 'New event')]);
  vi.mocked(runtimeApi.runtimeMetrics).mockResolvedValue([snapshot(3, 'b')]);
  await act(async () => { await vi.advanceTimersByTimeAsync(4000); });
  expect(vi.mocked(runtimeApi.runtimeLogs).mock.calls[1][2]).toBe(3);
  expect(vi.mocked(runtimeApi.runtimeMetrics).mock.calls[1][2]).toEqual(expect.any(String));
  expect(screen.getByText('Model loaded')).toBeInTheDocument();
  expect(screen.getByText('New event')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('tab', { name: /Requests/ }));
  expect(document.querySelectorAll('.log-request-record')).toHaveLength(2);
});

it('catches up to the latest window when more than 500 events accumulate between polls', async () => {
  vi.useFakeTimers();
  const view = render(<LogsPage />);
  await act(async () => {});
  vi.mocked(runtimeApi.runtimeLogs)
    .mockResolvedValueOnce(Array.from({ length: 500 }, (_, i) => event(i + 4, 'info', `backlog-${i + 4}`)))
    .mockResolvedValueOnce(Array.from({ length: 500 }, (_, i) => event(i + 504, 'info', `latest-${i + 504}`)));
  await act(async () => { await vi.advanceTimersByTimeAsync(4000); });
  expect(vi.mocked(runtimeApi.runtimeLogs).mock.calls[2][2]).toBeUndefined();
  expect(view.container.querySelectorAll('.log-event-record')).toHaveLength(500);
  expect(view.container.querySelector('.log-record')).toHaveTextContent('latest-1003');
  expect(screen.queryByText('backlog-503')).not.toBeInTheDocument();
  await act(async () => { await vi.advanceTimersByTimeAsync(4000); });
  expect(vi.mocked(runtimeApi.runtimeLogs).mock.calls[3][2]).toBe(1003);
});

it('does not move the last retrieved timestamp when both history queries fail', async () => {
  vi.useFakeTimers();
  const view = render(<LogsPage />);
  await act(async () => {});
  const retrieved = view.container.querySelector('.logs-footer')!.textContent;
  vi.mocked(runtimeApi.runtimeLogs).mockRejectedValue(new Error('events offline'));
  vi.mocked(runtimeApi.runtimeMetrics).mockRejectedValue(new Error('metrics offline'));
  await act(async () => { await vi.advanceTimersByTimeAsync(8000); });
  expect(view.container.querySelector('.logs-footer')).toHaveTextContent(retrieved!);
  expect(screen.getByText('PLE failed')).toBeInTheDocument();
  expect(screen.getByRole('status')).toHaveTextContent('events offline');
});

it('does not start backlog catch-up against a new server after the old query resolves', async () => {
  vi.useFakeTimers();
  const view = render(<LogsPage />);
  await act(async () => {});
  let resolve!: (entries: RuntimeLogEntry[]) => void;
  vi.mocked(runtimeApi.runtimeLogs).mockReturnValueOnce(new Promise((done) => { resolve = done; }));
  await act(async () => { await vi.advanceTimersByTimeAsync(4000); });
  state.connectionRevision = 1;
  vi.mocked(runtimeApi.runtimeLogs).mockResolvedValue([event(1004, 'info', 'New service')]);
  view.rerender(<LogsPage />);
  await act(async () => {});
  await act(async () => { resolve(Array.from({ length: 500 }, (_, i) => event(i + 4, 'info', `old-${i + 4}`))); });
  expect(runtimeApi.runtimeLogs).toHaveBeenCalledTimes(3);
  expect(screen.getByText('New service')).toBeInTheDocument();
  expect(screen.queryByText('old-503')).not.toBeInTheDocument();
});

it('supports keyboard tab navigation without tabbing through inactive tabs', async () => {
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  const events = screen.getByRole('tab', { name: /Service events/ });
  fireEvent.keyDown(events, { key: 'ArrowRight' });
  const requests = screen.getByRole('tab', { name: /Requests/ });
  expect(requests).toHaveAttribute('aria-selected', 'true');
  expect(requests).toHaveFocus();
  expect(events).toHaveAttribute('tabindex', '-1');
});

it('freezes live job changes while paused and refreshes them on resuming', async () => {
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.click(screen.getByRole('button', { name: 'Pause updates' }));
  await act(async () => { useJobStore.getState().setJobs([{ ...job('new-job'), kind: 'model.unload' }]); });
  fireEvent.click(screen.getByRole('tab', { name: /Jobs/ }));
  expect(screen.getByText('Load model')).toBeInTheDocument();
  expect(screen.queryByText('Unload model')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Resume updates' }));
  await screen.findByText('Unload model');
});

it('refreshes job history from the server on manual refresh while remaining paused', async () => {
  state.refreshRuntime.mockImplementationOnce(async () => {
    useJobStore.getState().setJobs([{ ...job('new-job'), kind: 'model.unload' }]);
    return true;
  });
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.click(screen.getByRole('button', { name: 'Pause updates' }));
  fireEvent.click(screen.getByRole('tab', { name: /Jobs/ }));
  expect(screen.getByText('Load model')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  await screen.findByText('Unload model');
  expect(state.refreshRuntime).toHaveBeenCalledTimes(1);
  expect(screen.getByText('Paused')).toBeInTheDocument();
});

it('reports a manual job-history refresh failure without discarding existing records', async () => {
  state.refreshError = 'job history offline';
  state.refreshRuntime.mockResolvedValue(false);
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  await screen.findByText(/job history offline/);
  expect(screen.getByText('PLE failed')).toBeInTheDocument();
  expect(screen.getByText('Refresh failed')).toBeInTheDocument();
});

it('does not mislabel a superseded runtime refresh as a server failure', async () => {
  state.refreshRuntime.mockResolvedValue(false);
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  await waitFor(() => expect(runtimeApi.runtimeLogs).toHaveBeenCalledTimes(2));
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
  expect(screen.queryByText('Refresh failed')).not.toBeInTheDocument();
  expect(screen.getByText('PLE failed')).toBeInTheDocument();
});

it('coalesces repeated manual refresh clicks while server job history is pending', async () => {
  let resolve!: (success: boolean) => void;
  state.refreshRuntime.mockReturnValueOnce(new Promise((done) => { resolve = done; }));
  render(<LogsPage />);
  await screen.findByText('PLE failed');
  const button = screen.getByRole('button', { name: 'Refresh' });
  fireEvent.click(button); fireEvent.click(button);
  expect(state.refreshRuntime).toHaveBeenCalledTimes(1);
  expect(button).toBeDisabled();
  await act(async () => { resolve(true); });
  await waitFor(() => expect(button).toBeEnabled());
});
