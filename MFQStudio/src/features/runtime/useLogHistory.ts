import { useEffect, useRef, useState } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import { errorMessage } from '../../app/formatters';
import { runtimeApi } from '../../shared/api/resources/runtime';
import type { JobResource, RuntimeLogEntry, RuntimeMetricSnapshot, RuntimeRequestMetrics } from '../../shared/api/types';
import { useJobStore } from '../../stores/jobStore';

export interface RequestLogRecord {
  key: string;
  instance_id?: string | null;
  model?: string | null;
  captured_at: string;
  request: RuntimeRequestMetrics;
}

export function mergeRequestRecords(records: RequestLogRecord[]): RequestLogRecord[] {
  const merged = new Map<string, RequestLogRecord>();
  for (const record of records) {
    const previous = merged.get(record.key);
    const request = { ...previous?.request, ...record.request };
    if (previous?.request.completed_at != null && record.request.completed_at != null)
      request.completed_at = Math.max(previous.request.completed_at, record.request.completed_at);
    merged.set(record.key, { ...record, model: record.model ?? previous?.model,
      request });
  }
  return [...merged.values()].sort((a, b) =>
    (b.request.completed_at != null ? b.request.completed_at * 1000 : Date.parse(b.captured_at))
      - (a.request.completed_at != null ? a.request.completed_at * 1000 : Date.parse(a.captured_at)));
}

export function requestRecords(snapshots: RuntimeMetricSnapshot[]): RequestLogRecord[] {
  const records: RequestLogRecord[] = [];
  for (const snapshot of [...snapshots].sort((a, b) => a.sequence - b.sequence)) {
    const request = snapshot.values.last_request;
    if (!request?.id) continue;
    const key = `${snapshot.instance_id || ''}:${request.id}`;
    records.push({ key, instance_id: snapshot.instance_id, model: snapshot.model,
      captured_at: snapshot.captured_at, request });
  }
  return mergeRequestRecords(records);
}

export function useLogHistory(paused: boolean) {
  const { ready, connectionRevision } = useRuntime();
  const jobs = useJobStore((state) => state.jobs);
  const [logs, setLogs] = useState<RuntimeLogEntry[]>([]);
  const [requests, setRequests] = useState<RequestLogRecord[]>([]);
  const [jobSnapshot, setJobSnapshot] = useState<JobResource[]>(jobs);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const manual = useRef(0);
  const cursor = useRef<{ log?: number; metrics?: string }>({});
  useEffect(() => {
    cursor.current = {};
    setLogs([]); setRequests([]); setJobSnapshot([]); setError(''); setUpdatedAt(null);
  }, [ready, connectionRevision]);
  useEffect(() => { if (!paused && ready) setJobSnapshot(jobs); }, [jobs, paused, ready]);
  useEffect(() => {
    const forced = manual.current !== refreshVersion;
    manual.current = refreshVersion;
    if (!ready || (paused && !forced)) { setLoading(false); return; }
    let disposed = false;
    let polling = false;
    let manualPending = forced;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();
    async function poll() {
      if (disposed || polling) return;
      clearTimeout(timer);
      if (document.hidden) {
        if (!paused) timer = setTimeout(() => void poll(), 4000);
        return;
      }
      polling = true;
      manualPending = false;
      setLoading(true);
      try {
        const after = cursor.current.log;
        const result = await Promise.allSettled([
          runtimeApi.runtimeLogs(500, controller.signal, after).then((entries) =>
            !disposed && after != null && entries.length === 500
              ? runtimeApi.runtimeLogs(500, controller.signal) : entries),
          runtimeApi.runtimeMetrics(500, controller.signal, cursor.current.metrics),
        ]);
        if (disposed) return;
        const [events, metrics] = result;
        if (events.status === 'fulfilled' && events.value.length) {
          cursor.current.log = Math.max(...events.value.map((entry) => entry.sequence));
          setLogs((previous) => [...new Map([...previous, ...events.value].map((entry) => [entry.sequence, entry])).values()]
            .sort((a, b) => b.sequence - a.sequence).slice(0, 500));
        }
        if (metrics.status === 'fulfilled' && metrics.value.length) {
          cursor.current.metrics = [...metrics.value].sort((a, b) => b.sequence - a.sequence)[0].captured_at;
          const next = requestRecords(metrics.value);
          setRequests((previous) => mergeRequestRecords([...previous, ...next]).slice(0, 500));
        }
        setJobSnapshot(useJobStore.getState().jobs);
        setError(result.filter((item) => item.status === 'rejected').map((item) => errorMessage(item.reason)).join(' · '));
        if (result.some((item) => item.status === 'fulfilled')) setUpdatedAt(Date.now());
      } finally {
        polling = false;
        if (!disposed) {
          setLoading(false);
          if (!paused) timer = setTimeout(() => void poll(), 4000);
        }
      }
    }
    function visible() { if (!document.hidden && (!paused || manualPending)) void poll(); }
    document.addEventListener('visibilitychange', visible);
    void poll();
    return () => { disposed = true; controller.abort(); clearTimeout(timer); document.removeEventListener('visibilitychange', visible); };
  }, [ready, connectionRevision, paused, refreshVersion]);
  return { logs, requests, jobs: jobSnapshot, loading, error, updatedAt, ready,
    refresh: () => setRefreshVersion((version) => version + 1) };
}
