import { describe, expect, it } from 'vitest';
import type { JobResource } from '../../shared/api/types';
import { downloadEtaSeconds } from './DownloadQueue';

const job: JobResource = { id: 'download', kind: 'download.modelscope', status: 'running', payload: {},
  progress: 0.25, cancel_requested: false, created_at: '', updated_at: '',
  progress_data: { downloaded_bytes: 25, total_bytes: 100, bytes_per_second: 10 } };

describe('downloadEtaSeconds', () => {
  it('uses remaining bytes and rounds fractional seconds up', () => {
    expect(downloadEtaSeconds(job)).toBe(8);
  });

  it.each(['queued', 'cancelling', 'succeeded', 'failed', 'cancelled', 'interrupted'] as const)(
    'does not predict for %s tasks', status => expect(downloadEtaSeconds({ ...job, status })).toBeNull());

  it.each([
    {}, { downloaded_bytes: 25 }, { downloaded_bytes: 25, total_bytes: 100 },
    { downloaded_bytes: 25, total_bytes: 100, bytes_per_second: 0 },
    { downloaded_bytes: 25, total_bytes: 100, bytes_per_second: 0.1 },
    { downloaded_bytes: -1, total_bytes: 100, bytes_per_second: 10 },
    { downloaded_bytes: 100, total_bytes: 100, bytes_per_second: 10 },
    { downloaded_bytes: 101, total_bytes: 100, bytes_per_second: 10 },
    { downloaded_bytes: 25, total_bytes: Infinity, bytes_per_second: 10 },
    { downloaded_bytes: NaN, total_bytes: 100, bytes_per_second: 10 },
    { downloaded_bytes: 25, total_bytes: 100, bytes_per_second: NaN },
  ])('rejects unavailable, stalled or invalid counters: %j', progress_data => {
    expect(downloadEtaSeconds({ ...job, progress_data })).toBeNull();
  });
});
