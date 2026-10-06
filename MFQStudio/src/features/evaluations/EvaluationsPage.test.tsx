/** Verify evaluation results refresh after a relevant background task finishes. */
import { i18n } from '../../i18n';
import { act, render, screen, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { beforeEach, expect, it, vi } from 'vitest';
import { EvaluationsPage } from './EvaluationsPage';
import { evaluationsApi } from '../../shared/api/resources/evaluations';
import { useJobStore } from '../../stores/jobStore';
import type { EvaluationResult, JobResource } from '../../shared/api/types';

vi.mock('../../shared/api/resources/evaluations', () => ({
  evaluationsApi: { datasets: vi.fn(), evaluations: vi.fn() },
}));
vi.mock('../settings/SettingsProvider', () => ({
  useSettings: () => ({ t: i18n.getFixedT('en') }),
}));
vi.mock('../../app/PanelDeck', () => ({
  PanelDeck: ({ children }: { children: ReactNode }) => <>{children}</>,
}));
vi.mock('../../app/ModelVendorMark', () => ({ ModelVendorMark: () => null }));

beforeEach(() => {
  useJobStore.getState().setJobs([]);
  vi.mocked(evaluationsApi.datasets).mockReset().mockResolvedValue([]);
  vi.mocked(evaluationsApi.evaluations).mockReset();
});

it('reloads results when an evaluation job succeeds while the page is open', async () => {
  const result = {
    id: 'result-1', job_id: 'job-1', model_id: 'model-a', kind: 'perplexity',
    created_at: '2026-10-05T00:00:00Z', metrics: { perplexity: 2.5 },
    parameters: {}, dataset_manifest: {}, hardware_identity: {}, runtime_identity: {},
    comparison_key: 'comparison-1',
  } satisfies EvaluationResult;
  vi.mocked(evaluationsApi.evaluations)
    .mockResolvedValueOnce([])
    .mockResolvedValue([result]);
  useJobStore.getState().setJobs([{
    id: 'job-1', kind: 'evaluate.perplexity', status: 'running',
  } as JobResource]);

  render(<EvaluationsPage />);
  await waitFor(() => expect(evaluationsApi.evaluations).toHaveBeenCalledTimes(1));
  expect(screen.getByText('No evaluation results yet. Register a dataset and run an evaluation job first.')).toBeInTheDocument();

  act(() => useJobStore.getState().updateJob('job-1', { status: 'succeeded' }));
  await waitFor(() => expect(evaluationsApi.evaluations).toHaveBeenCalledTimes(2));
  expect(await screen.findByText('model-a')).toBeInTheDocument();
});
