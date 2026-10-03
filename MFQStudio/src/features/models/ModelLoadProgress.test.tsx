import { act, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { ModelLoadProgress } from './ModelLoadProgress';
import { useJobStore } from '../../stores/jobStore';
import type { JobResource } from '../../shared/api/types';

vi.mock('../settings/SettingsProvider', () => ({
  useSettings: () => ({ tr: (_zh: string, en: string) => en }),
}));

beforeEach(() => useJobStore.setState({ jobs: [] }));

function job(progress: number, status: JobResource['status'] = 'running'): JobResource {
  return { id: 'load', kind: 'model.load', payload: { model: 'test-model' }, progress, status,
    cancel_requested: false, created_at: '2026-01-01', updated_at: '2026-01-01' };
}

it('shows an indeterminate bar until the worker reports measurable progress', () => {
  useJobStore.setState({ jobs: [job(0, 'queued')] });
  render(<ModelLoadProgress model="test-model" />);
  expect(screen.getByRole('progressbar')).not.toHaveAttribute('value');
  expect(screen.getByText('Queued')).toBeInTheDocument();
  act(() => useJobStore.setState({ jobs: [job(0.02)] }));
  expect(screen.getByRole('progressbar')).not.toHaveAttribute('value');
  expect(screen.getByText('Preparing')).toBeInTheDocument();
  act(() => useJobStore.setState({ jobs: [job(0.25)] }));
  expect(screen.getByRole('progressbar')).toHaveAttribute('value', '0.25');
  expect(screen.getByText('25%')).toBeInTheDocument();
  act(() => useJobStore.setState({ jobs: [job(0.75)] }));
  expect(screen.getByRole('progressbar')).toHaveAttribute('value', '0.75');
  act(() => useJobStore.setState({ jobs: [job(1, 'succeeded')] }));
  expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
});

it('isolates models and hides terminal jobs', () => {
  useJobStore.setState({ jobs: [job(0.5)] });
  const { rerender } = render(<ModelLoadProgress model="other-model" />);
  expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
  rerender(<ModelLoadProgress model="test-model" />);
  expect(screen.getByRole('progressbar')).toHaveAttribute('value', '0.5');
  act(() => useJobStore.setState({ jobs: [job(0.5, 'cancelling')] }));
  expect(screen.getByText('Cancelling')).toBeInTheDocument();
  act(() => useJobStore.setState({ jobs: [job(0.5, 'failed')] }));
  expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
});

it('does not render invalid or excessive percentages', () => {
  useJobStore.setState({ jobs: [job(Number.NaN)] });
  render(<ModelLoadProgress model="test-model" />);
  expect(screen.getByRole('progressbar')).not.toHaveAttribute('value');
  act(() => useJobStore.setState({ jobs: [job(1.5)] }));
  expect(screen.getByRole('progressbar')).toHaveAttribute('value', '1');
});
