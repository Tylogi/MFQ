import { act, fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router';
import { beforeEach, expect, it, vi } from 'vitest';
import type { RuntimeStatus } from '../../shared/api/contracts/runtime';
import type { JobResource } from '../../shared/api/types';
import { useJobStore } from '../../stores/jobStore';
import { RuntimeHero } from './RuntimeHero';

const state = vi.hoisted(() => ({ runtime: null as RuntimeStatus | null, instances: [], studio: null, chinese: false }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => state }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (zh: string, en: string) => state.chinese ? zh : en }) }));

function Location() {
  const location = useLocation();
  return <output data-testid="location">{location.pathname}{location.hash}</output>;
}

beforeEach(() => { state.runtime = null; state.chinese = false; useJobStore.getState().setJobs([]); });

it.each([false, true])('opens local checkpoints from the empty-model Load action (Chinese: %s)', chinese => {
  state.chinese = chinese;
  render(<MemoryRouter><RuntimeHero /><Location /></MemoryRouter>);
  expect(screen.queryByRole('button', { name: chinese ? '模型' : 'Models' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: chinese ? '加载' : 'Load' }));
  expect(screen.getByTestId('location')).toHaveTextContent('/models#local-checkpoints');
});

it('retains the normal Models action when a model is running', () => {
  state.runtime = { model: 'model-a', runtime_state: 'ready' };
  render(<MemoryRouter><RuntimeHero /><Location /></MemoryRouter>);
  expect(screen.queryByRole('button', { name: 'Load' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Models' }));
  expect(screen.getByTestId('location').textContent).toBe('/models');
});

it('shows an interrupted load after runtime cleanup and lets a new retry show its progress', () => {
  const interrupted = { id: 'old-load', kind: 'model.load', status: 'interrupted', progress: .2,
    cancel_requested: false, payload: { model: 'fixture-model' },
    created_at: '2026-01-01', updated_at: '2026-01-01' } as JobResource;
  useJobStore.getState().setJobs([interrupted]);
  render(<MemoryRouter><RuntimeHero /></MemoryRouter>);
  expect(screen.getByText('Interrupted')).toBeInTheDocument();
  expect(screen.queryByText('Loading')).not.toBeInTheDocument();
  expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
  act(() => useJobStore.getState().setJobs([{ ...interrupted, id: 'new-load', status: 'running', progress: .25 }, interrupted]));
  expect(screen.getByText('Loading')).toBeInTheDocument();
  expect(screen.getByRole('progressbar')).toHaveAttribute('value', '0.25');
  expect(screen.queryByText('Interrupted')).not.toBeInTheDocument();
});
