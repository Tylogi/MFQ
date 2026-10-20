import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { RuntimeStatus, ServingStatistics } from '../../shared/api/contracts/runtime';
import { OverviewPage } from './OverviewPage';

const state = vi.hoisted(() => ({ runtime: null as RuntimeStatus | null, models: [], instances: [],
  connectionRevision: 0, refreshRuntime: vi.fn(), listener: vi.fn().mockResolvedValue({ anthropic_port: null }),
  data: null as ServingStatistics | null, scope: 'session' }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => state }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('../../shared/api/resources/runtime', () => ({ runtimeApi: { runtimeListener: state.listener } }));
vi.mock('./RuntimeHero', () => ({ RuntimeHero: () => null }));
vi.mock('./MemoryHierarchy', () => ({ MemoryHierarchy: () => null }));
vi.mock('./useServingStatistics', () => ({ useServingStatistics: (scope: string) => { state.scope = scope; return { data: state.data, error: null }; } }));

beforeEach(() => { state.runtime = null; state.models = []; state.data = null; });
afterEach(() => { vi.restoreAllMocks(); });

it('uses placeholders without statistics, not unsupported or old request speed', async () => {
  state.runtime = { model: 'model', last_request: { decode_tps: 987.6 }, prefix_cache_supported: 0 };
  await act(async () => { render(<OverviewPage />); });
  expect(screen.getByText('Prefix reuse').closest('.metric-tile')).toHaveTextContent('--');
  expect(screen.queryByText('987.6')).not.toBeInTheDocument();
  expect(screen.queryByText('Live performance')).not.toBeInTheDocument();
  expect(screen.queryByText('Processed tokens / total prefill time')).not.toBeInTheDocument();
  expect(screen.queryByText('Generated tokens / total decode time')).not.toBeInTheDocument();
  expect(screen.getByText('Concurrency').closest('.metric-tile')).toHaveTextContent('--');
});

it('switches session and all-time and displays weighted completed statistics', async () => {
  vi.spyOn(Date, 'now').mockReturnValue(225000);
  state.runtime = { failed_requests: 3, active_requests: 99, uptime_seconds: 10 };
  state.data = { scope: 'session', started_at: 100, activity: [], totals: { completed_requests: 4,
    prompt_tokens: 400, prefill_tokens: 300, completion_tokens: 80, cached_tokens: 100,
    prefill_tps: 1234, decode_tps: 60, ttft_ms: 100, cache_hit_rate: 25 } };
  let view!: ReturnType<typeof render>;
  await act(async () => { view = render(<OverviewPage />); });
  expect(screen.getByText('Prefill').closest('.metric-tile')).toHaveTextContent('1,234');
  expect(screen.getByText('Prefix reuse').closest('.metric-tile')).toHaveTextContent('25%');
  expect(screen.getByText('Concurrency').closest('.metric-tile')?.querySelector('strong')).toHaveTextContent(/^0$/);
  const footer = view.container.querySelector('.overview-footer-grid')!;
  expect(footer.querySelector('.overview-endpoint-panel')).not.toBeNull();
  const summary = footer.querySelector('.overview-session-panel') as HTMLElement;
  expect(within(summary).getByRole('heading', { name: 'Session' })).toBeInTheDocument();
  expect(within(summary).getByText('Completed').previousElementSibling).toHaveTextContent('4');
  expect(within(summary).getByText('Input').previousElementSibling).toHaveTextContent('400');
  expect(within(summary).getByText('Output').previousElementSibling).toHaveTextContent('80');
  expect(within(summary).getByTitle('Session uptime')).toHaveTextContent('2m 5s');
  expect(summary.querySelector('p')).toHaveTextContent('3 failed requests · 0 active');
  state.data = { ...state.data!, scope: 'alltime', totals: { ...state.data!.totals,
    completed_requests: 10, prompt_tokens: 800, completion_tokens: 200 } };
  fireEvent.click(screen.getByRole('button', { name: 'All time' }));
  expect(state.scope).toBe('alltime');
  expect(screen.getByRole('button', { name: 'All time' })).toHaveAttribute('aria-pressed', 'true');
  expect(within(summary).getByRole('heading', { name: 'All time' })).toBeInTheDocument();
  expect(within(summary).getByText('Completed').previousElementSibling).toHaveTextContent('10');
  expect(within(summary).getByText('Input').previousElementSibling).toHaveTextContent('800');
  expect(within(summary).getByText('Output').previousElementSibling).toHaveTextContent('200');
  expect(within(summary).getByTitle('Session uptime')).toHaveTextContent('2m 5s');
  expect(summary.querySelector('p')).toHaveTextContent('3 failed requests · 0 active');
  expect(screen.getByText('Concurrency').closest('.metric-tile')?.querySelector('strong')).toHaveTextContent(/^0$/);
});

it('shows each current request separately from aggregate speed', async () => {
  state.runtime = { failed_requests: 0 };
  state.data = { scope: 'session', started_at: 100, totals: { completed_requests: 0,
    prompt_tokens: 0, prefill_tokens: 0, completion_tokens: 0, cached_tokens: 0,
    prefill_tps: null, decode_tps: null, ttft_ms: null, cache_hit_rate: null },
    activity: [{ instance_id: 'instance', model: 'model', state: 'busy', active_requests: 2,
      queued_requests: 1, sample_age_seconds: 0, requests: [
        { id: 'request-one', phase: 'generating', processed_tokens: 4096, generated_tokens: 128,
          prefill_tps: 2500, decode_tps: 68.2, elapsed_seconds: 3, updated_at: 100 },
        { id: 'request-two', phase: 'prefilling', processed_tokens: 2048, generated_tokens: 0,
          prefill_tps: 2400, decode_tps: 0, elapsed_seconds: 1, updated_at: 100 }] }] };
  await act(async () => { render(<OverviewPage />); });
  expect(screen.getByText('Generating').closest('.serving-request')).toHaveTextContent('68.2 tok/s');
  expect(screen.getByText('Prefilling').closest('.serving-request')).toHaveTextContent('2,048 tokens');
  expect(screen.getByText('Concurrency').closest('.metric-tile')?.querySelector('strong')).toHaveTextContent(/^2$/);
  expect(screen.getByText('Concurrency').closest('.metric-tile')).toHaveTextContent('1 queued');
  expect(screen.getByText('Prefilling').closest('.serving-request')).toHaveTextContent('2,400 tok/s');
  expect(screen.getByText('Decode').closest('.metric-tile')).toHaveTextContent('--');
  expect(screen.getByText('0 failed requests · 2 active')).toBeInTheDocument();
});
