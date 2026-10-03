import { render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { MemoryHierarchy } from './MemoryHierarchy';
import type { RuntimeInstance } from '../../shared/api/types';

vi.mock('../settings/SettingsProvider', () => ({
  useSettings: () => ({ tr: (_zh: string, en: string) => en }),
}));

function model(id: string, bytes: number): RuntimeInstance {
  return { id, model: `Model ${id}`, state: 'ready', devices: ['metal'], active_sessions: 0,
    queued_requests: 0, started_at: `2026-01-01T00:00:0${id}Z`, memory: {
      resident_weight_bytes: bytes, kv_bytes: bytes, context_count: 1, prefix_cache_blocks: 2,
      ssd_experts: true, ssd_expert_bytes: bytes, ssd_ple: true, ssd_ple_bytes: bytes,
    } };
}

it('shows a dotted model legend even with one loaded model and four resource tiers', () => {
  const { container } = render(<MemoryHierarchy instances={[model('1', 1024)]} />);
  expect(screen.getByText('Runtime resources')).toBeInTheDocument();
  expect(screen.getByLabelText('Model color legend').querySelector('i')).toHaveStyle({ backgroundColor: 'var(--accent)' });
  expect(container.querySelectorAll('[data-tier]')).toHaveLength(4);
  expect(screen.getByText('1 contexts · 2 cache blocks')).toBeInTheDocument();
  expect(screen.queryByText(/allocator/i)).not.toBeInTheDocument();
});

it('shares load-order colors across all four tiers and uses each tier’s byte proportions', () => {
  const models = [model('1', 1024), model('2', 3072), model('3', 4096), model('4', 8192)];
  const { container } = render(<MemoryHierarchy instances={[...models].reverse()} />);
  const legend = container.querySelector('.memory-model-legend')!;
  const colors = models.map((item) => (legend.querySelector(`[data-model-id="${item.id}"] i`) as HTMLElement).style.backgroundColor);
  expect(new Set(colors).size).toBe(4);
  expect(colors[0]).toBe('var(--accent)');
  for (const tier of container.querySelectorAll('.memory-tier-track')) {
    expect([...tier.children].map((item) => (item as HTMLElement).style.backgroundColor)).toEqual(colors);
    expect((tier.querySelector('[data-model-id="1"]') as HTMLElement).style.width).toBe('6.25%');
  }
});

it('keeps remaining model colors after unload/reordering and reuses a reloaded model color', () => {
  const first = model('1', 1024), second = model('2', 1024);
  const { container, rerender } = render(<MemoryHierarchy instances={[first, second]} />);
  const red = (container.querySelector('[data-tier="weights"] [data-model-id="2"]') as HTMLElement).style.backgroundColor;
  rerender(<MemoryHierarchy instances={[second]} />);
  expect(container.querySelector('[data-tier="kv"] [data-model-id="2"]')).toHaveStyle({ backgroundColor: red });
  rerender(<MemoryHierarchy instances={[second, { ...first, id: 'reload' }]} />);
  expect(container.querySelector('[data-tier="ple"] [data-model-id="reload"]')).toHaveStyle({ backgroundColor: 'var(--accent)' });
});

it('excludes failed/loading models and distinguishes unavailable telemetry from zero', () => {
  const zero = model('1', 0);
  const { container, rerender } = render(<MemoryHierarchy instances={[
    zero, { ...model('2', 500), state: 'failed' }, { ...model('3', 500), state: 'loading' },
  ]} />);
  expect(screen.queryByText('Model 2')).not.toBeInTheDocument();
  expect(screen.queryByText('Model 3')).not.toBeInTheDocument();
  expect(screen.getAllByText('0 B')).toHaveLength(4);
  rerender(<MemoryHierarchy instances={[{ ...zero, memory: null }]} />);
  expect(screen.getAllByText('Breakdown not reported')).toHaveLength(4);
  expect(container.querySelectorAll('.memory-tier-track > span')).toHaveLength(0);
  expect(screen.getAllByText('0 models · 1 not reported')).toHaveLength(2);
});

it('does not allocate SSD segments for full-resident weights or absent PLE', () => {
  const item = model('1', 1024);
  item.memory = { ...item.memory!, ssd_experts: false, ssd_ple: false };
  const { container } = render(<MemoryHierarchy instances={[item]} />);
  expect(container.querySelector('[data-tier="experts"] .memory-tier-track')?.children).toHaveLength(0);
  expect(container.querySelector('[data-tier="ple"] .memory-tier-track')?.children).toHaveLength(0);
});
