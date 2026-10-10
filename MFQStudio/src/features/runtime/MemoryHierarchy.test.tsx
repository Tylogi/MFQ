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
      ssd_kv: true, ssd_kv_bytes: bytes,
    } };
}

it('shows prefix usage against the combined quota without adding it to KV twice', () => {
  const first = model('1', 1024), second = model('2', 3072);
  first.memory!.prefix_cache_bytes = 256;
  second.memory!.prefix_cache_bytes = 768;
  first.memory!.prefix_cache_limit_bytes = 4096;
  second.memory!.prefix_cache_limit_bytes = 4096;
  const { container } = render(<MemoryHierarchy instances={[first, second]} memoryCapacityBytes={32768} />);
  expect(container.querySelector('.memory-tier-prefix-quota')).toHaveTextContent('1 KiB / 8 KiB');
  expect(container.querySelector('[data-tier="kv"]')).toHaveTextContent('4 KiB / 28 KiB');
});

it('reports Metal wiring separately from weight size and warns on unavailable wiring', () => {
  const instance = model('1', 1024);
  Object.assign(instance.memory!, { wired_available: true, wired_bytes: 2048, wired_limit_bytes: 4096 });
  const { container, rerender } = render(<MemoryHierarchy instances={[instance]} memoryCapacityBytes={8192} />);
  expect(screen.getByTitle('Model 1 · Metal wired 2 KiB / 4 KiB')).toBeInTheDocument();
  expect(container.querySelector('[data-tier="weights"]')).toHaveTextContent('1 KiB / 8 KiB');
  instance.memory!.wired_available = false;
  rerender(<MemoryHierarchy instances={[instance]} memoryCapacityBytes={8192} />);
  expect(screen.getByText(/weights may be paged out/)).toBeInTheDocument();
  instance.memory!.wired_available = null;
  rerender(<MemoryHierarchy instances={[instance]} memoryCapacityBytes={8192} />);
  expect(screen.queryByText(/weights may be paged out/)).not.toBeInTheDocument();
  expect(screen.getByTitle('Model 1')).toBeInTheDocument();
});

it('shows a dotted model legend even with one loaded model and five resource tiers', () => {
  const { container } = render(<MemoryHierarchy instances={[model('1', 1024)]} memoryCapacityBytes={4096} />);
  expect(screen.getByText('Runtime resources')).toBeInTheDocument();
  expect(screen.getByLabelText('Model color legend').querySelector('i')).toHaveStyle({ backgroundColor: 'var(--accent)' });
  expect(container.querySelectorAll('[data-tier]')).toHaveLength(5);
  expect(screen.getByText('1 contexts · 2 cache blocks')).toBeInTheDocument();
  expect(screen.queryByText(/allocator/i)).not.toBeInTheDocument();
  expect(screen.getAllByText('1 model')).toHaveLength(4);
  expect(screen.getAllByText('1 KiB')).toHaveLength(3);
  expect(screen.getByText('1 KiB / 4 KiB')).toBeInTheDocument();
  expect(screen.getByText('1 KiB / 3 KiB')).toBeInTheDocument();
  expect(container.querySelector('[data-tier="weights"] .memory-tier-track > span')).toHaveStyle({ width: '25%' });
  expect(parseFloat((container.querySelector('[data-tier="kv"] .memory-tier-track > span') as HTMLElement).style.width))
    .toBeCloseTo(100 / 3);
  expect(screen.queryByText(/Colors show each model/)).not.toBeInTheDocument();
});

it('shares load-order colors, uses memory capacity for resident tiers and byte shares for SSD', () => {
  const models = [model('1', 1024), model('2', 3072), model('3', 4096), model('4', 8192)];
  const { container } = render(<MemoryHierarchy instances={[...models].reverse()} memoryCapacityBytes={32768} />);
  const legend = container.querySelector('.memory-model-legend')!;
  const colors = models.map((item) => (legend.querySelector(`[data-model-id="${item.id}"] i`) as HTMLElement).style.backgroundColor);
  expect(new Set(colors).size).toBe(4);
  expect(colors[0]).toBe('var(--accent)');
  for (const tier of container.querySelectorAll('.memory-tier-track')) {
    expect([...tier.children].map((item) => (item as HTMLElement).style.backgroundColor)).toEqual(colors);
  }
  expect(container.querySelector('[data-tier="weights"] [data-model-id="1"]')).toHaveStyle({ width: '3.125%' });
  for (const tier of ['kv', 'experts', 'ple', 'ssd-kv']) {
    expect(container.querySelector(`[data-tier="${tier}"] [data-model-id="1"]`)).toHaveStyle({ width: '6.25%' });
  }
  expect(screen.getByText('16 KiB / 32 KiB')).toBeInTheDocument();
  expect(screen.getByText('16 KiB / 16 KiB')).toBeInTheDocument();
});

it('keeps remaining model colors after unload/reordering and reuses a reloaded model color', () => {
  const first = model('1', 1024), second = model('2', 1024);
  const { container, rerender } = render(<MemoryHierarchy instances={[first, second]} memoryCapacityBytes={4096} />);
  const red = (container.querySelector('[data-tier="weights"] [data-model-id="2"]') as HTMLElement).style.backgroundColor;
  rerender(<MemoryHierarchy instances={[second]} memoryCapacityBytes={4096} />);
  expect(container.querySelector('[data-tier="kv"] [data-model-id="2"]')).toHaveStyle({ backgroundColor: red });
  expect(screen.getByText('1 KiB / 3 KiB')).toBeInTheDocument();
  rerender(<MemoryHierarchy instances={[second, { ...first, id: 'reload' }]} memoryCapacityBytes={4096} />);
  expect(container.querySelector('[data-tier="ple"] [data-model-id="reload"]')).toHaveStyle({ backgroundColor: 'var(--accent)' });
  expect(screen.getByText('2 KiB / 2 KiB')).toBeInTheDocument();
});

it('excludes failed/loading models and distinguishes unavailable telemetry from zero', () => {
  const zero = model('1', 0);
  const { container, rerender } = render(<MemoryHierarchy instances={[
    zero, { ...model('2', 500), state: 'failed' }, { ...model('3', 500), state: 'loading' },
  ]} memoryCapacityBytes={4096} />);
  expect(screen.queryByText('Model 2')).not.toBeInTheDocument();
  expect(screen.queryByText('Model 3')).not.toBeInTheDocument();
  expect(screen.getAllByText('0 B')).toHaveLength(3);
  expect(screen.getAllByText('0 B / 4 KiB')).toHaveLength(2);
  rerender(<MemoryHierarchy instances={[{ ...zero, memory: null }]} memoryCapacityBytes={4096} />);
  expect(screen.getAllByText('Model 1: breakdown not reported')).toHaveLength(5);
  expect(screen.getByText('Breakdown not reported / --')).toBeInTheDocument();
  expect(container.querySelectorAll('.memory-tier-track > span')).toHaveLength(0);
  expect(screen.getAllByText('0 models · 1 not reported')).toHaveLength(3);
});

it('does not allocate SSD segments for full-resident weights or absent PLE', () => {
  const item = model('1', 1024);
  item.memory = { ...item.memory!, ssd_experts: false, ssd_ple: false };
  const { container } = render(<MemoryHierarchy instances={[item]} memoryCapacityBytes={4096} />);
  expect(container.querySelector('[data-tier="experts"] .memory-tier-track')?.children).toHaveLength(0);
  expect(container.querySelector('[data-tier="ple"] .memory-tier-track')?.children).toHaveLength(0);
});

it('refreshes numeric limits and segment widths when detected usable memory changes', () => {
  const item = model('1', 1024);
  const { container, rerender } = render(<MemoryHierarchy instances={[item]} memoryCapacityBytes={4096} />);
  rerender(<MemoryHierarchy instances={[item]} memoryCapacityBytes={8192} />);
  expect(screen.getByText('1 KiB / 8 KiB')).toBeInTheDocument();
  expect(screen.getByText('1 KiB / 7 KiB')).toBeInTheDocument();
  expect(container.querySelector('[data-tier="weights"] .memory-tier-track > span')).toHaveStyle({ width: '12.5%' });
});

it('does not invent a full resident bar when the limit or some weight residency is unknown', () => {
  const first = model('1', 1024);
  const { container, rerender } = render(<MemoryHierarchy instances={[first]} />);
  expect(screen.getAllByText('1 KiB / --')).toHaveLength(2);
  expect(container.querySelectorAll('[data-tier="weights"] .memory-tier-track > span, [data-tier="kv"] .memory-tier-track > span'))
    .toHaveLength(0);
  expect(container.querySelector('[data-tier="ple"] .memory-tier-track > span')).toHaveStyle({ width: '100%' });
  rerender(<MemoryHierarchy instances={[first, { ...model('2', 1024), memory: null }]} memoryCapacityBytes={4096} />);
  expect(screen.getByText('≥ 1 KiB / 4 KiB')).toBeInTheDocument();
  expect(screen.getByText('≥ 1 KiB / --')).toBeInTheDocument();
  expect(screen.getAllByText('Model 2: breakdown not reported')).toHaveLength(5);
  expect(container.querySelector('[data-tier="weights"] .memory-tier-track > span')).toHaveStyle({ width: '25%' });
  expect(container.querySelector('[data-tier="kv"] .memory-tier-track')?.children).toHaveLength(0);
});

it('clamps exhausted KV capacity to zero without changing numeric limits or producing invalid widths', () => {
  const first = model('1', 3072), second = model('2', 1024);
  const { container, rerender } = render(<MemoryHierarchy instances={[first, second]} memoryCapacityBytes={2048} />);
  expect(screen.getByText('4 KiB / 2 KiB')).toBeInTheDocument();
  expect(screen.getByText('4 KiB / 0 B')).toBeInTheDocument();
  for (const tier of ['weights', 'kv']) {
    expect(container.querySelector(`[data-tier="${tier}"] [data-model-id="1"]`)).toHaveStyle({ width: '75%' });
    expect(container.querySelector(`[data-tier="${tier}"] [data-model-id="2"]`)).toHaveStyle({ width: '25%' });
  }
  rerender(<MemoryHierarchy instances={[model('1', 0)]} memoryCapacityBytes={0} />);
  expect(screen.getAllByText('0 B / 0 B')).toHaveLength(2);
  expect(container.querySelectorAll('.memory-tier-track > span')).toHaveLength(0);
});

it('expands the same five tiers into per-model sizes, active KV, hot prefixes and wiring', () => {
  const first = model('1', 1024), second = model('2', 3072);
  Object.assign(first.memory!, { prefix_cache_bytes: 256, prefix_cache_limit_bytes: 2048,
    wired_available: true, wired_bytes: 1024, ssd_experts: false, ssd_ple: false });
  Object.assign(second.memory!, { prefix_cache_bytes: 768, prefix_cache_limit_bytes: 4096 });
  const { container } = render(<MemoryHierarchy instances={[first, second]} memoryCapacityBytes={32768} detailed />);
  expect(screen.getByText('Resource allocation')).toBeInTheDocument();
  expect(container.querySelectorAll('.resource-tier-model')).toHaveLength(10);
  expect(container.querySelector('[data-tier="kv"]')).toHaveTextContent('4 KiB / 28 KiB');
  expect(container.querySelector('[data-tier="kv"] .resource-tier-model')).toHaveTextContent('Active KV 768 B · Prefix RAM 256 B / 2 KiB');
  expect(container.querySelector('[data-tier="weights"] .resource-tier-model')).toHaveTextContent('METAL · Wired 1 KiB');
  expect(container.querySelector('[data-tier="experts"] .resource-tier-model')).toHaveTextContent('All experts resident');
  expect(container.querySelector('[data-tier="ple"] .resource-tier-model')).toHaveTextContent('No SSD PLE table');
  for (const tier of container.querySelectorAll('.resource-tier-details')) {
    expect(tier.querySelector('.resource-tier-model-name i')).toHaveStyle({ backgroundColor: 'var(--accent)' });
  }
});

it('detailed unknown telemetry never becomes fabricated zero sizes or active KV', () => {
  const { container } = render(<MemoryHierarchy instances={[{ ...model('1', 0), memory: null }]} detailed />);
  for (const amount of container.querySelectorAll('.resource-tier-model > strong')) expect(amount).toHaveTextContent('Not reported');
  expect(container.querySelector('[data-tier="kv"] .resource-tier-model')).toHaveTextContent('Active KV -- · Prefix RAM -- / --');
});

it('tracks actual streamed KV storage without adding SSD bytes or traffic to resident KV', () => {
  const first = model('1', 1024), second = model('2', 3072);
  Object.assign(first.memory!, { ssd_kv_bytes: 1024, ssd_kv_read_bytes: 4096, ssd_kv_written_bytes: 8192,
    ssd_kv_reads: 1, ssd_kv_hits: 3, streaming_kv_resident_bytes: 512, streaming_kv_budget_bytes: 2048,
    streaming_kv_pending_bytes: 128 });
  Object.assign(second.memory!, { ssd_kv_bytes: 3072, ssd_kv_read_bytes: 8192, ssd_kv_written_bytes: 16384,
    ssd_kv_reads: 3, ssd_kv_hits: 1 });
  const view = render(<MemoryHierarchy instances={[first, second]} memoryCapacityBytes={32768} />);
  const tier = view.container.querySelector('[data-tier="ssd-kv"]')!;
  expect(tier).toHaveTextContent('4 KiB');
  expect(tier).toHaveTextContent('Total read 12 KiB');
  expect(tier).toHaveTextContent('Total written 24 KiB');
  expect(tier).toHaveTextContent('RAM hit rate 50%');
  expect(tier.querySelector('[data-model-id="1"]')).toHaveStyle({ width: '25%', backgroundColor: 'var(--accent)' });
  expect(view.container.querySelector('[data-tier="kv"]')).toHaveTextContent('4 KiB / 28 KiB');
  view.rerender(<MemoryHierarchy instances={[first, second]} detailed memoryCapacityBytes={32768} />);
  const detail = view.container.querySelector('[data-tier="ssd-kv"] .resource-tier-model')!;
  expect(detail).toHaveTextContent('Resident 512 B / 2 KiB · Pending write 128 B');
  expect(detail).toHaveTextContent('RAM hit rate 75%');
  Object.assign(first.memory!, { ssd_kv: false });
  Object.assign(second.memory!, { ssd_kv: false });
  view.rerender(<MemoryHierarchy instances={[first, second]} memoryCapacityBytes={32768} />);
  expect(view.container.querySelector('[data-tier="ssd-kv"]')).toHaveTextContent('0 models');
  expect(view.container.querySelector('[data-tier="ssd-kv"] .memory-tier-track')?.children).toHaveLength(0);
  expect(view.container.querySelector('[data-tier="ssd-kv"] .memory-tier-streaming-summary')).toBeNull();
});

it('preserves independently reported context and cache-block counts', () => {
  const first = model('1', 1024), second = model('2', 1024);
  first.memory!.context_count = null;
  const { rerender } = render(<MemoryHierarchy instances={[first, second]} detailed />);
  expect(screen.getByText('-- contexts · 4 cache blocks')).toBeInTheDocument();
  first.memory!.context_count = 1;
  second.memory!.prefix_cache_blocks = null;
  rerender(<MemoryHierarchy instances={[first, second]} detailed />);
  expect(screen.getByText('2 contexts · -- cache blocks')).toBeInTheDocument();
});

it('retains unloading model allocations until the backend confirms removal', () => {
  const first = model('1', 1024), second = model('2', 2048);
  const { container, rerender } = render(<MemoryHierarchy instances={[first, second]} memoryCapacityBytes={8192} detailed />);
  const previousColor = (container.querySelector('[data-tier="weights"] [data-model-id="2"]') as HTMLElement).style.backgroundColor;
  const unloading: RuntimeInstance = { ...second, state: 'unloading',
    error: { code: 'runtime_unload_failed', message: 'process is still alive', retryable: true, details: {} } };
  rerender(<MemoryHierarchy instances={[first, unloading]} memoryCapacityBytes={8192} detailed />);
  expect(container.querySelector('[data-tier="weights"]')).toHaveTextContent('3 KiB / 8 KiB');
  expect(container.querySelector('[data-tier="kv"]')).toHaveTextContent('3 KiB / 5 KiB');
  expect(container.querySelectorAll('.resource-tier-model[data-model-id="2"]')).toHaveLength(5);
  expect(container.querySelector('[data-tier="weights"] [data-model-id="2"]')).toHaveStyle({ backgroundColor: previousColor });
  expect(screen.getByText('Model 2 · Unloading')).toBeInTheDocument();
  expect(screen.getByText('Unloading models retain their last reported allocation until release is confirmed.')).toBeInTheDocument();
  rerender(<MemoryHierarchy instances={[first]} memoryCapacityBytes={8192} detailed />);
  expect(container.querySelector('[data-tier="weights"]')).toHaveTextContent('1 KiB / 8 KiB');
  expect(container.querySelector('[data-tier="kv"]')).toHaveTextContent('1 KiB / 7 KiB');
  expect(container.querySelector('[data-model-id="2"]')).not.toBeInTheDocument();
  expect(screen.queryByText(/Unloading models retain/)).not.toBeInTheDocument();
});

it('does not fabricate zero usage for an unloading model without a reported allocation', () => {
  const item: RuntimeInstance = { ...model('1', 1024), state: 'unloading', memory: null };
  render(<MemoryHierarchy instances={[item]} memoryCapacityBytes={8192} detailed />);
  expect(screen.getByText('Model 1 · Unloading')).toBeInTheDocument();
  expect(screen.getByText('Breakdown not reported / 8 KiB')).toBeInTheDocument();
  expect(screen.getByText('Breakdown not reported / --')).toBeInTheDocument();
  expect(screen.queryByText('0 B / 8 KiB')).not.toBeInTheDocument();
});
