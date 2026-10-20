import { fireEvent, render as baseRender, screen } from '@testing-library/react';
import type { ReactElement } from 'react';
import { beforeEach, expect, it, vi } from 'vitest';
import { MemoryHierarchy } from './MemoryHierarchy';
import { TooltipProvider } from '../../shared/ui/Tooltip';

beforeEach(() => {
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} });
});

function render(ui: ReactElement) { return baseRender(ui, { wrapper: TooltipProvider }); }

it('shows eight separate VRAM, RAM and SSD tiers without treating unknown RAM telemetry as zero', () => {
  const item = model('1', 1024);
  Object.assign(item.memory!, { ram_experts: true, ram_expert_bytes: 4096,
    ram_kv: true, ram_kv_bytes: 2048, ram_kv_limit_bytes: 8192, prefix_cache_bytes: 256, ssd_experts: false });
  const { container, rerender } = render(<MemoryHierarchy instances={[item]} runtime={{ memory_architecture: 'discrete' }} memoryCapacityBytes={32768} detailed />);
  expect(container.querySelectorAll('[data-tier]')).toHaveLength(8);
  expect([...container.querySelectorAll('[data-storage]')].map(group => group.getAttribute('data-storage'))).toEqual(['vram', 'ram', 'ssd']);
  expect(screen.getByRole('heading', { level: 3, name: 'VRAM · GPU 0' })).toBeInTheDocument();
  expect(screen.getByRole('heading', { level: 3, name: 'RAM' })).toBeInTheDocument();
  expect(container.querySelector('[data-storage="vram"] [data-tier="weights"]')).toBeInTheDocument();
  expect(container.querySelector('[data-storage="ram"] [data-tier="ram-experts"]')).toBeInTheDocument();
  expect(container.querySelector('[data-storage="ssd"] [data-tier="ple"]')).toBeInTheDocument();
  expect(screen.getByText('VRAM-resident weights and buffers')).toBeInTheDocument();
  expect(screen.getByText('RAM-streamed experts')).toBeInTheDocument();
  expect(screen.getByText('RAM-streamed KV')).toBeInTheDocument();
  expect(screen.getByText('Not yet supported on this device')).toBeInTheDocument();
  expect(container.querySelector('[data-tier="ram-experts"] .resource-tier-model strong')).toHaveTextContent('4 KiB');
  expect(container.querySelector('[data-tier="ram-kv"] .resource-tier-model strong')).toHaveTextContent('2 KiB');
  expect(container.querySelector('[data-tier="kv"]')).toHaveTextContent('Prefix VRAM');
  rerender(<MemoryHierarchy instances={[{ ...item, memory: { ...item.memory!, ram_expert_bytes: null } }]} runtime={{ memory_architecture: 'discrete' }} />);
  expect(container.querySelector('[data-tier="ram-experts"]')).toHaveTextContent('Breakdown not reported');
  rerender(<MemoryHierarchy instances={[item]} runtime={{ memory_architecture: 'unified', backend: 'cuda' }} />);
  expect(container.querySelectorAll('[data-tier]')).toHaveLength(5);
  expect(container.querySelector('[data-storage="vram"]')).not.toBeInTheDocument();
  expect(screen.queryByText('RAM-streamed experts')).not.toBeInTheDocument();
});
import type { RuntimeInstance } from '../../shared/api/types';

it('shows independent GPU capacities, model allocations and an idle third GPU', () => {
  const item = model('split', 8192);
  item.devices = ['cuda:0', 'cuda:2'];
  item.memory!.devices = [
    { device_id: '0', resident_weight_bytes: 1024, resident_buffer_bytes: 512, kv_bytes: 512, prefix_cache_bytes: 0 },
    { device_id: '2', resident_weight_bytes: 7168, resident_buffer_bytes: 1024, kv_bytes: 1024, prefix_cache_bytes: 256 },
  ];
  const { container } = render(<MemoryHierarchy instances={[item]} detailed runtime={{ memory_architecture: 'discrete',
    gpu_devices: [
      { id: '0', name: 'GPU A', memory_total_bytes: 8192, memory_available_bytes: 2048 },
      { id: '2', name: 'GPU B', memory_total_bytes: 16384, memory_available_bytes: 1024 },
      { id: '3', name: 'GPU C', memory_total_bytes: 8192, memory_available_bytes: 4096 },
    ] }} />);
  expect(screen.getByRole('heading', { name: 'VRAM · GPU 2' })).toBeInTheDocument();
  const first = container.querySelector('[data-gpu-id="0"]')!;
  const second = container.querySelector('[data-gpu-id="2"]')!;
  expect(first.querySelector('[data-tier="weights"] .memory-tier-heading')).toHaveTextContent('2 KiB / 4 KiB');
  expect(first.querySelector('[data-tier="weights"] .memory-tier-track [data-model-id="split"]')).toHaveStyle({ width: '37.5%' });
  expect(second.querySelector('[data-tier="weights"] .memory-tier-heading')).toHaveTextContent('8 KiB / 10 KiB');
  expect(container.querySelector('[data-gpu-id="3"] [data-model-id]')).toBeNull();
  expect(container.querySelectorAll('[data-storage="ram"]')).toHaveLength(1);
});

it('includes per-model buffers in VRAM usage and subtracts them from the KV capacity', () => {
  const first = model('1', 1024), second = model('2', 2048);
  first.memory!.resident_buffer_bytes = 2048;
  second.memory!.resident_buffer_bytes = 1024;
  const { container, rerender } = render(<MemoryHierarchy instances={[first, second]} detailed
    runtime={{ memory_architecture: 'discrete' }} memoryCapacityBytes={16384} />);
  expect(container.querySelector('[data-tier="weights"] .memory-tier-heading')).toHaveTextContent('6 KiB / 16 KiB');
  expect(container.querySelector('[data-tier="kv"] .memory-tier-heading')).toHaveTextContent('3 KiB / 10 KiB');
  expect(container.querySelector('[data-tier="weights"] .memory-tier-track [data-model-id="1"]')).toHaveStyle({ width: '18.75%' });
  expect(container.querySelector('[data-tier="weights"] .resource-tier-model small')).toHaveTextContent('Weights 1 KiB · Buffers 2 KiB');
  rerender(<MemoryHierarchy instances={[first, second]} runtime={{ memory_architecture: 'unified' }} memoryCapacityBytes={16384} />);
  expect(container.querySelector('[data-tier="weights"]')).toHaveTextContent('3 KiB / 16 KiB');
});

it('scales RAM weights and KV against the shared RAM capacity', () => {
  const item = model('1', 1024);
  Object.assign(item.memory!, { ram_experts: true, ram_expert_bytes: 4096, ram_kv: true, ram_kv_bytes: 2048 });
  const { container, rerender } = render(<MemoryHierarchy instances={[item]} memoryCapacityBytes={32768}
    runtime={{ memory_architecture: 'discrete', host_memory_total_bytes: 16384, host_memory_available_bytes: 8192 }} />);
  expect(container.querySelector('[data-tier="ram-experts"]')).toHaveTextContent('4 KiB / 14 KiB');
  expect(container.querySelector('[data-tier="ram-kv"]')).toHaveTextContent('2 KiB / 10 KiB');
  expect(parseFloat((container.querySelector('[data-tier="ram-experts"] .memory-tier-track > span') as HTMLElement).style.width)).toBeCloseTo(400 / 14);
  rerender(<MemoryHierarchy instances={[{ ...item, memory: { ...item.memory!, ram_expert_bytes: null } }]}
    runtime={{ memory_architecture: 'discrete', host_memory_total_bytes: 16384 }} />);
  expect(container.querySelector('[data-tier="ram-kv"]')).toHaveTextContent('2 KiB / --');
});

it('uses current free space for both pools and returns capacity when other applications release it', () => {
  const first = model('1', 4096), second = model('2', 2048);
  first.memory!.resident_buffer_bytes = 2048;
  second.memory!.resident_buffer_bytes = 1024;
  for (const item of [first, second]) Object.assign(item.memory!, {
    ram_experts: true, ram_expert_bytes: 4096, ram_kv: true, ram_kv_bytes: 1024,
  });
  const { container, rerender } = render(<MemoryHierarchy instances={[first, second]} memoryCapacityBytes={32768}
    runtime={{ memory_architecture: 'discrete', device_free_bytes: 2048,
      host_memory_total_bytes: 65536, host_memory_available_bytes: 4096 }} />);
  expect(container.querySelector('[data-tier="weights"] .memory-tier-heading')).toHaveTextContent('9 KiB / 17 KiB');
  expect(container.querySelector('[data-tier="kv"] .memory-tier-heading')).toHaveTextContent('6 KiB / 8 KiB');
  expect(container.querySelector('[data-tier="ram-experts"] .memory-tier-heading')).toHaveTextContent('8 KiB / 14 KiB');
  expect(container.querySelector('[data-tier="ram-kv"] .memory-tier-heading')).toHaveTextContent('2 KiB / 6 KiB');
  rerender(<MemoryHierarchy instances={[first, second]} memoryCapacityBytes={32768}
    runtime={{ memory_architecture: 'discrete', device_free_bytes: 65536,
      host_memory_total_bytes: 65536, host_memory_available_bytes: 65536 }} />);
  expect(container.querySelector('[data-tier="weights"] .memory-tier-heading')).toHaveTextContent('9 KiB / 32 KiB');
  expect(container.querySelector('[data-tier="ram-experts"] .memory-tier-heading')).toHaveTextContent('8 KiB / 64 KiB');
});

vi.mock('../settings/SettingsProvider', () => ({
  useSettings: () => ({ tr: (_zh: string, en: string) => en }),
}));

it('counts disposable RAM prefixes in available RAM without adding them to VRAM', () => {
  const item = model('1', 1024);
  item.memory = { ...item.memory!, ram_experts: true, ram_expert_bytes: 4096,
    ram_kv: true, ram_kv_bytes: 2048, prefix_cache_ram_bytes: 1024,
    prefix_cache_ram_limit_bytes: 4096, prefix_cache_ram_blocks: 4 };
  const { container } = render(<MemoryHierarchy instances={[item]} memoryCapacityBytes={32768}
    runtime={{ memory_architecture: 'discrete', device_free_bytes: 32768,
      host_memory_total_bytes: 65536, host_memory_available_bytes: 8192 }} />);
  expect(container.querySelector('[data-tier="ram-experts"] .memory-tier-heading')).toHaveTextContent('4 KiB / 15 KiB');
  expect(container.querySelector('[data-tier="ram-prefix"] .memory-tier-heading')).toHaveTextContent('1 KiB / 9 KiB');
  expect(container.querySelector('[data-tier="kv"] .memory-tier-heading')).toHaveTextContent('1 KiB / 31 KiB');
});

function model(id: string, bytes: number): RuntimeInstance {
  return { id, model: `Model ${id}`, state: 'ready', devices: ['metal'], active_sessions: 0,
    queued_requests: 0, started_at: `2026-01-01T00:00:0${id}Z`, memory: {
      resident_weight_bytes: bytes, kv_bytes: bytes, context_count: 1, prefix_cache_blocks: 2,
      ssd_experts: true, ssd_expert_bytes: bytes, ssd_ple: true, ssd_ple_bytes: bytes,
      ssd_kv: true, ssd_kv_bytes: bytes,
    } };
}

it.each([false, true])('groups the same resource tiers into RAM and SSD on unified memory: detailed=%s', async (detailed) => {
  const { container } = render(<MemoryHierarchy instances={[model('1', 1024)]} runtime={{ memory_architecture: 'unified' }} detailed={detailed} />);
  expect([...container.querySelectorAll('[data-memory-group]')].map(group => group.getAttribute('data-memory-group'))).toEqual(['RAM', 'SSD']);
  expect(container.querySelector('[data-memory-group="RAM"]')?.querySelectorAll('[data-tier]')).toHaveLength(2);
  expect(container.querySelector('[data-memory-group="SSD"]')?.querySelectorAll('[data-tier]')).toHaveLength(3);
  expect(container.querySelector('.overview-memory-heading > div > p')).toHaveTextContent('capacity available to MFQ after system and other process usage');
  expect(container.querySelector('.overview-memory-heading > div > p')).not.toHaveTextContent('RAM holds');
  for (const heading of container.querySelectorAll('.memory-tier-group-heading')) {
    expect(heading.children[0].tagName).toBe('H3');
    expect(heading.children[1].tagName).toBe('I');
    expect(heading.children[1]).toHaveAttribute('aria-hidden', 'true');
  }
  const helpButton = screen.getByRole('button', { name: 'Memory tier details' });
  expect(helpButton.closest('.memory-heading-title')).toContainElement(screen.getByRole('heading', { level: 2 }));
  expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
  fireEvent.focus(helpButton);
  const help = await screen.findByRole('tooltip');
  expect(help).toHaveTextContent('unified-memory devices share it between CPU and GPU');
  expect(help).toHaveTextContent('excludes prefix caches');
});

it('keeps separate VRAM, RAM and SSD categories and their device-specific explanations', async () => {
  const { container } = render(<MemoryHierarchy instances={[model('1', 1024)]} runtime={{ memory_architecture: 'discrete' }} />);
  expect([...container.querySelectorAll('[data-memory-group]')].map(group => group.getAttribute('data-memory-group'))).toEqual(['VRAM', 'RAM', 'SSD']);
  expect(container.querySelector('[data-memory-group="VRAM"]')?.querySelectorAll('[data-tier]')).toHaveLength(2);
  expect(container.querySelector('[data-memory-group="RAM"]')?.querySelectorAll('[data-tier]')).toHaveLength(3);
  fireEvent.focus(screen.getByRole('button', { name: 'Memory tier details' }));
  expect(await screen.findByRole('tooltip')).toHaveTextContent('transferred to VRAM on demand');
  expect(container.querySelector('.overview-memory-heading > div > p')).toHaveTextContent('Memory and VRAM limits below');
  expect(container.querySelector('.overview-memory-heading > div > p')).toHaveTextContent('including MFQ allocations and remaining free space');
});

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
  expect(screen.getByLabelText('Model color legend').querySelector('i')).toHaveStyle({ backgroundColor: 'var(--bar-fill)' });
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

it('uses one neutral fill with stable load order and preserves per-model resource shares', () => {
  const models = [model('1', 1024), model('2', 3072), model('3', 4096), model('4', 8192)];
  const { container } = render(<MemoryHierarchy instances={[...models].reverse()} memoryCapacityBytes={32768} />);
  const legend = container.querySelector('.memory-model-legend')!;
  const colors = models.map((item) => (legend.querySelector(`[data-model-id="${item.id}"] i`) as HTMLElement).style.backgroundColor);
  expect(new Set(colors).size).toBe(1);
  expect(colors[0]).toBe('var(--bar-fill)');
  expect([...legend.children].map(item => (item as HTMLElement).dataset.modelId)).toEqual(['1', '2', '3', '4']);
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

it('keeps the same neutral fill and stable order after unload/reordering/reload', () => {
  const first = model('1', 1024), second = model('2', 1024);
  const { container, rerender } = render(<MemoryHierarchy instances={[first, second]} memoryCapacityBytes={4096} />);
  rerender(<MemoryHierarchy instances={[second]} memoryCapacityBytes={4096} />);
  expect(container.querySelector('[data-tier="kv"] [data-model-id="2"]')).toHaveStyle({ backgroundColor: 'var(--bar-fill)' });
  expect(screen.getByText('1 KiB / 3 KiB')).toBeInTheDocument();
  rerender(<MemoryHierarchy instances={[second, { ...first, id: 'reload' }]} memoryCapacityBytes={4096} />);
  expect(container.querySelector('[data-tier="ple"] [data-model-id="reload"]')).toHaveStyle({ backgroundColor: 'var(--bar-fill)' });
  expect([...container.querySelector('.memory-model-legend')!.children].map(item => (item as HTMLElement).dataset.modelId)).toEqual(['reload', '2']);
  expect(screen.getByText('2 KiB / 2 KiB')).toBeInTheDocument();
});

it('excludes failed/loading models and distinguishes unavailable telemetry from zero', () => {
  const zero = model('1', 0);
  const { container, rerender } = render(<MemoryHierarchy instances={[
    zero, { ...model('2', 500), state: 'failed' }, { ...model('3', 500), state: 'loading' },
  ]} memoryCapacityBytes={4096} />);
  expect(screen.queryByText('Model 2')).not.toBeInTheDocument();
  expect(screen.queryByText('Model 3')).not.toBeInTheDocument();
  expect([...container.querySelectorAll('.memory-tier-heading > span')].filter(item => item.textContent === '0 B')).toHaveLength(3);
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
    expect(tier.querySelector('.resource-tier-model-name i')).toHaveStyle({ backgroundColor: 'var(--bar-fill)' });
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
  expect(tier.querySelector('[data-model-id="1"]')).toHaveStyle({ width: '25%', backgroundColor: 'var(--bar-fill)' });
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
