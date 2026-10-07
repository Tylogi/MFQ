import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import type { HubModelInfo, HubSystemProfile, OfficialModelList } from '../../shared/api/types';
import { modelsApi } from '../../shared/api/resources/models';
import { ModelBrowser } from './ModelBrowser';

vi.mock('../../shared/api/resources/models', () => ({
  modelsApi: { officialHubModels: vi.fn(), hubModelInfo: vi.fn(), resolveHubModel: vi.fn() },
}));

const configuration = {
  status: 'unknown' as const, recommendation: 'unknown' as const, reasons: [],
};

function catalog(refreshing: boolean, available: boolean): OfficialModelList {
  return {
    refreshing,
    system: { platform: 'test', machine: 'test', backend: 'unknown' },
    data: ['first', 'second'].map((id) => {
      const source = {
        provider: 'modelscope' as const, repo_id: `owner/${id}`,
        revision: 'master', url: `https://modelscope.cn/models/owner/${id}`, available,
      };
      return {
        id, name: id, family: 'Test', architecture: 'test', description: '', description_zh: '',
        modalities: [], capabilities: [], precision_options: [], supports_ssd_streaming: false,
        sources: [source], selected_source: source, revision: 'master', downloads: 0, likes: 0,
        configuration, variants: available ? [{
          id: 'S4', label: `${id}-S4`, format: 'mfq' as const, files: ['S4.mfq'],
          byte_size: 42, configuration,
        }] : [],
      };
    }),
  };
}

afterEach(() => {
  vi.useRealTimers();
  vi.clearAllMocks();
});

it('shows the resident estimate rather than the payload baseline and keeps PLE separate', async () => {
  const data = catalog(false, true);
  const variant = data.data[0].variants[0];
  variant.byte_size = 110 * 2 ** 30;
  variant.resident_weight_bytes = 80 * 2 ** 30;
  variant.estimated_resident_weight_bytes = 88 * 2 ** 30;
  variant.ssd_ple_bytes = 30 * 2 ** 30;
  variant.configuration = {
    status: 'warning', recommendation: 'caution',
    required_memory_bytes: 88 * 2 ** 30, available_memory_bytes: 80 * 2 ** 30, reasons: [],
  };
  data.data[0].configuration = variant.configuration;
  vi.mocked(modelsApi.officialHubModels).mockResolvedValue(data);
  render(<ModelBrowser tab="official" onTabChange={vi.fn()} jobKinds={[]} onError={vi.fn()} onJobCreated={vi.fn()} tr={(_, en) => en} />);
  expect(await screen.findByText(/file 110.0 GiB · est. resident weights 88.0 GiB · SSD PLE 30.0 GiB/)).toBeInTheDocument();
  expect(screen.getByText('Smallest tier estimated resident weights')).toBeInTheDocument();
  expect(screen.queryByText(/resident weight baseline 80.0 GiB/)).not.toBeInTheDocument();
});

it('keeps older servers without a resident estimate readable', async () => {
  const data = catalog(false, true);
  data.data[0].variants[0].resident_weight_bytes = 80 * 2 ** 30;
  data.data[0].variants[0].configuration.required_memory_bytes = 80 * 2 ** 30;
  vi.mocked(modelsApi.officialHubModels).mockResolvedValue(data);
  render(<ModelBrowser tab="official" onTabChange={vi.fn()} jobKinds={[]} onError={vi.fn()} onJobCreated={vi.fn()} tr={(_, en) => en} />);
  expect(await screen.findByText(/resident weight baseline 80.0 GiB/)).toBeInTheDocument();
});

it.each([true, false])('shows logical parameter components and the requested nine metadata cells: PLE=%s', async (ple) => {
  const data = catalog(false, true);
  data.data[0].parameter_breakdown = {
    total: (ple ? 176.2 : 125) * 1e9, dense: 4.2e9, routed_experts: 120.8e9,
    ple: ple ? 51.2e9 : 0, active: 6e9,
  };
  data.data[0].mtp_supported = true;
  data.data[0].capabilities = ['MTP', 'MoE', 'SSD streaming'];
  data.data[0].supports_ssd_streaming = true;
  vi.mocked(modelsApi.officialHubModels).mockResolvedValue(data);
  render(<ModelBrowser tab="official" onTabChange={vi.fn()} jobKinds={[]} onError={vi.fn()} onJobCreated={vi.fn()} tr={(zh) => zh} />);
  expect(await screen.findByText(ple ? '176.2B（4.2B 稠密 + 120.8B 路由专家 + 51.2B PLE）' : '125B（4.2B 稠密 + 120.8B 路由专家）')).toBeInTheDocument();
  expect(document.querySelector('.model-mtp-support dd')).toHaveTextContent('是');
  expect([...document.querySelectorAll('.model-metadata-grid > div > dt')].map((item) => item.textContent)).toEqual(['架构', '参数', '激活参数', '模态', 'MTP支持', '许可', '下载', '收藏', '发布时间']);
  expect(document.querySelector('.model-metadata-grid')!.children).toHaveLength(9);
  expect(screen.queryByText('内存压力')).not.toBeInTheDocument();
  expect(document.querySelector('.configuration-badge')).toBeNull();
  expect(document.querySelector('.memory-pressure-guide')).toBeNull();
  expect(screen.queryByText('功能')).not.toBeInTheDocument();
  expect(screen.queryByText(/SSD streaming/)).not.toBeInTheDocument();
  expect(document.querySelector('.streaming-note')).toBeNull();
});

it('keeps exactly one MTP property and calculator after repeated model switches', async () => {
  const data = catalog(false, true);
  data.data[0].mtp_supported = true;
  data.data[1].mtp_supported = false;
  vi.mocked(modelsApi.officialHubModels).mockResolvedValue(data);
  const view = render(<ModelBrowser tab="official" onTabChange={vi.fn()} jobKinds={[]} onError={vi.fn()} onJobCreated={vi.fn()} tr={(_, en) => en} />);
  await screen.findByText('first-S4');
  for (let index = 0; index < 8; index += 1) {
    fireEvent.click(screen.getByText('KV Cache curve & calculator'));
    fireEvent.click(screen.getByRole('button', { name: index % 2 === 0 ? /secondTest/ : /firstTest/ }));
    expect(view.container.querySelectorAll('.model-mtp-support')).toHaveLength(1);
    expect(view.container.querySelectorAll('.kv-cache-planner')).toHaveLength(1);
    expect(screen.getAllByText('MTP support')).toHaveLength(1);
    expect(screen.getByText('KV Cache curve & calculator').closest('details')!.open).toBe(false);
    expect(view.container.querySelector('.model-mtp-support dd')).toHaveTextContent(index % 2 === 0 ? 'No' : 'Yes');
    expect(view.container.querySelector('.model-metadata-grid')!.children).toHaveLength(9);
  }
});

it('keeps one MTP property with the selected source metadata after repeated source switches', async () => {
  const data = catalog(false, true);
  const first = data.data[0];
  first.mtp_supported = true;
  first.sources.push({ provider: 'huggingface', repo_id: 'owner/first', revision: 'master',
    url: 'https://huggingface.co/owner/first', available: true });
  vi.mocked(modelsApi.officialHubModels).mockResolvedValue(data);
  vi.mocked(modelsApi.hubModelInfo).mockImplementation(async (provider, repo_id, revision) => ({
    provider, repo_id, revision: revision || 'master', mtp_supported: provider === 'modelscope',
    downloads: 0, likes: 0, total_bytes: 42, files: [], tags: [], architectures: [],
    modalities: [], gated: false, variants: first.variants,
  }));
  const view = render(<ModelBrowser tab="official" onTabChange={vi.fn()} jobKinds={[]} onError={vi.fn()} onJobCreated={vi.fn()} tr={(_, en) => en} />);
  await screen.findByText('first-S4');
  for (let index = 0; index < 8; index += 1) {
    await act(async () => {
      fireEvent.change(screen.getByRole('combobox', { name: 'Download source' }), { target: { value: index % 2 === 0 ? '1' : '0' } });
    });
    expect(view.container.querySelectorAll('.model-mtp-support')).toHaveLength(1);
    expect(view.container.querySelectorAll('.kv-cache-planner')).toHaveLength(1);
    expect(screen.getAllByText('MTP support')).toHaveLength(1);
    expect(view.container.querySelector('.model-mtp-support dd')).toHaveTextContent(index % 2 === 0 ? 'No' : 'Yes');
  }
});

it.each(['zh', 'en'])('translates model modalities for the current language: %s', async (language) => {
  const data = catalog(false, true);
  data.data[0].modalities = ['text', 'image', 'video', 'audio'];
  vi.mocked(modelsApi.officialHubModels).mockResolvedValue(data);
  render(<ModelBrowser tab="official" onTabChange={vi.fn()} jobKinds={[]} onError={vi.fn()} onJobCreated={vi.fn()} tr={(zh, en) => language === 'zh' ? zh : en} />);
  expect(await screen.findByText(language === 'zh' ? '文本 · 图像 · 视频 · 音频' : 'text · image · video · audio')).toBeInTheDocument();
});

it('uses the same nine-cell layout for third-party models', async () => {
  vi.mocked(modelsApi.officialHubModels).mockResolvedValue(catalog(false, true));
  vi.mocked(modelsApi.resolveHubModel).mockResolvedValue({ provider: 'huggingface', repo_id: 'owner/model', revision: 'master',
    downloads: 1234, likes: 56, total_bytes: 42, files: [], tags: [], architectures: ['test'],
    modalities: ['text'], mtp_supported: false, gated: false, variants: [], published_at: '2026-10-01T00:00:00Z' });
  const view = render(<ModelBrowser tab="community" onTabChange={vi.fn()} jobKinds={[]} onError={vi.fn()} onJobCreated={vi.fn()} tr={(zh) => zh} />);
  fireEvent.change(screen.getByPlaceholderText('模型名称、owner/repo 或仓库链接'), { target: { value: 'owner/model' } });
  await act(async () => fireEvent.click(screen.getByRole('button', { name: '查找' })));
  expect([...view.container.querySelectorAll('.model-metadata-grid > div > dt')].map((item) => item.textContent)).toEqual(['架构', '参数', '激活参数', '模态', 'MTP支持', '许可', '下载', '收藏', '发布时间']);
  expect(view.container.querySelector('.model-mtp-support dd')).toHaveTextContent('否');
});

it('applies KV planning to every precision tier and recommendations without changing weight estimates', async () => {
  const data = catalog(false, true);
  const first = data.data[0];
  first.cache_profile = { max_context: 32768, fixed_bytes: 0,
    components: [{ bytes_per_row: 2 ** 20, tokens_per_row: 1, minimum_rows: 0, allocation: 'exact' }] };
  first.variants = [8, 9].map((gib) => ({ ...first.variants[0], id: String(gib), label: `S${gib}`,
    estimated_resident_weight_bytes: gib * 2 ** 30,
    configuration: { status: 'recommended', recommendation: 'three_stars', reasons: [],
      required_memory_bytes: gib * 2 ** 30, available_memory_bytes: 10 * 2 ** 30 } }));
  first.configuration = first.variants[0].configuration;
  vi.mocked(modelsApi.officialHubModels).mockResolvedValue(data);
  render(<ModelBrowser tab="official" onTabChange={vi.fn()} jobKinds={[]} onError={vi.fn()} onJobCreated={vi.fn()} tr={(_, en) => en} />);
  await screen.findByText('S8');
  expect(screen.getByRole('progressbar', { name: 'Estimated share of runtime budget: 80.0%' })).toBeInTheDocument();
  fireEvent.click(screen.getByText('KV Cache curve & calculator'));
  fireEvent.change(screen.getByRole('spinbutton', { name: 'Expected ctx' }), { target: { value: '4096' } });
  expect(screen.getByRole('progressbar', { name: 'Estimated share of runtime budget: 80.0%' })).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
  expect(screen.getByRole('progressbar', { name: 'Estimated share of runtime budget: 120.0%' })).toHaveAttribute('aria-valuenow', '100');
  expect(screen.getByRole('progressbar', { name: 'Estimated share of runtime budget: 130.0%' })).toBeInTheDocument();
  expect(screen.getByText(/est. resident weights 8.0 GiB/)).toBeInTheDocument();
  expect(screen.getByText('Planned KV/recurrent state 4.0 GiB · Est. total residency 12.0 GiB')).toBeInTheDocument();
  expect(screen.getByText('Resident weights + planned KV/recurrent state: 0/2 tiers fit the budget.')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: /secondTest/ }));
  expect(screen.queryByText(/Planned KV\/recurrent state/)).not.toBeInTheDocument();
  expect(screen.getByText('KV Cache curve & calculator').closest('details')!.open).toBe(false);
  fireEvent.click(screen.getByRole('button', { name: /firstTest/ }));
  expect(screen.getByRole('progressbar', { name: 'Estimated share of runtime budget: 120.0%' })).toBeInTheDocument();
  fireEvent.click(screen.getByText('KV Cache curve & calculator'));
  fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
  expect(screen.getByRole('progressbar', { name: 'Estimated share of runtime budget: 80.0%' })).toBeInTheDocument();
  expect(modelsApi.hubModelInfo).not.toHaveBeenCalled();
});

it('renders immediately and preserves the chosen card while metadata arrives', async () => {
  vi.useFakeTimers();
  vi.mocked(modelsApi.officialHubModels)
    .mockResolvedValueOnce(catalog(true, false))
    .mockResolvedValueOnce(catalog(false, true));
  const view = render(<ModelBrowser tab="official" onTabChange={vi.fn()} jobKinds={[]} onError={vi.fn()} onJobCreated={vi.fn()} tr={(_, en) => en} />);
  await act(async () => {});
  expect(screen.getByRole('button', { name: 'Refreshing' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: /secondTest/ }));
  await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
  expect(screen.getByRole('heading', { name: 'second' })).toBeInTheDocument();
  expect(screen.getByText('second-S4')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Refresh' })).toBeEnabled();
  await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
  expect(modelsApi.officialHubModels).toHaveBeenCalledTimes(2);
  view.unmount();
});

it('cancels catalog requests and polling on unmount', async () => {
  vi.useFakeTimers();
  vi.mocked(modelsApi.officialHubModels).mockResolvedValue(catalog(true, false));
  const onError = vi.fn();
  const view = render(<ModelBrowser tab="official" onTabChange={vi.fn()} jobKinds={[]} onError={onError} onJobCreated={vi.fn()} tr={(_, en) => en} />);
  await act(async () => {});
  const signal = vi.mocked(modelsApi.officialHubModels).mock.calls[0][1]!;
  view.unmount();
  expect(signal.aborted).toBe(true);
  await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
  expect(modelsApi.officialHubModels).toHaveBeenCalledTimes(1);
  expect(onError).not.toHaveBeenCalled();
});

it('does not discard a source detail request when the catalog is updated', async () => {
  vi.useFakeTimers();
  const initial = catalog(true, true);
  const alternate = {
    provider: 'huggingface' as const, repo_id: 'owner/first', revision: 'revision',
    url: 'https://huggingface.co/owner/first', available: true,
  };
  initial.data[0].sources.push(alternate);
  const completed = { ...initial, refreshing: false };
  vi.mocked(modelsApi.officialHubModels).mockResolvedValueOnce(initial).mockResolvedValueOnce(completed);
  let finish!: (info: HubModelInfo) => void;
  vi.mocked(modelsApi.hubModelInfo).mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
  const view = render(<ModelBrowser tab="official" onTabChange={vi.fn()} jobKinds={[]} onError={vi.fn()} onJobCreated={vi.fn()} tr={(_, en) => en} />);
  await act(async () => {});
  fireEvent.change(screen.getByRole('combobox', { name: 'Download source' }), { target: { value: '1' } });
  await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
  await act(async () => {
    finish({
      provider: 'huggingface', repo_id: alternate.repo_id, revision: alternate.revision,
      downloads: 0, likes: 0, total_bytes: 42, files: [], tags: [], architectures: [],
      modalities: [], gated: false,
      variants: [{ ...initial.data[0].variants[0], label: 'Alternate S4' }],
    });
  });
  expect(screen.getByRole('combobox', { name: 'Download source' })).toHaveValue('1');
  expect(screen.getByText('Alternate S4')).toBeInTheDocument();
  view.unmount();
});

it.each([
  [{ backend: 'metal', cpu_name: 'Apple M5 Max', cpu_cores: 18, gpu_names: ['Apple M5 Max'], gpu_cores: 40, physical_memory_bytes: 128 * 2 ** 30 }, 'Apple M5 Max · 18 CPU / 40 GPU · 128 GiB URAM', 'Apple · METAL'],
  [{ backend: 'cuda', cpu_name: 'AMD Ryzen 5 9600X', gpu_names: ['NVIDIA GeForce RTX 5090'], physical_memory_bytes: 64 * 2 ** 30 }, 'NVIDIA GeForce RTX 5090 · 64 GiB RAM · AMD Ryzen 5 9600X', 'NVIDIA · CUDA'],
  [{ backend: 'rocm', cpu_name: 'AMD Ryzen 9', gpu_names: ['AMD Radeon'], physical_memory_bytes: 64 * 2 ** 30 }, 'AMD Radeon · 64 GiB RAM · AMD Ryzen 9', 'AMD · ROCM'],
] as [Partial<HubSystemProfile>, string, string][])('shows concrete hardware and a monochrome backend vendor badge: %s', async (hardware, summary, badge) => {
  const data = catalog(false, false);
  data.system = { ...data.system, ...hardware };
  vi.mocked(modelsApi.officialHubModels).mockResolvedValue(data);
  render(<ModelBrowser tab="official" onTabChange={vi.fn()} jobKinds={[]} onError={vi.fn()} onJobCreated={vi.fn()} tr={(_, en) => en} />);
  expect(await screen.findByText(summary)).toBeInTheDocument();
  expect(screen.getByRole('img', { name: badge }).querySelector('svg')).toHaveAttribute('fill', 'none');
  expect(screen.queryByText(/test · test/)).not.toBeInTheDocument();
});

it.each([
  [[{ kind: 'uma', capacity_bytes: 128 * 2 ** 30, bandwidth_bytes_per_second: 614e9 }], ['128 GiB URAM'], ['571.8 GiB/s']],
  [[{ kind: 'vram', capacity_bytes: 32 * 2 ** 30, bandwidth_bytes_per_second: 1726 * 2 ** 30 }, { kind: 'ram', capacity_bytes: 64 * 2 ** 30, bandwidth_bytes_per_second: 104 * 2 ** 30 }], ['32 GiB VRAM', '64 GiB RAM'], ['1,726 GiB/s', '104 GiB/s']],
  [[{ kind: 'ram', capacity_bytes: 64 * 2 ** 30 }], ['64 GiB RAM'], ['Bandwidth unavailable']],
  [[{ kind: 'vram' }, { kind: 'ram', capacity_bytes: 64 * 2 ** 30 }], ['— VRAM', '64 GiB RAM'], ['Bandwidth unavailable', 'Bandwidth unavailable']],
] as [NonNullable<HubSystemProfile['memory_pools']>, string[], string[]][])('shows each memory capacity with its own bandwidth underneath: %s', async (pools, capacities, bandwidths) => {
  const data = catalog(false, false);
  data.system = { ...data.system, memory_pools: pools, runtime_memory_budget_bytes: 96 * 2 ** 30 };
  vi.mocked(modelsApi.officialHubModels).mockResolvedValue(data);
  render(<ModelBrowser tab="official" onTabChange={vi.fn()} jobKinds={[]} onError={vi.fn()} onJobCreated={vi.fn()} tr={(_, en) => en} />);
  await screen.findByText(capacities[0]);
  const rows = document.querySelectorAll('.detected-memory-pool');
  expect(rows).toHaveLength(capacities.length);
  rows.forEach((row, index) => {
    expect(row.querySelector('strong')).toHaveTextContent(capacities[index]);
    expect(row.querySelector('small')).toHaveTextContent(bandwidths[index]);
  });
  expect(screen.queryByText('96.0 GiB')).not.toBeInTheDocument();
  expect(screen.queryByText('614 GiB/s')).not.toBeInTheDocument();
});
