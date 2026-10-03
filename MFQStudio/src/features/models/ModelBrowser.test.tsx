import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import type { HubModelInfo, HubSystemProfile, OfficialModelList } from '../../shared/api/types';
import { modelsApi } from '../../shared/api/resources/models';
import { ModelBrowser } from './ModelBrowser';

vi.mock('../../shared/api/resources/models', () => ({
  modelsApi: { officialHubModels: vi.fn(), hubModelInfo: vi.fn() },
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
  [{ backend: 'metal', cpu_name: 'Apple M5 Max', cpu_cores: 18, gpu_names: ['Apple M5 Max'], gpu_cores: 40, physical_memory_bytes: 128 * 2 ** 30 }, 'Apple M5 Max · 18 CPU / 40 GPU · 128 GiB RAM', 'Apple · METAL'],
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
