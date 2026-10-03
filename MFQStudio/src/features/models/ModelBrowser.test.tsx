import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import type { HubModelInfo, OfficialModelList } from '../../shared/api/types';
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
  const view = render(<ModelBrowser jobKinds={[]} onError={vi.fn()} onJobCreated={vi.fn()} tr={(_, en) => en} />);
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
  const view = render(<ModelBrowser jobKinds={[]} onError={onError} onJobCreated={vi.fn()} tr={(_, en) => en} />);
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
  const view = render(<ModelBrowser jobKinds={[]} onError={vi.fn()} onJobCreated={vi.fn()} tr={(_, en) => en} />);
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
