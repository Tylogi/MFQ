import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { useModelCatalog } from './useModelCatalog';
import { ModelsPage } from './ModelsPage';
import { LoadedModels } from './LoadedModels';
import { LocalCheckpoints } from './LocalCheckpoints';
import { ModelLoadPolicy } from './ModelLoadPolicy';
import { ModelDirectoryDialog } from './ModelDirectoryDialog';

vi.mock('../settings/SettingsProvider', () => ({
  useSettings: () => ({ tr: (zh: string) => zh }),
}));
vi.mock('./useModelCatalog', () => ({ useModelCatalog: vi.fn() }));

function catalog() {
  return {
    runtime: null,
    busy: false,
    artifacts: [],
    instances: [],
    availableModelNames: [],
    modelFilter: '',
    modelFolderPath: '/',
    modelBrowserOpen: false,
    modelDirectoryPath: '/',
    setModelBrowserOpen: vi.fn(),
    filteredInstances: [],
    filteredArtifacts: [],
    loadPinned: false,
    loadIdleTtl: null,
    chooseModelDirectory: vi.fn(),
    openModelFiles: vi.fn(),
    openCurrentDirectoryInFinder: vi.fn(),
    unloadInstance: vi.fn(),
    loadArtifact: vi.fn(),
    setLoadPinned: vi.fn(),
    setLoadIdleTtl: vi.fn(),
  } as unknown as ReturnType<typeof useModelCatalog>;
}

it('空列表操作仍打开模型目录', () => {
  const state = catalog();
  render(<><LoadedModels catalog={state} /><LocalCheckpoints catalog={state} /></>);
  fireEvent.click(screen.getByRole('button', { name: '添加模型' }));
  fireEvent.click(screen.getByRole('button', { name: '选择模型文件夹' }));
  expect(state.chooseModelDirectory).toHaveBeenCalledTimes(2);
});

it('模型文件夹栏显示实际路径，单独的更改按钮打开目录选择器', () => {
  const state = catalog();
  state.modelFolderPath = '/actual/model/folder';
  vi.mocked(useModelCatalog).mockReturnValue(state);
  const { container } = render(<ModelsPage />);
  const row = container.querySelector('.model-folder-location');
  expect(row).toHaveTextContent('模型文件夹');
  expect(row?.querySelector('code')).toHaveTextContent('/actual/model/folder');
  expect(row?.querySelector('code')).toHaveAttribute('title', '/actual/model/folder');
  fireEvent.click(screen.getByRole('button', { name: '更改' }));
  expect(state.chooseModelDirectory).toHaveBeenCalledOnce();
});

it('固定与空闲卸载仍调用对应策略操作', () => {
  const state = catalog();
  render(<ModelLoadPolicy catalog={state} />);
  fireEvent.click(screen.getByRole('switch', { name: '固定到内存' }));
  fireEvent.change(screen.getByRole('combobox'), { target: { value: '900' } });
  expect(state.setLoadPinned).toHaveBeenCalledWith(true);
  expect(state.setLoadIdleTtl).toHaveBeenCalledWith(900);
});

it('本地架构标识在模型行右侧，改名的模型仍按架构识别', () => {
  const state = catalog();
  state.filteredArtifacts = [
    { id: 'qwen', name: 'renamed-checkpoint', architecture: 'qwen4_exp', loadable: true },
    { id: 'ds', name: 'DeepSeek-V4.1', architecture: 'deepseek_v4', loadable: true },
    { id: 'glm', name: 'GLM-5.3', architecture: 'glm5_next', loadable: true },
    { id: 'other', name: 'MiniCPM-o', architecture: 'minicpm', loadable: true },
  ] as typeof state.filteredArtifacts;
  state.artifacts = state.filteredArtifacts;
  render(<LocalCheckpoints catalog={state} />);
  for (const name of ['Qwen', 'DeepSeek', 'Z.ai']) {
    expect(screen.getByRole('img', { name }).closest('.model-row-actions')).not.toBeNull();
  }
  expect(screen.getAllByRole('img')).toHaveLength(3);
  fireEvent.click(screen.getAllByRole('button', { name: '加载' })[0]);
  expect(state.loadArtifact).toHaveBeenCalledWith('renamed-checkpoint');
  fireEvent.click(screen.getAllByRole('button', { name: '模型文件' })[0]);
  expect(state.openModelFiles).toHaveBeenCalledWith('qwen');
  expect(screen.queryByText(/qwen4_exp|minicpm|deepseek_v4|glm5_next/)).not.toBeInTheDocument();
});

it('模型文件弹窗显示实际路径且不提供重复注册操作', () => {
  const state = catalog();
  Object.assign(state, { modelFilesMode: true, modelBrowserOpen: true, modelDirectoryPath: '/actual/model/S4-L',
    modelBrowser: { current_id: 'directory', current_path: '/actual/model/S4-L', model_file_count: 1, data: [] },
    setModelBrowserOpen: vi.fn(), openModelDirectory: vi.fn(), jumpToModelDirectory: vi.fn(), setModelDirectoryPath: vi.fn() });
  render(<ModelDirectoryDialog catalog={state} />);
  expect(screen.getByRole('dialog', { name: '模型文件' })).toBeInTheDocument();
  expect(screen.getByRole('textbox', { name: '当前目录' })).toHaveValue('/actual/model/S4-L');
  expect(screen.queryByRole('button', { name: '使用此文件夹' })).not.toBeInTheDocument();
});

it('目录浏览器同时显示子目录和 MFQ 文件大小，访达按钮只打开当前目录', () => {
  const state = catalog();
  Object.assign(state, { modelFilesMode: true, modelBrowserOpen: true, modelDirectoryPath: '/model/files',
    modelBrowser: { current_id: 'directory', current_path: '/model/files', can_open_in_finder: true,
      model_file_count: 1, data: [{ id: 'child', name: 'nested', model_file_count: 0 }],
      files: [{ name: 'model-00001-of-00002.mfq', byte_size: 2 ** 30 }, { name: 'model-00002-of-00002.mfq', byte_size: 2 * 2 ** 30 }] },
    openModelDirectory: vi.fn(), jumpToModelDirectory: vi.fn(), setModelDirectoryPath: vi.fn() });
  render(<ModelDirectoryDialog catalog={state} />);
  expect(screen.getByRole('button', { name: 'nested' })).toBeInTheDocument();
  expect(screen.getByText('model-00001-of-00002.mfq')).toBeInTheDocument();
  expect(screen.getByText('1 GiB')).toBeInTheDocument();
  expect(screen.getByText('2 GiB')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: '在访达中打开' }));
  expect(state.openCurrentDirectoryInFinder).toHaveBeenCalledOnce();
  expect(screen.queryByRole('button', { name: '使用此文件夹' })).not.toBeInTheDocument();
});

it('没有子目录但存在 MFQ 文件时不误报空目录', () => {
  const state = catalog();
  Object.assign(state, { modelBrowserOpen: true, modelBrowser: { current_id: 'folder', current_path: '/model',
    data: [], files: [{ name: 'model.mfq', byte_size: 0 }] } });
  render(<ModelDirectoryDialog catalog={state} />);
  expect(screen.getByText('model.mfq')).toBeInTheDocument();
  expect(screen.getByText('0 B')).toBeInTheDocument();
  expect(screen.queryByText(/没有子文件夹/)).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: '在访达中打开' })).toBeDisabled();
});

it('本地检查点保留大小和分片数，常驻压力使用实时剩余内存而非总预算', () => {
  const state = catalog();
  state.filteredArtifacts = [{ id: 'model', name: 'Checkpoint', architecture: 'internal-recipe',
    total_bytes: 40 * 2 ** 30, shard_count: 6, complete: true, loadable: true,
    estimated_resident_weight_bytes: 12 * 2 ** 30, ssd_ple_bytes: 28 * 2 ** 30 }] as typeof state.filteredArtifacts;
  state.runtime = { runtime_memory_headroom_bytes: 16 * 2 ** 30,
    runtime_memory_effective_budget_bytes: 100 * 2 ** 30 } as typeof state.runtime;
  const { rerender } = render(<LocalCheckpoints catalog={state} />);
  expect(screen.getByText('40 GiB · 6 个分片')).toBeInTheDocument();
  expect(screen.getByText('预计常驻内存 12 GiB · SSD PLE 28 GiB')).toBeInTheDocument();
  expect(screen.getByRole('progressbar', { name: '预计占剩余可用内存: 75.0%' })).toHaveAttribute('aria-valuenow', '75');
  expect(screen.queryByText(/internal-recipe/)).not.toBeInTheDocument();
  state.runtime = { ...state.runtime, runtime_memory_headroom_bytes: 8 * 2 ** 30 } as typeof state.runtime;
  rerender(<LocalCheckpoints catalog={state} />);
  expect(screen.getByText('剩余可用内存 8 GiB')).toBeInTheDocument();
  expect(screen.getByRole('progressbar', { name: '预计占剩余可用内存: 150.0%' })).toHaveAttribute('aria-valuenow', '100');
});

it('分片不全保留已下载文件大小及缺片数，不按文件大小伪造常驻估算', () => {
  const state = catalog();
  state.filteredArtifacts = [{ id: 'model', name: 'Incomplete', total_bytes: 6 * 2 ** 30,
    shard_count: 6, missing_shards: 2, complete: false, loadable: false }] as typeof state.filteredArtifacts;
  render(<LocalCheckpoints catalog={state} />);
  expect(screen.getByText('6 GiB · 6 个分片，缺 2 片')).toBeInTheDocument();
  expect(screen.getByText('预计常驻内存 —')).toBeInTheDocument();
  expect(screen.getByRole('progressbar')).not.toHaveAttribute('aria-valuenow');
});

it('剩余内存为零时显示耗尽而非未知或无穷百分比', () => {
  const state = catalog();
  state.filteredArtifacts = [{ id: 'model', name: 'Checkpoint', total_bytes: 2 ** 30,
    shard_count: 1, estimated_resident_weight_bytes: 2 ** 30, loadable: true }] as typeof state.filteredArtifacts;
  state.runtime = { runtime_memory_headroom_bytes: 0 } as typeof state.runtime;
  render(<LocalCheckpoints catalog={state} />);
  expect(screen.getByText('剩余可用内存 0 GiB')).toBeInTheDocument();
  expect(screen.getByRole('progressbar', { name: '预计占剩余可用内存: 无可用内存' })).toHaveAttribute('aria-valuenow', '100');
});

it.each([[[], '0 B'], [[32 * 2 ** 30, 8 * 2 ** 30], '40 GiB']] as [number[], string][])('资产总大小汇总已登记文件，不使用当前会话模型: %s', (sizes, total) => {
  const state = catalog();
  state.runtime = { model: 'Loaded but not registered' } as typeof state.runtime;
  state.artifacts = sizes.map((total_bytes, index) => ({ id: `asset-${index}`, total_bytes })) as typeof state.artifacts;
  vi.mocked(useModelCatalog).mockReturnValue(state);
  render(<ModelsPage />);
  expect(screen.getByText('注册模型资产总大小').parentElement?.querySelector('strong')).toHaveTextContent(total);
  expect(screen.queryByText('当前对话模型')).not.toBeInTheDocument();
  expect(screen.queryByText(state.runtime!.model!)).not.toBeInTheDocument();
});

it('已加载模型只提供卸载管理，不再重复提供对话模型选择', () => {
  const state = catalog();
  state.filteredInstances = ['ready', 'busy', 'loading'].map((status, index) => ({
    id: `instance-${index}`, model: `Qwen model ${index}`, state: status, context_size: 32768,
  })) as typeof state.filteredInstances;
  state.artifacts = state.filteredInstances.map((item) => ({ name: item.model, architecture: 'qwen35' })) as typeof state.artifacts;
  const { container } = render(<LoadedModels catalog={state} />);
  expect(screen.queryByRole('button', { name: '用于对话' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: '当前' })).not.toBeInTheDocument();
  const buttons = screen.getAllByRole('button', { name: '卸载' });
  expect(buttons).toHaveLength(3);
  expect(buttons[0]).toBeEnabled();
  expect(buttons[1]).toBeDisabled();
  expect(buttons[2]).toBeDisabled();
  fireEvent.click(buttons[0]);
  expect(state.unloadInstance).toHaveBeenCalledWith('instance-0');
  expect(container.querySelector('.model-row-actions')?.firstElementChild).toHaveClass('model-vendor-mark');
});
