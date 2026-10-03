import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { useModelCatalog } from './useModelCatalog';
import { ModelsPage } from './ModelsPage';
import { LoadedModels } from './LoadedModels';
import { LocalCheckpoints } from './LocalCheckpoints';
import { ModelLoadPolicy } from './ModelLoadPolicy';

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
    filteredInstances: [],
    filteredArtifacts: [],
    loadPinned: false,
    loadIdleTtl: null,
    chooseModelDirectory: vi.fn(),
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
