/** 验证模型面板拆分后仍将交互交给目录控制器。 */
import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import type { useModelCatalog } from './useModelCatalog';
import { LoadedModels } from './LoadedModels';
import { LocalCheckpoints } from './LocalCheckpoints';
import { ModelLoadPolicy } from './ModelLoadPolicy';

vi.mock('../settings/SettingsProvider', () => ({
  useSettings: () => ({ tr: (zh: string) => zh }),
}));

/** 构造供展示面板使用的最小模型目录。 */
function catalog() {
  return {
    runtime: null,
    model: '',
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
    selectModel: vi.fn(),
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
  fireEvent.click(screen.getByRole('checkbox', { name: '固定到内存' }));
  fireEvent.change(screen.getByRole('combobox'), { target: { value: '900' } });
  expect(state.setLoadPinned).toHaveBeenCalledWith(true);
  expect(state.setLoadIdleTtl).toHaveBeenCalledWith(900);
});
