/** Verify model panel actions and status rendering with English interface labels. */
import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { useModelCatalog } from './useModelCatalog';
import { ModelsPage } from './ModelsPage';
import { LoadedModels } from './LoadedModels';
import { LocalCheckpoints } from './LocalCheckpoints';
import { ModelLoadPolicy } from './ModelLoadPolicy';

vi.mock('../settings/SettingsProvider', () => ({
  useSettings: () => ({ tr: (_zh: string, en: string) => en }),
}));
vi.mock('./useModelCatalog', () => ({ useModelCatalog: vi.fn() }));

function catalog() {
  return {
    runtime: null,
    busy: false,
    unloadingInstanceIds: new Set<string>(),
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

it('verifies ModelPanels test behavior 1', () => {
  const state = catalog();
  render(<><LoadedModels catalog={state} /><LocalCheckpoints catalog={state} /></>);
  fireEvent.click(screen.getByRole('button', { name: 'Add model' }));
  fireEvent.click(screen.getByRole('button', { name: 'Choose model folder' }));
  expect(state.chooseModelDirectory).toHaveBeenCalledTimes(2);
});

it('verifies ModelPanels test behavior 2', () => {
  const state = catalog();
  render(<ModelLoadPolicy catalog={state} />);
  fireEvent.click(screen.getByRole('switch', { name: 'Pin in memory' }));
  fireEvent.change(screen.getByRole('combobox'), { target: { value: '900' } });
  expect(state.setLoadPinned).toHaveBeenCalledWith(true);
  expect(state.setLoadIdleTtl).toHaveBeenCalledWith(900);
});

it('verifies ModelPanels test behavior 3', () => {
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
  fireEvent.click(screen.getAllByRole('button', { name: 'Load' })[0]);
  expect(state.loadArtifact).toHaveBeenCalledWith('renamed-checkpoint');
});

it.each([[[], '0 B'], [[32 * 2 ** 30, 8 * 2 ** 30], '40 GiB']] as [number[], string][])('verifies ModelPanels test behavior 4', (sizes, total) => {
  const state = catalog();
  state.runtime = { model: 'Loaded but not registered' } as typeof state.runtime;
  state.artifacts = sizes.map((total_bytes, index) => ({ id: `asset-${index}`, total_bytes })) as typeof state.artifacts;
  vi.mocked(useModelCatalog).mockReturnValue(state);
  render(<ModelsPage />);
  expect(screen.getByText('Registered model assets size').parentElement?.querySelector('strong')).toHaveTextContent(total);
  expect(screen.queryByText('Current chat model')).not.toBeInTheDocument();
  expect(screen.queryByText(state.runtime!.model!)).not.toBeInTheDocument();
});

it('verifies ModelPanels test behavior 5', () => {
  const state = catalog();
  state.filteredInstances = ['ready', 'busy', 'loading'].map((status, index) => ({
    id: `instance-${index}`, model: `Qwen model ${index}`, state: status, context_size: 32768,
  })) as typeof state.filteredInstances;
  state.artifacts = state.filteredInstances.map((item) => ({ name: item.model, architecture: 'qwen35' })) as typeof state.artifacts;
  const { container } = render(<LoadedModels catalog={state} />);
  expect(screen.queryByRole('button', { name: 'Use for chat' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Current' })).not.toBeInTheDocument();
  const buttons = screen.getAllByRole('button', { name: 'Unload' });
  expect(buttons).toHaveLength(3);
  expect(buttons[0]).toBeEnabled();
  expect(buttons[1]).toBeDisabled();
  expect(buttons[2]).toBeDisabled();
  fireEvent.click(buttons[0]);
  expect(state.unloadInstance).toHaveBeenCalledWith('instance-0');
  expect(container.querySelector('.model-row-actions')?.firstElementChild).toHaveClass('model-vendor-mark');
});


it.each(['pending', 'unloading'])('shows %s unload feedback in both model panels', (phase) => {
  const state = catalog();
  const instance = { id: 'instance-1', model: 'Local Model', state: phase === 'pending' ? 'ready' : 'unloading' } as typeof state.instances[number];
  state.instances = state.filteredInstances = [instance];
  state.artifacts = state.filteredArtifacts = [{ id: 'asset-1', name: instance.model, loadable: true }] as typeof state.artifacts;
  if (phase === 'pending') state.unloadingInstanceIds.add(instance.id);
  render(<><LoadedModels catalog={state} /><LocalCheckpoints catalog={state} /></>);
  const buttons = screen.getAllByRole('button', { name: 'Unloading…' });
  expect(buttons).toHaveLength(2);
  buttons.forEach((button) => {
    expect(button).toBeDisabled();
    fireEvent.click(button);
  });
  expect(state.unloadInstance).not.toHaveBeenCalled();
});
