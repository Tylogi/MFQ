import { render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import type { ModelArtifact } from '../../shared/api/types';
import type { useModelCatalog } from './useModelCatalog';
import { LocalCheckpoints } from './LocalCheckpoints';
import userEvent from '@testing-library/user-event';

vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('./ModelLoadProgress', () => ({ ModelLoadProgress: () => null }));

it('opens deletion confirmation by ID and prevents deleting loaded models', async () => {
  const artifact = { id: 'a', name: 'Model', architecture: 'qwen35', format: 'mfq', shard_count: 1,
    total_bytes: 100, tensor_count: 1, record_count: 1, dtypes: [], complete: true, loadable: true, modified_at: '' } as ModelArtifact;
  const prepareModelDeletion = vi.fn();
  const catalog = { runtime: null, artifacts: [artifact], filteredArtifacts: [artifact], instances: [], busy: false,
    modelFilter: '', setModelFilter: vi.fn(), unloadInstance: vi.fn(), loadArtifact: vi.fn(),
    chooseModelDirectory: vi.fn(), openModelFiles: vi.fn(), prepareModelDeletion } as unknown as ReturnType<typeof useModelCatalog>;
  const view = render(<LocalCheckpoints catalog={catalog} />);
  await userEvent.click(screen.getByRole('button', { name: 'Delete model: Model' }));
  expect(prepareModelDeletion).toHaveBeenCalledExactlyOnceWith('a');
  catalog.runtime = { model: 'Model' };
  view.rerender(<LocalCheckpoints catalog={catalog} />);
  expect(screen.getByRole('button', { name: 'Delete model: Model' })).toBeDisabled();
});

it('uses one combined MoE pressure bar without capacity details, matching the download page', () => {
  const GiB = 2 ** 30;
  const artifact: ModelArtifact = { id: 'a', name: 'MoE', architecture: 'qwen4_exp', format: 'mfq',
    shard_count: 1, total_bytes: 133 * GiB, estimated_resident_weight_bytes: 83 * GiB, ssd_ple_bytes: 50 * GiB,
    estimated_weight_bytes_by_role: { dense: 2 * GiB, experts: 80 * GiB, embedding: GiB },
    tensor_count: 1, record_count: 1, dtypes: [], complete: true, loadable: true, modified_at: '2026-10-11' };
  const catalog = { runtime: { memory_architecture: 'discrete', device_memory_total_bytes: 24 * GiB,
    host_memory_total_bytes: 64 * GiB, runtime_memory_headroom_bytes: 8 * GiB },
    artifacts: [artifact], filteredArtifacts: [artifact], instances: [], busy: false,
    modelFilter: '', setModelFilter: vi.fn(), unloadInstance: vi.fn(), loadArtifact: vi.fn(),
    chooseModelDirectory: vi.fn(), openModelFiles: vi.fn() } as unknown as ReturnType<typeof useModelCatalog>;
  render(<LocalCheckpoints catalog={catalog} />);
  expect(screen.getByRole('progressbar', { name: 'VRAM + RAM pressure: 94.3%' })).toBeInTheDocument();
  expect(screen.getAllByRole('progressbar')).toHaveLength(1);
  expect(document.querySelector('.variant-memory-caption')).toBeNull();
  expect(screen.getByText(/SSD PLE 50/)).toBeInTheDocument();
});

it('shows the same zero shared budget as the model-loading estimate, not raw free memory', () => {
  const artifact: ModelArtifact = { id: 'a', name: 'MoE', architecture: 'qwen4_exp', format: 'mfq',
    shard_count: 1, total_bytes: 2 ** 30, estimated_resident_weight_bytes: 2 ** 30,
    tensor_count: 1, record_count: 1, dtypes: [], complete: true, loadable: true, modified_at: '2026-10-18' };
  const catalog = { runtime: { memory_architecture: 'unified', host_memory_available_bytes: 3 * 2 ** 30,
    runtime_memory_headroom_bytes: 0 }, artifacts: [artifact], filteredArtifacts: [artifact], instances: [], busy: false,
    modelFilter: '', setModelFilter: vi.fn(), unloadInstance: vi.fn(), loadArtifact: vi.fn(),
    chooseModelDirectory: vi.fn(), openModelFiles: vi.fn() } as unknown as ReturnType<typeof useModelCatalog>;
  render(<LocalCheckpoints catalog={catalog} />);
  expect(screen.getByText('Remaining available memory 0 GiB')).toBeInTheDocument();
  expect(screen.getByRole('progressbar', { name: 'Estimated share of remaining available memory: No available memory' })).toBeInTheDocument();
});

it.each(['unified', 'discrete'] as const)('allows loading a low-memory MoE and describes SSD streaming on %s memory', async (memory_architecture) => {
  const GiB = 2 ** 30;
  const artifact: ModelArtifact = { id: 'a', name: 'MoE', architecture: 'new-moe-architecture', format: 'mfq',
    shard_count: 1, total_bytes: 100 * GiB, estimated_resident_weight_bytes: 80 * GiB, ssd_ple_bytes: 20 * GiB,
    estimated_weight_bytes_by_role: { dense: 4 * GiB, experts: 76 * GiB },
    tensor_count: 1, record_count: 1, dtypes: [], complete: true, loadable: true, modified_at: '' };
  const loadArtifact = vi.fn();
  const catalog = { runtime: { memory_architecture, host_memory_available_bytes: 16 * GiB,
    runtime_memory_headroom_bytes: 16 * GiB, device_free_bytes: 8 * GiB }, loadDeviceIds: [],
    artifacts: [artifact], filteredArtifacts: [artifact], instances: [], busy: false,
    modelFilter: '', setModelFilter: vi.fn(), unloadInstance: vi.fn(), loadArtifact,
    chooseModelDirectory: vi.fn(), openModelFiles: vi.fn() } as unknown as ReturnType<typeof useModelCatalog>;
  const view = render(<LocalCheckpoints catalog={catalog} />);
  const button = screen.getByRole('button', { name: 'Load' });
  expect(button).toBeEnabled();
  expect(button).toHaveAccessibleDescription('Insufficient remaining memory. Experts that do not fit will be streamed from SSD during inference.');
  await userEvent.click(button);
  expect(loadArtifact).toHaveBeenCalledExactlyOnceWith('MoE');
  catalog.busy = true;
  view.rerender(<LocalCheckpoints catalog={catalog} />);
  expect(button).toBeDisabled();
});

it.each([
  { experts: 0, resident: 80, loadable: true, available: 16 },
  { experts: 76, resident: 16, loadable: true, available: 16 },
  { experts: 76, resident: 80, loadable: false, available: 16 },
  { experts: 76, resident: 80, loadable: true, available: undefined },
])('does not claim SSD expert streaming for a dense, fitting, invalid or unmeasured model: %j', ({ experts, resident, loadable, available }) => {
  const artifact: ModelArtifact = { id: 'a', name: 'Model', architecture: 'qwen4_exp', format: 'mfq',
    shard_count: 1, total_bytes: 100, estimated_resident_weight_bytes: resident,
    estimated_weight_bytes_by_role: { experts }, tensor_count: 1, record_count: 1,
    dtypes: [], complete: loadable, loadable, modified_at: '' };
  const catalog = { runtime: { runtime_memory_headroom_bytes: available },
    artifacts: [artifact], filteredArtifacts: [artifact], instances: [], busy: false,
    modelFilter: '', setModelFilter: vi.fn(), unloadInstance: vi.fn(), loadArtifact: vi.fn(),
    chooseModelDirectory: vi.fn(), openModelFiles: vi.fn() } as unknown as ReturnType<typeof useModelCatalog>;
  render(<LocalCheckpoints catalog={catalog} />);
  expect(screen.queryByText(/Experts that do not fit/)).not.toBeInTheDocument();
  if (!loadable) expect(screen.queryByRole('button', { name: 'Load' })).not.toBeInTheDocument();
});
