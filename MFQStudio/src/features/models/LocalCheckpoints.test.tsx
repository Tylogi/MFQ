import { render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import type { ModelArtifact } from '../../shared/api/types';
import type { useModelCatalog } from './useModelCatalog';
import { LocalCheckpoints } from './LocalCheckpoints';

vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('./ModelLoadProgress', () => ({ ModelLoadProgress: () => null }));

it('uses the same VRAM and RAM estimates as the download page', () => {
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
  expect(screen.getByRole('progressbar', { name: 'VRAM pressure: 8.3%' })).toBeInTheDocument();
  expect(screen.getByRole('progressbar', { name: 'RAM pressure: 126.6%' })).toBeInTheDocument();
  expect(screen.getByText(/SSD PLE 50/)).toBeInTheDocument();
});
