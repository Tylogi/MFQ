import { useState } from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { SourceLoadingControl } from './SourceLoadingControl';
import { quantizationApi } from '../../shared/api/resources/quantization';

vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_: string, en: string) => en }) }));
vi.mock('../../shared/api/resources/quantization', () => ({ quantizationApi: { loadingPlan: vi.fn() } }));

function Control({ model = '/source', contextSize = 512 }: { model?: string; contextSize?: number }) {
  const [layerwise, setLayerwise] = useState(true);
  return <SourceLoadingControl model={model} purpose="wt2" contextSize={contextSize} layerwise={layerwise} onChange={setLayerwise} />;
}

beforeEach(() => vi.clearAllMocks());

it('defaults to layerwise with the recommendation', () => {
  render(<Control />);
  expect(screen.getByRole('checkbox', { name: 'Layerwise' })).toBeChecked();
  expect(screen.getByText('For larger models, recommended on most devices')).toBeInTheDocument();
  expect(quantizationApi.loadingPlan).not.toHaveBeenCalled();
});

it('rechecks and explains insufficient free memory when unchecked', async () => {
  vi.mocked(quantizationApi.loadingPlan).mockResolvedValue({ backend: 'metal', resident_allowed: false, available_bytes: 2 ** 30, resident_required_bytes: 4 * 2 ** 30, weights_bytes: 0, workspace_bytes: 0 });
  render(<Control />);
  fireEvent.click(screen.getByRole('checkbox', { name: 'Layerwise' }));
  await screen.findByText('Insufficient free memory (1.0 GiB available, 4.0 GiB estimated); layerwise loading restored.');
  expect(screen.getByRole('checkbox', { name: 'Layerwise' })).toBeChecked();
  expect(quantizationApi.loadingPlan).toHaveBeenCalledWith('/source', 'wt2', 512);
});

it('allows resident mode only after a successful check and resets on context change', async () => {
  vi.mocked(quantizationApi.loadingPlan).mockResolvedValue({ backend: 'metal', resident_allowed: true, available_bytes: 8 * 2 ** 30, resident_required_bytes: 4 * 2 ** 30, weights_bytes: 0, workspace_bytes: 0 });
  const view = render(<Control />);
  fireEvent.click(screen.getByRole('checkbox', { name: 'Layerwise' }));
  await waitFor(() => expect(screen.getByRole('checkbox', { name: 'Layerwise' })).not.toBeChecked());
  view.rerender(<Control contextSize={4096} />);
  expect(screen.getByRole('checkbox', { name: 'Layerwise' })).toBeChecked();
});

it('does not accept stale checks after changing source', async () => {
  let resolve!: (value: Awaited<ReturnType<typeof quantizationApi.loadingPlan>>) => void;
  vi.mocked(quantizationApi.loadingPlan).mockImplementation(() => new Promise((done) => { resolve = done; }));
  const view = render(<Control />);
  fireEvent.click(screen.getByRole('checkbox', { name: 'Layerwise' }));
  view.rerender(<Control model="/other" />);
  resolve({ backend: 'metal', resident_allowed: true, available_bytes: 8 * 2 ** 30, resident_required_bytes: 4 * 2 ** 30, weights_bytes: 0, workspace_bytes: 0 });
  await waitFor(() => expect(screen.getByRole('checkbox', { name: 'Layerwise' })).toBeChecked());
  expect(screen.queryByText(/Residency needs/)).not.toBeInTheDocument();
});

it('returns to layerwise if the resource check fails', async () => {
  vi.mocked(quantizationApi.loadingPlan).mockRejectedValue(new Error('offline'));
  render(<Control />);
  fireEvent.click(screen.getByRole('checkbox', { name: 'Layerwise' }));
  await screen.findByText('Available resources could not be verified; layerwise loading restored.');
  expect(screen.getByRole('checkbox', { name: 'Layerwise' })).toBeChecked();
});
