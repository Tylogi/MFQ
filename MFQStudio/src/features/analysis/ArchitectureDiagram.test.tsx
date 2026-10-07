import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { expect, it } from 'vitest';
import { analysisFixture } from '../../../tests/fixtures/checkpointAnalysis';
import { ArchitectureDiagram } from './ArchitectureDiagram';
const tr = (_zh: string, en: string) => en;

it('opens component subgraphs from overview nodes and supports expanded inspection', async () => {
  render(<ArchitectureDiagram analysis={analysisFixture()} tr={tr} />);
  expect(screen.getByText('3 × [3 × GDN → QSA]')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Inspect GDN' }));
  expect(screen.getByRole('img', { name: 'GDN · architecture detail' })).toBeInTheDocument();
  expect(screen.getByText('Causal Conv')).toBeInTheDocument();
  expect(screen.getByText('SiLU')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('tab', { name: 'MoE' }));
  expect(screen.getByText('Shared expert')).toBeInTheDocument();
  const trigger = screen.getByRole('button', { name: 'Expand architecture' });
  trigger.focus(); fireEvent.click(trigger);
  const dialog = await screen.findByRole('dialog', { name: 'Architecture details' });
  fireEvent.click(within(dialog).getByRole('tab', { name: 'QSA' }));
  expect(within(dialog).getByText('Top-k blocks')).toBeInTheDocument();
  fireEvent.click(within(dialog).getByRole('button', { name: 'Close architecture' }));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  await waitFor(() => expect(trigger).toHaveFocus());
});
