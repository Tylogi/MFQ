import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it, vi } from 'vitest';
import type { useModelCatalog } from './useModelCatalog';
import { ModelDeleteDialog } from './ModelDeleteDialog';

vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));

function fixture() {
  return { modelDeletion: { model_id: 'model', name: 'Test model', files: ['/models/model-00001-of-00002.mfq', '/models/model-00002-of-00002.mfq'], total_bytes: 2 ** 30, token: 'a'.repeat(64) },
    busy: false, cancelModelDeletion: vi.fn(), confirmModelDeletion: vi.fn(), deletionTriggerRef: { current: null } } as unknown as ReturnType<typeof useModelCatalog>;
}

it('shows the actual files and irreversible warning; only confirmation deletes', async () => {
  const catalog = fixture();
  render(<ModelDeleteDialog catalog={catalog} />);
  expect(screen.getByRole('dialog', { name: 'Delete model' })).toBeInTheDocument();
  expect(screen.getByText(/permanently deleted/)).toBeInTheDocument();
  expect(screen.getByText('Test model')).toBeInTheDocument();
  expect(screen.getByRole('list', { name: 'Files to delete' }).children).toHaveLength(2);
  expect(catalog.confirmModelDeletion).not.toHaveBeenCalled();
  await userEvent.click(screen.getAllByRole('button', { name: 'Cancel' }).at(-1)!);
  expect(catalog.cancelModelDeletion).toHaveBeenCalledOnce();
  expect(catalog.confirmModelDeletion).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole('button', { name: 'Confirm deletion' }));
  expect(catalog.confirmModelDeletion).toHaveBeenCalledOnce();
});

it('disables confirmation and cancellation during deletion', () => {
  const catalog = fixture();
  catalog.busy = true;
  render(<ModelDeleteDialog catalog={catalog} />);
  expect(screen.getByRole('button', { name: 'Deleting…' })).toBeDisabled();
  expect(screen.getAllByRole('button', { name: 'Cancel' }).at(-1)).toBeDisabled();
});
