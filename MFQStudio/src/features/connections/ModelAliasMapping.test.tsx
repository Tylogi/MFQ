import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { ApiError } from '../../shared/api/client';
import { ModelAliasMapping } from './ModelAliasMapping';

vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ connectionRevision: 0 }) }));
beforeEach(() => {
  vi.restoreAllMocks();
  vi.spyOn(runtimeApi, 'modelAliases').mockResolvedValue({ aliases: { 'model-a': 'original' } });
  vi.spyOn(runtimeApi, 'configureModelAliases').mockResolvedValue({ aliases: { 'model-a': 'new-name' } });
});
async function open() {
  const view = render(<ModelAliasMapping models={['model-a', 'model-b']} selectedModel="model-a" />);
  const button = screen.getByRole('button', { name: 'Alias mapping' });
  await waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(button);
  return view;
}
it('keeps the active alias outside the editor and immediately rejects duplicates', async () => {
  const view = await open();
  fireEvent.change(screen.getByLabelText('model-a alias'), { target: { value: 'model-b' } });
  expect(view.container.querySelector('strong')).toHaveTextContent('original');
  fireEvent.click(screen.getByRole('button', { name: 'Save' }));
  expect(screen.getByRole('alert')).toHaveTextContent('Model aliases must be unique');
  expect(screen.getByLabelText('model-a alias')).toHaveValue('original');
  expect(runtimeApi.configureModelAliases).not.toHaveBeenCalled();
});
it('restores saved aliases after a rejected change', async () => {
  vi.mocked(runtimeApi.configureModelAliases).mockRejectedValue(new ApiError(409, { error: { code: 'model_alias_conflict', message: 'internal mapping conflict', retryable: false, details: {} } }));
  await open();
  fireEvent.change(screen.getByLabelText('model-a alias'), { target: { value: 'new-name' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Model aliases must be unique');
  expect(screen.getByLabelText('model-a alias')).toHaveValue('original');
});
