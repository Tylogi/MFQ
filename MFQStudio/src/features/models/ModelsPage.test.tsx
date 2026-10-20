import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { beforeEach, expect, it, vi } from 'vitest';
import { ModelsPage } from './ModelsPage';

vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('./useModelCatalog', () => ({ useModelCatalog: () => ({ artifacts: [], busy: false, availableModelNames: [], modelFolderPath: '/models' }) }));
vi.mock('./LoadedModels', () => ({ LoadedModels: () => null }));
vi.mock('./ModelLoadPolicy', () => ({ ModelLoadPolicy: () => null }));
vi.mock('./LocalCheckpoints', () => ({ LocalCheckpoints: () => <h2>Local checkpoints list</h2> }));
vi.mock('./ModelDirectoryDialog', () => ({ ModelDirectoryDialog: () => null }));

const scroll = vi.fn();
beforeEach(() => { scroll.mockClear(); Element.prototype.scrollIntoView = scroll; });

it('positions the local checkpoint section when opened through the Load shortcut', () => {
  render(<MemoryRouter initialEntries={['/models#local-checkpoints']}><ModelsPage /></MemoryRouter>);
  const section = screen.getByText('Local checkpoints list').closest('section')!;
  expect(section.id).toBe('local-checkpoints');
  expect(scroll).toHaveBeenCalledExactlyOnceWith({ block: 'start' });
  expect(scroll.mock.instances[0]).toBe(section);
});

it('keeps normal model-page navigation at its original position', () => {
  render(<MemoryRouter initialEntries={['/models']}><ModelsPage /></MemoryRouter>);
  expect(scroll).not.toHaveBeenCalled();
});
