import { render } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { QuantizationPage } from './QuantizationPage';
import { useQuantizationWorkspace } from './useQuantizationWorkspace';

vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('./useQuantizationWorkspace', () => ({ useQuantizationWorkspace: vi.fn() }));

it('leaves four empty windows without mounting forms, task APIs or artifact operations', () => {
  const { container } = render(<QuantizationPage />);
  expect(container.querySelectorAll('.quantization-empty-panel')).toHaveLength(4);
  expect(container.querySelectorAll('button, input, select, form')).toHaveLength(0);
  expect(container.textContent).toBe('');
  expect(useQuantizationWorkspace).not.toHaveBeenCalled();
});
