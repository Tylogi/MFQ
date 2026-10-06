/** Verify QuantizationPage behavior and integration contracts. */
import { i18n } from '../../i18n';
import { render } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { QuantizationPage } from './QuantizationPage';
import { useQuantizationWorkspace } from './useQuantizationWorkspace';

vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ t: i18n.getFixedT('en') }) }));
vi.mock('./useQuantizationWorkspace', () => ({ useQuantizationWorkspace: vi.fn() }));

it('leaves four empty windows without mounting forms, task APIs or artifact operations', () => {
  const { container } = render(<QuantizationPage />);
  expect(container.querySelectorAll('.quantization-empty-panel')).toHaveLength(4);
  expect(container.querySelectorAll('button, input, select, form')).toHaveLength(0);
  expect(container.textContent).toBe('');
  expect(useQuantizationWorkspace).not.toHaveBeenCalled();
});
