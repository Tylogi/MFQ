/** Verify that the full-screen loading page presents a waiting state with consistent copy. */
import { i18n } from '../i18n';
import { render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { LoadingPage } from './LoadingPage';

vi.mock('../features/settings/SettingsProvider', () => ({
  useSettings: () => ({ t: i18n.getFixedT('en') }),
}));

it('verifies LoadingPage test behavior 1', () => {
  render(<LoadingPage />);
  expect(screen.getByRole('status')).toHaveTextContent('Loading…');
  expect(screen.getByRole('heading', { name: 'Loading…' })).toBeTruthy();
  expect(screen.getByRole('status').querySelector('.loading-page-symbol')).toBeTruthy();
  expect(screen.queryByText('CONNECTION / 01')).toBeNull();
});
