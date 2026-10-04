/** Verify that the full-screen loading page presents a waiting state with consistent copy. */
import { render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { LoadingPage } from './LoadingPage';

vi.mock('../features/settings/SettingsProvider', () => ({
  useSettings: () => ({ tr: (_chinese: string, english: string) => english }),
}));

it('verifies LoadingPage test behavior 1', () => {
  render(<LoadingPage />);
  expect(screen.getByRole('status')).toHaveTextContent('Loading…');
  expect(screen.getByRole('heading', { name: 'Loading…' })).toBeTruthy();
  expect(screen.getByRole('status').querySelector('.loading-page-symbol')).toBeTruthy();
  expect(screen.queryByText('CONNECTION / 01')).toBeNull();
});
