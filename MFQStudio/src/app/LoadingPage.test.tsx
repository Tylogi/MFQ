/** 验证全屏加载页以统一文案呈现等待状态。 */
import { render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { LoadingPage } from './LoadingPage';

vi.mock('../features/settings/SettingsProvider', () => ({
  useSettings: () => ({ tr: (chinese: string) => chinese }),
}));

it('显示统一的加载文案和无障碍状态', () => {
  render(<LoadingPage />);
  expect(screen.getByRole('status')).toHaveTextContent('正在加载中');
  expect(screen.getByRole('heading', { name: '正在加载中' })).toBeTruthy();
  expect(screen.getByRole('status').querySelector('.loading-page-symbol')).toBeTruthy();
  expect(screen.queryByText('CONNECTION / 01')).toBeNull();
});
