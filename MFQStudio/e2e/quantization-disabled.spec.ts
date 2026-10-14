import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

test('quantization has no navigation entry, model shortcut or reachable workspace', async ({ page }) => {
  const state = await mockStudioServer(page);
  await page.goto('/models');
  await expect(page.getByRole('heading', { name: 'Models', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Quantization', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Open quantization workspace' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Model downloads', exact: true })).toBeVisible();
  await page.goto('/quantization');
  await expect(page.getByRole('heading', { name: 'Page not found' })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Quantization workspace' })).toHaveCount(0);
  expect(state.requests.filter(request => request.includes('/api/v1/quantization/'))).toEqual([]);
  expect(state.unexpected).toEqual([]);
});
