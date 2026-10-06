/** Verify themed native pickers, keyboard selection, and responsive menu placement. */
import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

test('styles closed and expanded selectors while retaining native keyboard behavior', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  await page.goto('/settings');
  const theme = page.getByRole('combobox', { name: 'Theme', exact: true });
  await expect(page.getByRole('combobox', { name: 'Saved presets' })).toBeDisabled();
  await theme.scrollIntoViewIfNeeded();
  await theme.click();
  await expect(page.getByRole('option', { name: 'Dark', exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('select-light.png') });
  await theme.press('Escape');
  await expect(theme).toBeFocused();
  await expect(theme).toHaveValue('light');
  await theme.press('Space');
  await page.keyboard.press('End');
  await page.keyboard.press('Enter');
  await expect(theme).toHaveValue('dark');
  await page.getByRole('button', { name: 'Apply settings', exact: true }).first().click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await theme.scrollIntoViewIfNeeded();
  await theme.click();
  await expect(page.getByRole('option', { name: 'Dark', exact: true })).toBeVisible();
  const menuOption = await page.getByRole('option', { name: 'Dark', exact: true }).boundingBox();
  expect(menuOption).not.toBeNull();
  expect(menuOption!.x).toBeGreaterThanOrEqual(0);
  expect(menuOption!.x + menuOption!.width).toBeLessThanOrEqual(page.viewportSize()!.width);
  await page.screenshot({ path: testInfo.outputPath('select-dark.png') });
  await theme.press('Escape');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
