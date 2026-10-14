import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

test('sidebar labels and icons use neutral ink in every theme', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  await page.goto('/');
  await expect(page.locator('main h1')).toBeVisible();
  if (testInfo.project.name === 'mobile') {
    await page.getByRole('button', { name: 'Open sidebar', exact: true }).click();
  }
  const nav = page.locator('#studio-sidebar .sectioned-nav');
  const active = nav.locator('button[aria-current="page"]');
  const models = nav.getByRole('button', { name: 'Models', exact: true });
  await expect(nav.getByRole('button', { name: 'Quantization', exact: true })).toHaveCount(0);
  await expect(nav.getByRole('button', { name: 'Resources', exact: true }).locator('svg > path'))
    .toHaveAttribute('d', 'M12 3C10 6.3 5 10.7 5 14a7 7 0 0 0 14 0c0-3.3-5-7.7-7-11z');
  for (const label of await nav.locator('button, .sidebar-group-label').all()) {
    await expect(label).toHaveCSS('font-weight', '400');
  }
  for (const theme of ['light', 'dark', 'system'] as const) {
    for (const scheme of theme === 'system' ? ['light', 'dark'] as const : [theme]) {
      await page.emulateMedia({ colorScheme: scheme });
      await page.evaluate(value => document.documentElement.dataset.theme = value, theme);
      const color = scheme === 'light' ? 'rgb(51, 51, 51)' : 'rgb(222, 222, 222)';
      for (const element of await nav.locator('button, .sidebar-group-label, .ui-icon').all()) {
        await expect(element).toHaveCSS('color', color);
      }
      const background = await active.evaluate(element => getComputedStyle(element).backgroundColor);
      expect(background).not.toBe('rgba(0, 0, 0, 0)');
      await models.hover();
      await expect(models).toHaveCSS('color', color);
      await expect(models.locator('.ui-icon')).toHaveCSS('color', color);
      await expect(active).toHaveCSS('background-color', background);
    }
  }
});
