/** Verify persisted interface languages and responsive translated pages in a real browser. */
import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

test('switches the interface, persists after reload, and follows the system again', async ({
  page,
}, testInfo) => {
  await mockStudioServer(page);
  await page.goto('/settings');
  await page
    .getByRole('combobox', { name: 'Interface language' })
    .selectOption('zh-CN');
  await page
    .getByRole('button', { name: 'Apply settings', exact: true })
    .first()
    .click();
  await expect(page.locator('html')).toHaveAttribute('lang', 'zh-CN');
  await expect(
    page.getByRole('heading', { name: '设置', exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole('region', { name: '通知提示' }).getByRole('status'),
  ).toContainText('设置已应用');
  await page.reload();
  await expect(page.getByRole('combobox', { name: '界面语言' })).toHaveValue(
    'zh-CN',
  );
  await expect(page.locator('html')).toHaveAttribute('lang', 'zh-CN');
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: testInfo.outputPath('settings-zh-CN.png'),
    fullPage: true,
  });
  await page.getByRole('combobox', { name: '界面语言' }).selectOption('en');
  await page
    .getByRole('button', { name: '应用设置', exact: true })
    .first()
    .click();
  await expect(
    page.getByRole('heading', { name: 'Settings', exact: true }),
  ).toBeVisible();
  await expect(page.locator('html')).toHaveAttribute('lang', 'en');
  await page.screenshot({
    path: testInfo.outputPath('settings-en.png'),
    fullPage: true,
  });
  await page
    .getByRole('combobox', { name: 'Interface language' })
    .selectOption('system');
  await page
    .getByRole('button', { name: 'Apply settings', exact: true })
    .first()
    .click();
  await page.reload();
  await expect(
    page.getByRole('combobox', { name: 'Interface language' }),
  ).toHaveValue('system');
  await expect(page.locator('html')).toHaveAttribute('lang', 'en');
  await page.goto('/chat');
  if (testInfo.project.name === 'mobile') {
    await page.getByRole('button', { name: 'Expand conversations' }).click();
  }
  await expect(
    page.getByRole('button', { name: 'New chat', exact: true }),
  ).toBeVisible();
  await expect(page.locator('body')).not.toContainText(
    /(?:chat|settings|common):[a-zA-Z]/,
  );
});
