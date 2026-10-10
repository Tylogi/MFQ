import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['zh-CN', 'en'] as const) {
  test(`lightweight applications belong to Playground (${language})`, async ({ page }, testInfo) => {
    const state = await mockStudioServer(page, { language });
    const en = language === 'en';
    await page.addInitScript(() => {
      Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
        writeText: async (value: string) => { document.documentElement.dataset.copiedConfiguration = value; },
      } });
    });
    await page.route('**/api/v1/runtime/model-aliases', route => route.fulfill({ json: { aliases: { 'Studio Test Model': 'my-model' } } }));
    await page.goto('/applications');
    const groups = page.locator('.sectioned-nav > section');
    const interaction = groups.filter({ has: page.locator('.sidebar-group-label', { hasText: en ? 'Playground' : '交互' }) });
    const tools = groups.filter({ has: page.locator('.sidebar-group-label', { hasText: en ? 'Tools' : '工具' }) });
    await expect(interaction.getByRole('button', { name: en ? 'Applications' : '应用', exact: true })).toHaveCount(1);
    await expect(tools.getByRole('button', { name: en ? 'Applications' : '应用', exact: true })).toHaveCount(0);
    await expect(page.locator('.application-entry')).toHaveCount(5);
    await expect(page.getByRole('combobox', { name: en ? 'Model' : '模型' })).toHaveValue('my-model');
    await expect(page.getByText('http://127.0.0.1:8091/v1/messages', { exact: true })).toBeVisible();
    const opencode = page.locator('.application-entry').filter({ has: page.getByRole('heading', { name: 'OpenCode', exact: true }) });
    await opencode.getByRole('button', { name: en ? 'Copy configuration' : '复制配置' }).click();
    const copied = await page.locator('html').getAttribute('data-copied-configuration');
    expect(JSON.parse(copied!).model).toBe('mfq/my-model');
    await opencode.locator('summary').click();
    await expect(opencode.locator('pre')).toContainText('YOUR_MFQ_API_KEY');
    await expect(page.getByRole('button', { name: /一键配置|打开安装|启动应用|Configure and connect|Open installer|Launch application/ })).toHaveCount(0);
    expect(state.requests.filter(request => /^(POST|PUT|DELETE) /.test(request))).toEqual([]);
    expect(state.requests.some(request => request.includes('/applications'))).toBe(false);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.locator('.applications-page').screenshot({ path: testInfo.outputPath('applications.png'), animations: 'disabled' });
  });
}
