import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['zh-CN', 'en'] as const) {
  test(`overview shows and copies both actual API URLs and refreshes changed ports (${language})`, async ({ page }, testInfo) => {
    const state = await mockStudioServer(page, { language });
    let port: number | null = 8091;
    await page.addInitScript(() => {
      Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
        writeText: async (value: string) => { document.documentElement.dataset.copiedEndpoint = value; },
      } });
    });
    await page.route('**/api/v1/runtime/listener', route => route.fulfill({ json: {
      host: '127.0.0.1', port: 8090, anthropic_port: port, configurable: true,
    } }));
    await page.goto('/');
    const panel = page.locator('.overview-endpoint-panel');
    await expect(panel.locator('code')).toHaveText(['http://127.0.0.1:8090/v1', 'http://127.0.0.1:8091/v1/messages']);
    await panel.getByRole('button', { name: language === 'en' ? 'Copy OpenAI endpoint' : '复制 OpenAI 端点' }).click();
    await expect(page.locator('html')).toHaveAttribute('data-copied-endpoint', 'http://127.0.0.1:8090/v1');
    await panel.getByRole('button', { name: language === 'en' ? 'Copy Anthropic endpoint' : '复制 Anthropic 端点' }).click();
    await expect(page.locator('html')).toHaveAttribute('data-copied-endpoint', 'http://127.0.0.1:8091/v1/messages');
    port = 8092;
    await page.evaluate(() => dispatchEvent(new Event('focus')));
    await expect(panel.locator('code')).toHaveText(['http://127.0.0.1:8090/v1', 'http://127.0.0.1:8092/v1/messages']);
    await panel.scrollIntoViewIfNeeded();
    const endpointBounds = await panel.boundingBox();
    const sessionBounds = await page.locator('.overview-session-panel').boundingBox();
    expect(endpointBounds!.height).toBeLessThanOrEqual(190);
    expect(sessionBounds!.height).toBeLessThanOrEqual(190);
    expect(await panel.locator('.overview-api-entry').first().evaluate(element => {
      const label = element.querySelector('strong')!.getBoundingClientRect();
      const url = element.querySelector('code')!.getBoundingClientRect();
      return Math.abs((label.top + label.height / 2) - (url.top + url.height / 2)) < 2;
    })).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.locator('.overview-footer-grid').screenshot({ path: testInfo.outputPath('overview-api-endpoints.png'), animations: 'disabled' });
    port = null;
    await page.evaluate(() => dispatchEvent(new Event('focus')));
    await expect(panel.getByRole('button', { name: language === 'en' ? 'Copy Anthropic endpoint' : '复制 Anthropic 端点' })).toBeDisabled();
    expect(state.requests.filter(request => /^(POST|PUT|DELETE) /.test(request))).toEqual([]);
  });

  test(`independent API ports apply on blur and Enter with error feedback (${language})`, async ({ page }, testInfo) => {
    await mockStudioServer(page, { language });
    const writes: unknown[] = [];
    let anthropicPort = 8091;
    await page.route('**/api/v1/runtime/listener', async route => {
      if (route.request().method() === 'PUT') {
        const body = route.request().postDataJSON();
        writes.push(body);
        if (body.port === 8090) {
          return route.fulfill({ status: 409, json: { error: { code: 'listener_change_failed', message: 'port in use' } } });
        }
        anthropicPort = body.port;
      }
      return route.fulfill({ json: { host: '127.0.0.1', port: 8090, anthropic_port: anthropicPort, configurable: true } });
    });
    await page.goto('/runtime');
    const panel = page.locator('.server-api-panel');
    const oai = panel.getByRole('spinbutton', { name: language === 'en' ? 'OpenAI port' : 'OpenAI 端口' });
    const anthropic = panel.getByRole('spinbutton', { name: language === 'en' ? 'Anthropic port' : 'Anthropic 端口' });
    await expect(anthropic).toHaveValue('8091');
    await expect(panel.getByRole('button', { name: /Save server|保存服务器/ })).toHaveCount(0);
    expect(writes).toEqual([]);
    await anthropic.fill('8092');
    expect(writes).toEqual([]);
    await anthropic.press('Enter');
    await expect.poll(() => writes).toEqual([{ port: 8092, protocol: 'anthropic' }]);
    await expect(anthropic).toBeEnabled();
    await anthropic.blur();
    expect(writes).toHaveLength(1);
    await expect(panel.getByText('http://127.0.0.1:8092/v1/messages')).toBeVisible();
    await expect(oai).toHaveValue('8090');
    await anthropic.fill('8090');
    await anthropic.blur();
    await expect(panel.getByRole('alert')).toContainText('port in use');
    await expect(panel.getByText('http://127.0.0.1:8092/v1/messages')).toBeVisible();
    await anthropic.fill('65536');
    await anthropic.blur();
    await expect(panel.getByRole('alert')).toHaveText(language === 'en'
      ? 'Port must be an integer between 1 and 65535' : '端口必须为 1–65535 的整数');
    expect(writes).toHaveLength(2);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await panel.screenshot({ path: testInfo.outputPath('api-ports.png'), animations: 'disabled' });
  });
}
