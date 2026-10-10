import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['zh-CN', 'en'] as const) {
  test(`service and chat have no managed MCP dependency (${language})`, async ({ page }, testInfo) => {
    const state = await mockStudioServer(page, { language });
    await page.route('**/api/v1/runtime/model-aliases', route => route.fulfill({ json: { aliases: {} } }));
    await page.route('**/api/v1/runtime/memory-policy', route => route.fulfill({ json: {
      total_resident_budget_bytes: null, model_resident_budget_bytes: null, prefix_ram_budget_bytes: null,
    } }));
    await page.route('**/api/v1/runtime/context-policy', route => route.fulfill({ json: { max_context_size: null, model_overrides: {} } }));
    await page.route('**/api/v1/runtime/yarn/*', route => route.fulfill({ json: {
      supported: false, native_context: 32768, maximum_context: 32768, maximum_factor: 1, enabled: false,
    } }));
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('/runtime');
    await expect(page.getByText(language === 'zh-CN' ? '远程路由' : 'Remote routing', { exact: true })).toBeVisible();
    await expect(page.getByRole('textbox', { name: language === 'zh-CN' ? '节点名称' : 'Node name' })).toBeVisible();
    await expect(page.getByText('MCP', { exact: true })).toHaveCount(0);
    await expect.poll(() => state.requests.includes('GET /api/v1/cluster/nodes')).toBe(true);
    await page.locator('.remote-routing-grid').scrollIntoViewIfNeeded();
    expect(await page.locator('.remote-routing-grid .cluster-panel').evaluate(panel => {
      const bounds = panel.getBoundingClientRect();
      return [...panel.querySelectorAll('.node-form input, .node-form button')].every(control => {
        const rect = control.getBoundingClientRect();
        return rect.left >= bounds.left && rect.right <= bounds.right;
      });
    })).toBe(true);
    await page.screenshot({ path: testInfo.outputPath('remote-routing.png'), animations: 'disabled' });
    await page.goto('/chat');
    await expect(page.locator('main h1')).toBeVisible();
    expect(state.requests.some(request => request.includes('/mcp/'))).toBe(false);
    expect(state.unexpected).toEqual([]);
    expect(errors).toEqual([]);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  });
}
