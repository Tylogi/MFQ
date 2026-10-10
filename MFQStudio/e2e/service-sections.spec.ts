import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['zh-CN', 'en'] as const) {
  test(`service keeps context management separate from memory and persistent cache (${language})`, async ({ page }, testInfo) => {
    const state = await mockStudioServer(page, { language });
    await page.route('**/api/v1/runtime/instances', route => route.fulfill({ json: { data: [{
      id: 'instance-1', model: 'Qwen3.8-Flash-Next-EWQ-MFQ-S4-L-MHC-NINT-LMHead8-MTP8', state: 'ready',
      devices: ['metal'], active_sessions: 0, queued_requests: 0, context_size: 8192, context_capacity: 262144,
    }] } }));
    await page.goto('/runtime');
    await expect(page.locator('.server-page > .section-label')).toHaveText(language === 'zh-CN'
      ? ['API参数', '内存规划', '上下文管理', '持久化前缀缓存', '自动化']
      : ['API parameters', 'Memory plan', 'Context management', 'Persistent prefix cache', 'Automation']);
    const contexts = page.locator('.server-context-panel');
    await expect(contexts.locator('.model-context-settings')).toHaveCount(1);
    await expect(contexts.locator('.model-context-entry').getByRole('spinbutton')).toHaveValue('');
    await expect(page.locator('.server-memory-panel .model-context-settings')).toHaveCount(0);
    await expect(page.locator('.server-memory-panel')).not.toContainText(language === 'zh-CN' ? '前缀块大小' : 'Prefix block size');
    await expect(page.locator('.server-prefix-panel')).toContainText(language === 'zh-CN' ? '前缀块大小' : 'Prefix block size');
    await expect(page.locator('.server-prefix-panel').getByRole('checkbox')).toBeDisabled();
    await expect(page.getByRole('spinbutton', { name: language === 'zh-CN' ? 'OpenAI 端口' : 'OpenAI port', exact: true })).toBeEnabled();
    await contexts.scrollIntoViewIfNeeded();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath('service-context-section.png'), animations: 'disabled', fullPage: true });
    expect(state.requests.filter(request => /^(POST|PUT|DELETE) /.test(request))).toEqual([]);
  });
}
