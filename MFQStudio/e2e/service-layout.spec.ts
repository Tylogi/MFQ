import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['zh-CN', 'en'] as const) {
  test(`service layout keeps model controls aligned, readable and separate (${language})`, async ({ page }, testInfo) => {
    const state = await mockStudioServer(page, { language });
    const names = ['Qwen3.8-Flash-Next-EWQ-MFQ-S4-L-MHC-NINT-LMHead8-MTP8', 'Qwen3.8-27B-EWQ-MFQ-S4-M'];
    const instances = names.map((model, index) => ({ id: `instance-${index + 1}`, model, state: 'ready',
      devices: ['metal'], active_sessions: 0, queued_requests: 0, context_size: index ? 65536 : 262144,
      context_capacity: 262144, resident_bytes: 2 ** 30 }));
    await page.route('**/api/v1/runtime/instances', route => route.fulfill({ json: { data: instances } }));
    await page.route('**/api/v1/runtime/models', route => route.fulfill({ json: { data: names.map(id => ({ id })) } }));
    await page.route('**/api/v1/runtime/status*', route => route.fulfill({ json: { runtime_state: 'ready',
      instance_id: 'instance-1', model: names[0], context_capacity: 262144, mtp_service_enabled: false } }));
    const saves: unknown[] = [];
    const updates: unknown[] = [];
    await page.route('**/api/v1/runtime/context', route => {
      const payload = route.request().postDataJSON();
      updates.push(payload);
      return route.fulfill({ json: { id: 'context-layout-job', kind: 'runtime.context.configure', status: 'running',
        payload, progress: .3, progress_data: { phase: 'reloading' }, created_at: '2026-10-09T00:00:00Z',
        updated_at: '2026-10-09T00:00:00Z', cancel_requested: false } });
    });
    await page.route('**/api/v1/runtime/context-policy', route => {
      if (route.request().method() === 'PUT') saves.push(route.request().postDataJSON());
      return route.fulfill({ json: { max_context_size: route.request().method() === 'PUT' ? 65536 : null,
        model_overrides: {}, fallback_context_size: 32768 } });
    });
    await page.goto('/runtime');
    const panel = page.locator('.server-context-panel');
    await expect(panel.locator('.model-context-entry')).toHaveCount(2);
    const longTitle = panel.locator('.model-context-entry strong').first();
    await expect(longTitle).toHaveAttribute('title', names[0]);
    expect(await longTitle.evaluate(element => {
      const style = getComputedStyle(element);
      return style.whiteSpace === 'nowrap' && style.textOverflow === 'ellipsis';
    })).toBe(true);
    const metrics = await panel.locator('.model-context-settings').evaluate(element => {
      const style = getComputedStyle(element);
      return { top: style.marginTop, padding: style.paddingTop, border: style.borderTopWidth };
    });
    expect(metrics).toEqual({ top: '0px', padding: '0px', border: '0px' });
    await expect(page.locator('.server-api-panel').getByRole('button', {
      name: language === 'zh-CN' ? '保存服务器设置' : 'Save server settings', exact: true,
    })).toHaveCount(0);
    const entries = panel.locator('.model-context-entry');
    const first = (await entries.nth(0).getByRole('spinbutton').boundingBox())!;
    const second = (await entries.nth(1).getByRole('spinbutton').boundingBox())!;
    expect(Math.abs(first.x - second.x)).toBeLessThanOrEqual(1);
    const firstAction = (await entries.nth(0).getByRole('button').boundingBox())!;
    expect((await entries.nth(1).boundingBox())!.y - firstAction.y - firstAction.height).toBeGreaterThanOrEqual(12);
    await panel.scrollIntoViewIfNeeded();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await panel.screenshot({ path: testInfo.outputPath('service-context-layout.png'), animations: 'disabled' });
    await page.evaluate(() => document.documentElement.dataset.theme = 'dark');
    await panel.screenshot({ path: testInfo.outputPath('service-context-layout-dark.png'), animations: 'disabled' });
    const global = panel.getByRole('spinbutton', { name: language === 'zh-CN' ? '全局上下文上限' : 'Global context cap', exact: true });
    if (await global.count()) {
      await global.fill('65536');
      await panel.getByRole('button', { name: language === 'zh-CN' ? '保存全局上下文上限' : 'Save global context cap' }).click();
      await expect.poll(() => saves).toEqual([{ max_context_size: 65536 }]);
      await expect(panel.getByRole('status').filter({ hasText: language === 'zh-CN' ? '已保存' : 'Saved' })).toBeVisible();
      await entries.nth(0).getByRole('spinbutton').fill('8192');
      await entries.nth(1).getByRole('spinbutton').fill('32768');
      await entries.nth(0).getByRole('button').click();
      await expect.poll(() => updates).toEqual([{ context_size: 8192, instance_id: 'instance-1' }]);
      await expect(entries.nth(1).getByRole('spinbutton')).toHaveValue('32768');
      await expect(entries.nth(1).getByRole('button')).toBeEnabled();
    }
    expect(state.requests.filter(request => /^(POST|DELETE) /.test(request))).toEqual([]);
  });
}
