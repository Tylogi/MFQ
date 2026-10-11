import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['zh-CN', 'en'] as const) {
  test(`prefix RAM budget keeps compact controls and a separate action row (${language})`, async ({ page }, testInfo) => {
    await mockStudioServer(page, { language });
    const updates: unknown[] = [];
    await page.route('**/api/v1/runtime/memory-policy', route => {
      if (route.request().method() === 'PUT') {
        updates.push(route.request().postDataJSON());
        return route.fulfill({ json: { operation_id: 'memory-layout-job' } });
      }
      return route.fulfill({ json: { total_limit_bytes: null, effective_total_limit_bytes: 107.5 * 2 ** 30,
        model_limit_bytes: null, prefix_limit_bytes: null } });
    });
    await page.route('**/api/v1/jobs/memory-layout-job', route => route.fulfill({ json: {
      id: 'memory-layout-job', kind: 'runtime.memory.configure', status: 'succeeded', payload: {},
      progress: 1, created_at: '2026-10-09T00:00:00Z', updated_at: '2026-10-09T00:00:00Z', cancel_requested: false,
    } }));
    await page.goto('/runtime');
    const panel = page.locator('.server-memory-panel');
    const name = language === 'zh-CN' ? '前缀 RAM 配额' : 'Prefix RAM allowance';
    const mode = panel.getByRole('combobox', { name: `${name} ${language === 'zh-CN' ? '模式' : 'mode'}`, exact: true });
    await expect(mode).toBeEnabled();
    const row = mode.locator('xpath=ancestor::div[contains(concat(" ", normalize-space(@class), " "), " setting-row ")][1]');
    const automatic = (await panel.boundingBox())!;
    const actions = panel.locator('.memory-budget-actions');
    expect(await actions.evaluate(element => getComputedStyle(element).borderTopWidth)).toBe('1px');
    expect(await row.evaluate(element => getComputedStyle(element).minHeight)).toBe('56px');
    expect(updates).toEqual([]);
    await panel.scrollIntoViewIfNeeded();
    await panel.screenshot({ path: testInfo.outputPath('memory-budget-automatic.png'), animations: 'disabled' });
    await mode.selectOption('manual');
    const input = panel.getByRole('spinbutton', { name: `${name} ${language === 'zh-CN' ? '预算上限' : 'budget limit'}`, exact: true });
    await expect(input).toHaveValue('4');
    const frame = row.locator('.memory-budget-input');
    const box = (await frame.boundingBox())!;
    expect(box.height).toBe(34);
    expect(box.width).toBeLessThanOrEqual(120);
    expect(await input.evaluate(element => getComputedStyle(element).borderTopWidth)).toBe('0px');
    expect((await mode.boundingBox())!.height).toBe(34);
    const manual = (await panel.boundingBox())!;
    expect(manual.height).toBeLessThanOrEqual(automatic.height + 1);
    const button = panel.getByRole('button', { name: language === 'zh-CN' ? '应用预算' : 'Apply budgets', exact: true });
    await input.fill('');
    await expect(button).toBeDisabled();
    await input.fill('0');
    await expect(button).toBeEnabled();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await panel.screenshot({ path: testInfo.outputPath('memory-budget-manual.png'), animations: 'disabled' });
    await page.evaluate(() => document.documentElement.dataset.theme = 'dark');
    await panel.screenshot({ path: testInfo.outputPath('memory-budget-manual-dark.png'), animations: 'disabled' });
    await button.click();
    await expect.poll(() => updates).toEqual([{ total_limit_bytes: null, model_limit_bytes: null, prefix_limit_bytes: 0 }]);
    await mode.selectOption('automatic');
    await expect(input).toBeHidden();
    await button.click();
    await expect.poll(() => updates).toEqual([
      { total_limit_bytes: null, model_limit_bytes: null, prefix_limit_bytes: 0 },
      { total_limit_bytes: null, model_limit_bytes: null, prefix_limit_bytes: null },
    ]);
  });
}
