import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['zh-CN', 'en'] as const) {
  test(`per-model YaRN limits and save-time clamping (${language})`, async ({ page }, testInfo) => {
    await mockStudioServer(page, { language });
    const model = 'Qwen3.8-Flash-Next-EWQ-MFQ-S4-L-MHC-NINT-LMHead8-MTP8';
    await page.route('**/api/v1/models', route => route.fulfill({ json: { data: [{ id: 'artifact-qwen', name: model }] } }));
    await page.route('**/api/v1/models/*/cache-profile', route => route.fulfill({ json: {
      max_context: 262144, fixed_bytes: 29978, fixed_components: [{ group: 'QSA', name: 'indexer_tail', bytes: 29978 }], components: [
        { group: 'QSA', name: 'raw_kv', layers: 13, bytes_per_row: 26624, tokens_per_row: 1, allocation: 'power_of_two', minimum_rows: 16 },
        { group: 'QSA', name: 'indexer_pooled', subgroup: 'indexer', layers: 13, bytes_per_row: 6656, tokens_per_row: 4,
          allocation: 'power_of_two', minimum_rows: 16, row_rounding: 'floor', active_after: 0 },
      ],
    } }));
    await page.route('**/api/v1/runtime/instances', route => route.fulfill({ json: { data: [{
      id: 'qwen', model, state: 'ready', context_size: 262144, context_capacity: 262144,
    }] } }));
    await page.route('**/api/v1/runtime/context-policy', route => route.fulfill({ json: {
      max_context_size: null, model_overrides: {}, model_yarn_enabled: {}, fallback_context_size: 32768,
    } }));
    await page.route('**/api/v1/runtime/yarn/*', route => route.fulfill({ json: {
      supported: true, native_context: 262144, maximum_context: 1048576, maximum_factor: 4, enabled: false, effective_factor: 1,
    } }));
    const writes: unknown[] = [];
    await page.route('**/api/v1/runtime/context', route => {
      const payload = route.request().postDataJSON();
      writes.push(payload);
      return route.fulfill({ json: { id: `ctx-${writes.length}`, kind: 'runtime.context.configure', payload,
        status: 'succeeded', progress: 1, result: { max_context: payload.context_size, context_override: payload.context_size },
        cancel_requested: false, created_at: `2026-10-09T00:00:0${writes.length}Z`, updated_at: '2026-10-09T00:01:00Z' } });
    });
    await page.goto('/runtime');
    const panel = page.locator('.server-context-panel');
    const toggle = panel.getByRole('checkbox', { name: `${model} ${language === 'zh-CN' ? 'YaRN上下文上限扩展' : 'YaRN context limit extension'}` });
    const input = panel.getByRole('spinbutton', { name: `${model} ${language === 'zh-CN' ? '最大上下文' : 'maximum context'}` });
    const save = panel.getByRole('button', { name: language === 'zh-CN' ? '保存到该模型' : 'Save to this model', exact: true });
    await expect(save).toBeEnabled();
    await expect(toggle).not.toBeChecked();
    const heading = panel.locator('.model-context-list-heading');
    const row = panel.locator('.model-context-entry > .setting-row').first();
    await expect(heading).toHaveCSS('border-bottom-width', '1px');
    await expect(heading).toHaveCSS('border-bottom-style', 'solid');
    const headingBox = await heading.boundingBox();
    const descriptionBox = await heading.locator('p').boundingBox();
    const modelBox = await row.locator(':scope > div').first().boundingBox();
    expect(headingBox!.y + headingBox!.height - descriptionBox!.y - descriptionBox!.height).toBeGreaterThanOrEqual(18);
    expect(modelBox!.y - headingBox!.y - headingBox!.height).toBeGreaterThanOrEqual(18);
    const estimate = panel.locator('.model-context-kv');
    await expect(estimate).toContainText('KV 6.91 GiB');
    await input.fill('524288');
    await expect(estimate).toContainText('KV 13.81 GiB');
    await expect(estimate).toContainText(language === 'zh-CN' ? '原始KV 13.00 GiB · Indexer 0.81 GiB' : 'Raw KV 13.00 GiB · Indexer 0.81 GiB');
    expect(writes).toEqual([]);
    if (testInfo.project.name === 'desktop') {
      const box = await estimate.boundingBox();
      const field = await input.boundingBox();
      expect(box!.x + box!.width).toBeLessThan(field!.x);
      expect(Math.abs(box!.y + box!.height / 2 - field!.y - field!.height / 2)).toBeLessThan(4);
    }
    await input.fill('678920');
    await expect(estimate).toContainText('KV 17.89 GiB');
    await input.fill('524288');
    await save.click();
    await expect.poll(() => writes).toEqual([{ instance_id: 'qwen', context_size: 262144, yarn_enabled: false }]);
    await expect(input).toHaveValue('262144');
    await expect(panel.getByRole('alert')).toContainText(language === 'zh-CN' ? '不能设置超出最大值的值' : 'Cannot set a value above the native context limit');
    await toggle.check();
    await input.fill('2000000');
    await save.click();
    await expect.poll(() => writes).toHaveLength(2);
    expect(writes[1]).toEqual({ instance_id: 'qwen', context_size: 1048576, yarn_enabled: true });
    await expect(input).toHaveValue('1048576');
    await expect(estimate).toContainText('KV 27.63 GiB');
    await expect(toggle).toBeChecked();
    await expect(panel.getByRole('alert')).toContainText(language === 'zh-CN' ? '超出YaRN扩展上下文上限' : 'Exceeds the YaRN extended context limit');
    await expect(panel.getByRole('alert')).toContainText('1,048,576');
    await panel.scrollIntoViewIfNeeded();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await panel.screenshot({ path: testInfo.outputPath('yarn-context-light.png'), animations: 'disabled' });
  });
}
