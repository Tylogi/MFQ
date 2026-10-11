import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['zh-CN', 'en'] as const) {
  test(`one selected model saves context, YaRN and streaming together (${language})`, async ({ page }, testInfo) => {
    await mockStudioServer(page, { language });
    const models = [{ id: 'qsa', model: 'Qwen3.8-Flash-Next-EWQ-MFQ-S4-L-MHC-NINT-LMHead8-MTP8',
      state: 'ready', context_size: 678920, context_capacity: 262144, qsa_kv_offload_supported: true },
    { id: 'dense', model: 'Qwen3.8-27B', state: 'ready', context_size: 32768, context_capacity: 32768, qsa_kv_offload_supported: false }];
    const policy = { max_context_size: null, model_overrides: { [models[0].model]: 678920 } as Record<string, number>,
      model_yarn_enabled: { [models[0].model]: true } as Record<string, boolean>,
      model_qsa_kv_offload: {} as Record<string, { enabled: boolean; budget_bytes: number }>, fallback_context_size: 32768 };
    await page.route('**/api/v1/runtime/instances', route => route.fulfill({ json: { data: models } }));
    await page.route('**/api/v1/runtime/context-policy', route => route.fulfill({ json: policy }));
    await page.route('**/api/v1/runtime/yarn/*', route => {
      const supported = route.request().url().endsWith('/qsa');
      return route.fulfill({ json: { supported, native_context: supported ? 262144 : 32768,
        maximum_context: supported ? 1048576 : 32768, maximum_factor: supported ? 4 : null,
        enabled: supported, effective_factor: supported ? 678920 / 262144 : 1 } });
    });
    await page.route('**/api/v1/models', route => route.fulfill({ json: { data: models.map(model => ({ id: `artifact-${model.id}`, name: model.model })) } }));
    await page.route('**/api/v1/models/*/cache-profile', route => route.fulfill({ json: {
      max_context: 262144, fixed_bytes: 29978, fixed_components: [{ group: 'QSA', name: 'indexer_tail', bytes: 29978 }], components: [
        { group: 'QSA', name: 'raw_kv', layers: 13, bytes_per_row: 26624, tokens_per_row: 1, allocation: 'power_of_two', minimum_rows: 16, max_read_rows_per_token: 2048 },
        { group: 'QSA', name: 'indexer_pooled', subgroup: 'indexer', bytes_per_row: 6656, tokens_per_row: 4,
          allocation: 'power_of_two', minimum_rows: 16, row_rounding: 'floor', active_after: 0 },
      ],
    } }));
    const writes: unknown[] = [];
    const writePaths: string[] = [];
    page.on('request', request => {
      if (['POST', 'PUT', 'DELETE'].includes(request.method())) writePaths.push(new URL(request.url()).pathname);
    });
    await page.route('**/api/v1/runtime/context', route => {
      const payload = route.request().postDataJSON();
      writes.push(payload);
      const model = models.find(item => item.id === payload.instance_id)!;
      model.context_size = payload.context_size ?? model.context_capacity;
      if (payload.context_size == null) delete policy.model_overrides[model.model];
      else policy.model_overrides[model.model] = payload.context_size;
      policy.model_yarn_enabled[model.model] = payload.yarn_enabled;
      policy.model_qsa_kv_offload[model.model] = payload.qsa_kv_offload;
      return route.fulfill({ json: { id: `ctx-${writes.length}`, kind: 'runtime.context.configure', status: 'succeeded', payload,
        progress: 1, result: { model: model.model, instance_id: model.id, max_context: model.context_size,
          context_override: payload.context_size, qsa_kv_offload: payload.qsa_kv_offload },
        cancel_requested: false, created_at: `2026-10-09T00:00:0${writes.length}Z`, updated_at: '2026-10-09T00:01:00Z' } });
    });
    await page.goto('/runtime');
    const panel = page.locator('.server-context-panel');
    const selected = panel.getByRole('combobox', { name: language === 'zh-CN' ? '单模型上下文模型' : 'Per-model context model' });
    const stream = panel.locator('.qsa-kv-settings:not(.kv-quantization-settings)');
    const toggle = stream.getByRole('checkbox');
    const disclosure = stream.getByRole('button');
    const budget = stream.getByRole('spinbutton');
    const target = stream.getByRole('status');
    const context = panel.getByRole('spinbutton', { name: `${models[0].model} ${language === 'zh-CN' ? '最大上下文' : 'maximum context'}` });
    const save = panel.getByRole('button', { name: language === 'zh-CN' ? '保存到该模型' : 'Save to this model', exact: true });
    const summary = panel.locator('.model-context-summary');
    await expect(save).toBeEnabled();
    await expect(panel.locator('.model-context-entry')).toHaveCount(1);
    await expect(panel.locator('.model-context-controls button')).toHaveCount(0);
    await expect(summary).not.toContainText('SSD');
    await expect(toggle).not.toBeChecked();
    await expect(disclosure).toHaveAttribute('aria-expanded', 'false');
    await expect(stream.locator('.qsa-kv-body')).toBeHidden();
    await disclosure.click();
    const estimate = stream.locator('.qsa-kv-estimate');
    await expect(estimate.locator('b')).toHaveText(['1.05 GiB', '906.58 MiB', '0 MiB', '52 MiB']);
    await budget.fill('1');
    await expect(estimate.locator('b')).toHaveText(['1 GiB', '0 MiB', '53.42 MiB', '52 MiB']);
    await toggle.check();
    await expect(summary).toContainText('SSD');
    await expect(summary).toContainText('53.42 MiB');
    await expect(summary).toContainText('52 MiB');
    await context.fill('524288');
    await expect(target).toContainText('524,288');
    await expect(summary).toContainText('524,288 tokens');
    await expect(summary).toContainText('2.00×');
    expect(writes).toEqual([]);
    await selected.selectOption(models[1].model);
    await expect(panel.locator('.model-context-entry')).toHaveCount(1);
    await expect(toggle).toBeDisabled();
    await expect(summary).not.toContainText('SSD');
    await selected.selectOption(models[0].model);
    await expect(context).toHaveValue('524288');
    await expect(toggle).toBeChecked();
    await disclosure.click();
    await expect(budget).toHaveValue('1');
    if (testInfo.project.name === 'desktop') {
      const summaryBox = (await summary.boundingBox())!;
      const saveBox = (await save.boundingBox())!;
      expect(saveBox.x).toBeGreaterThan(summaryBox.x + summaryBox.width);
      expect(Math.abs(saveBox.y + saveBox.height - summaryBox.y - summaryBox.height)).toBeLessThan(1);
      const row = (await estimate.boundingBox())!;
      expect(row.height).toBeLessThan(48);
    }
    await save.click();
    await expect.poll(() => writes).toEqual([{ instance_id: 'qsa', context_size: 524288, yarn_enabled: true,
      qsa_kv_offload: { enabled: true, budget_bytes: 2 ** 30 } }]);
    await expect(save).toBeEnabled();
    expect(writePaths).toEqual(['/api/v1/runtime/context']);
    await toggle.uncheck();
    await expect(summary).not.toContainText('SSD');
    await expect(summary).not.toContainText(language === 'zh-CN' ? '最大读取量/token' : 'Maximum reads/token');
    await save.click();
    await expect.poll(() => writes).toHaveLength(2);
    expect(writes[1]).toEqual({ instance_id: 'qsa', context_size: 524288, yarn_enabled: true,
      qsa_kv_offload: { enabled: false, budget_bytes: 2 ** 30 } });
    await expect(save).toBeEnabled();
    await toggle.check();
    await budget.fill('0');
    await expect(save).toBeDisabled();
    await budget.fill('2');
    await context.fill('678920');
    await expect(summary).toContainText('906.58 MiB');
    await expect(summary).toContainText('52 MiB');
    while (await page.locator('.studio-toast-close').count()) await page.locator('.studio-toast-close').first().click();
    await panel.locator('.model-context-list-heading').click();
    await panel.scrollIntoViewIfNeeded();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await panel.screenshot({ path: testInfo.outputPath('model-context-settings-light.png'), animations: 'disabled' });
    await panel.locator('.model-context-footer').screenshot({ path: testInfo.outputPath('model-context-footer-light.png'), animations: 'disabled' });
    await page.evaluate(() => document.documentElement.dataset.theme = 'dark');
    await panel.screenshot({ path: testInfo.outputPath('model-context-settings-dark.png'), animations: 'disabled' });
  });
}
