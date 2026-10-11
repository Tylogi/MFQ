import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['zh-CN', 'en'] as const) {
  test(`KV quantization uses the model settings bundle and preserves Indexer (${language})`, async ({ page }, testInfo) => {
    await mockStudioServer(page, { language });
    const model = { id: 'qsa', model: 'Qwen3.8-Flash-Next', state: 'ready', context_size: 678920,
      context_capacity: 262144, qsa_kv_offload_supported: true, kv_quantization_supported: true };
    await page.route('**/api/v1/runtime/instances', route => route.fulfill({ json: { data: [model] } }));
    await page.route('**/api/v1/runtime/context-policy', route => route.fulfill({ json: { max_context_size: null,
      model_overrides: { [model.model]: 678920 }, model_yarn_enabled: { [model.model]: true }, model_kv_quantization: {} } }));
    await page.route('**/api/v1/runtime/yarn/*', route => route.fulfill({ json: { supported: true,
      native_context: 262144, maximum_context: 1048576, maximum_factor: 4, enabled: true } }));
    await page.route('**/api/v1/models', route => route.fulfill({ json: { data: [{ id: 'artifact', name: model.model }] } }));
    await page.route('**/api/v1/models/*/cache-profile', route => route.fulfill({ json: { max_context: 262144,
      fixed_bytes: 29978, fixed_components: [{ group: 'QSA', name: 'indexer_tail', bytes: 29978 }], components: [
        { group: 'QSA', name: 'raw_kv', layers: 13, head_dimension: 256, kv_heads: 2, bytes_per_row: 26624,
          tokens_per_row: 1, allocation: 'power_of_two', minimum_rows: 16, max_read_rows_per_token: 2048 },
        { group: 'QSA', name: 'indexer_pooled', subgroup: 'indexer', bytes_per_row: 6656,
          tokens_per_row: 4, allocation: 'power_of_two', minimum_rows: 16, row_rounding: 'floor' },
      ] } }));
    const writes: unknown[] = [];
    await page.route('**/api/v1/runtime/context', route => {
      const payload = route.request().postDataJSON(); writes.push(payload);
      return route.fulfill({ json: { id: 'save', kind: 'runtime.context.configure', status: 'running', payload,
        progress: .1, cancel_requested: false, created_at: '2026-10-10T00:00:00Z', updated_at: '2026-10-10T00:00:00Z' } });
    });
    await page.goto('/runtime');
    const panel = page.locator('.kv-quantization-settings');
    const caret = panel.getByRole('button');
    const toggle = panel.getByRole('checkbox');
    await expect(toggle).toBeEnabled();
    await expect(toggle).not.toBeChecked();
    await expect(caret).toHaveAttribute('aria-expanded', 'false');
    await expect(panel.locator('.qsa-kv-body')).toBeHidden();
    await toggle.check();
    const level = panel.getByRole('combobox');
    await expect(level).toHaveValue('4');
    await expect(level.locator('option')).toHaveText(['2 bit', '2.5 bit', '3 bit', '3.5 bit', '4 bit', '6 bit', '8 bit']);
    await level.selectOption('2.5');
    await expect(panel).toContainText(language === 'zh-CN'
      ? '对于稀疏注意力而言，Indexer Cache 保持原精度；对于混合模型而言，线性注意力的递推状态不参与量化。'
      : 'For sparse attention, the Indexer Cache retains its original precision; for hybrid models, linear-attention recurrent states are not quantized.');
    await expect(page.locator('.model-context-kv')).toContainText('3.81 GiB');
    await expect(page.locator('.model-context-kv')).toContainText('1.05 GiB');
    await expect(page.locator('.model-context-summary')).toContainText('TurboQuant · K 2 / V 3 bit');
    expect(writes).toEqual([]);
    await page.screenshot({ path: testInfo.outputPath('kv-quantization.png'), fullPage: true });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.getByRole('button', { name: language === 'zh-CN' ? '保存到该模型' : 'Save to this model', exact: true }).click();
    await expect.poll(() => writes.length).toBe(1);
    expect(writes[0]).toEqual({ context_size: 678920, instance_id: 'qsa', yarn_enabled: true,
      qsa_kv_offload: { enabled: false, budget_bytes: 2 ** 31 },
      kv_quantization: { enabled: true, bits: 2.5, algorithm: 'turboquant' } });
  });
}
