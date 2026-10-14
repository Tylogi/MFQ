import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

test('load context defaults reach the model load request and remain readable in both themes', async ({ page }, testInfo) => {
  await mockStudioServer(page, { language: 'zh-CN' });
  const writes: Record<string, unknown>[] = [];
  await page.route('**/api/v1/models', route => route.fulfill({ json: { data: [{ id: 'local', name: 'Qwen3.5',
    architecture: 'qwen3_5', format: 'mfq', shard_count: 1, total_bytes: 1024, tensor_count: 1,
    record_count: 1, dtypes: [], complete: true, loadable: true, modified_at: '2026-10-14T00:00:00Z' }] } }));
  await page.route('**/api/v1/models/load', route => {
    writes.push(route.request().postDataJSON());
    return route.fulfill({ json: { operation_id: 'load', status: 'accepted' } });
  });
  await page.goto('/models');
  const panel = page.getByRole('region', { name: '上下文管理' });
  await expect(panel.getByRole('spinbutton')).toHaveAttribute('placeholder', '32768');
  await expect(panel.getByRole('button', { name: '高级' })).toHaveAttribute('aria-expanded', 'false');
  await panel.getByRole('checkbox', { name: '自动' }).uncheck();
  await panel.getByRole('checkbox', { name: 'RAM 预分配' }).uncheck();
  await expect(panel.getByRole('checkbox', { name: 'SSD KV' })).toBeChecked();
  await panel.getByRole('button', { name: '非稀疏注意力模型' }).click();
  await expect(panel.getByRole('checkbox', { name: 'SSD KV' })).toHaveCount(0);
  await expect(panel).not.toContainText('SSD KV');
  await panel.getByRole('checkbox', { name: 'VRAM 预分配' }).uncheck();
  await panel.getByRole('combobox').selectOption('4');
  await panel.getByRole('spinbutton').fill('65536');
  for (const theme of ['light', 'dark']) {
    await page.evaluate(value => { document.documentElement.dataset.theme = value; }, theme);
    await panel.scrollIntoViewIfNeeded();
    await page.screenshot({ path: testInfo.outputPath(`context-${theme}.png`), fullPage: true });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  }
  await page.reload();
  await expect(panel.getByRole('spinbutton')).toHaveValue('65536');
  await page.getByRole('button', { name: '加载', exact: true }).click();
  await expect.poll(() => writes.length).toBe(1);
  expect(writes[0]).toMatchObject({ context_size: 65536, context_management: {
    automatic: false, ssd_kv: true, ram_preallocated: false, vram_preallocated: false,
    kv_quantization: { enabled: true, bits: 4, algorithm: 'turboquant' },
  } });
  expect(writes[0].context_management).not.toHaveProperty('context_size');
});
