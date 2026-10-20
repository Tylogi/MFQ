import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

test('load context defaults reach the model load request and remain readable in both themes', async ({ page }, testInfo) => {
  await mockStudioServer(page, { language: 'zh-CN' });
  const writes: Record<string, unknown>[] = [];
  await page.route(/\/api\/v1\/models(?:\?.*)?$/, route => route.fulfill({ json: { data: [{ id: 'local', name: 'Qwen3.5',
    architecture: 'qwen3_5', format: 'mfq', shard_count: 1, total_bytes: 40 * 2 ** 30,
    estimated_resident_weight_bytes: 8 * 2 ** 30, context_capacity: 131072, tensor_count: 1,
    record_count: 1, dtypes: [], complete: true, loadable: true, modified_at: '2026-10-14T00:00:00Z' }] } }));
  await page.route('**/api/v1/models/local/cache-profile?include_recurrent=true', route => route.fulfill({ json: {
    max_context: 131072, fixed_bytes: 2 ** 30,
    fixed_components: [{ group: 'GDN', name: 'recurrent_state', bytes: 2 ** 30 }],
    components: [{ group: 'GQA', name: 'raw_kv', layers: 2, kv_heads: 16, head_dimension: 256,
      bytes_per_row: 32768, tokens_per_row: 1, allocation: 'exact', minimum_rows: 16 }],
  } }));
  await page.route('**/api/v1/runtime/status*', route => route.fulfill({ json: {
    runtime_state: 'ready', model: 'Studio Test Model', model_type: 'Test', max_context: 8192,
    total_requests: 0, active_requests: 0, sampling_defaults: {}, memory_architecture: 'discrete',
    host_memory_available_bytes: 12 * 2 ** 30,
    gpu_devices: [{ id: '0', name: 'GPU', memory_available_bytes: 3 * 2 ** 30 }],
  } }));
  await page.route('**/api/v1/models/load', route => {
    writes.push(route.request().postDataJSON());
    return route.fulfill({ json: { operation_id: 'load', status: 'accepted' } });
  });
  await page.goto('/models');
  const panel = page.getByRole('region', { name: '上下文管理' });
  await expect(panel.getByRole('spinbutton')).toHaveAttribute('placeholder', '32768');
  await expect(panel.getByRole('combobox', { name: '估算模型' })).toHaveValue('');
  await expect(panel.getByText(/当前剩余/)).toContainText('15 GiB');
  await panel.getByRole('combobox', { name: '估算模型' }).selectOption('local');
  await expect(panel.getByLabel('预计总占用', { exact: true })).toHaveText('10 GiB');
  expect(writes).toHaveLength(0);
  await expect(panel.getByRole('button', { name: '高级' })).toHaveAttribute('aria-expanded', 'false');
  await panel.getByRole('checkbox', { name: '自动' }).uncheck();
  await panel.getByRole('checkbox', { name: 'RAM 预分配' }).uncheck();
  await expect(panel.getByRole('checkbox', { name: 'SSD KV' })).toBeChecked();
  await panel.getByRole('button', { name: '非稀疏注意力模型' }).click();
  await expect(panel.getByRole('checkbox', { name: 'SSD KV' })).toHaveCount(0);
  await expect(panel).not.toContainText('SSD KV');
  await panel.getByRole('checkbox', { name: 'VRAM 预分配' }).uncheck();
  await panel.getByRole('combobox', { name: '默认 KV 量化档位' }).selectOption('4');
  await panel.getByRole('spinbutton').fill('65536');
  await expect(panel.getByLabel('预计总占用', { exact: true })).toHaveText('9.5 GiB');
  for (const theme of ['light', 'dark']) {
    await page.evaluate(value => { document.documentElement.dataset.theme = value; }, theme);
    await panel.scrollIntoViewIfNeeded();
    await page.screenshot({ path: testInfo.outputPath(`context-${theme}.png`), fullPage: true });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  }
  await page.reload();
  await expect(panel.getByRole('spinbutton')).toHaveValue('65536');
  await expect(panel.getByRole('combobox', { name: '估算模型' })).toHaveValue('');
  await page.getByRole('button', { name: '加载', exact: true }).click();
  await expect.poll(() => writes.length).toBe(1);
  expect(writes[0]).toMatchObject({ context_size: 65536, context_management: {
    automatic: false, ssd_kv: true, ram_preallocated: false, vram_preallocated: false,
    kv_quantization: { enabled: true, bits: 4, algorithm: 'turboquant' },
  } });
  expect(writes[0].context_management).not.toHaveProperty('context_size');
});
