import { expect, test } from '@playwright/test';
import { mockStudioServer, officialCatalog } from './mockServer';

for (const moe of [true, false]) {
  test(`local and download ${moe ? 'MoE' : 'dense PLE'} memory pressure agree`, async ({ page }, testInfo) => {
    await mockStudioServer(page, { language: 'zh-CN' });
    const GiB = 2 ** 30;
    const roles = moe ? { dense: 2 * GiB, experts: 80 * GiB, embedding: GiB }
      : { dense: 27 * GiB, experts: 0, embedding: GiB };
    const weights = moe ? 83 * GiB : 28 * GiB;
    await page.route('**/api/v1/runtime/status*', route => route.fulfill({ json: {
      runtime_state: 'idle', backend: 'cuda', memory_architecture: 'discrete', unified_memory: false,
      device_memory_total_bytes: 24 * GiB, host_memory_total_bytes: 64 * GiB,
      active_requests: 0, sampling_defaults: {},
    } }));
    await page.route('**/api/v1/models', route => route.fulfill({ json: { data: [{
      id: 'a'.repeat(32), name: '压力测试模型', architecture: moe ? 'qwen4_exp' : 'qwen35', format: 'mfq',
      shard_count: 1, total_bytes: weights + 50 * GiB, estimated_resident_weight_bytes: weights,
      estimated_weight_bytes_by_role: roles, ssd_ple_bytes: 50 * GiB,
      tensor_count: 3, record_count: 3, dtypes: ['F16'], complete: true, loadable: true,
      modified_at: '2026-10-11T00:00:00Z',
    }] } }));
    const catalog = structuredClone(officialCatalog);
    catalog.system = { platform: 'Windows', machine: 'AMD64', backend: 'cuda', physical_memory_bytes: 64 * GiB,
      memory_pools: [{ kind: 'vram', capacity_bytes: 24 * GiB }, { kind: 'ram', capacity_bytes: 64 * GiB }] };
    catalog.data = [{ ...catalog.data[0], name: '压力测试模型', capabilities: moe ? ['MoE'] : [],
      variants: [{ ...catalog.data[0].variants[0], estimated_resident_weight_bytes: weights,
        estimated_weight_bytes_by_role: roles, ssd_ple_bytes: 50 * GiB }] }];
    await page.route('**/api/v1/hub/official*', route => route.fulfill({ json: catalog }));
    for (const path of ['/models', '/model-hub']) {
      await page.goto(path);
      await expect(page.getByRole('progressbar', { name: moe ? '显存压力: 8.3%' : '显存压力: 116.7%' })).toBeVisible();
      await expect(page.getByRole('progressbar', { name: 'RAM 压力: 126.6%' })).toHaveCount(moe ? 1 : 0);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.getByRole('progressbar', { name: moe ? '显存压力: 8.3%' : '显存压力: 116.7%' }).scrollIntoViewIfNeeded();
      await page.locator(path === '/models' ? '.model-library-panel .model-row' : '.model-variant-list').screenshot({ path: testInfo.outputPath(`${moe ? 'moe' : 'dense'}-${path.slice(1)}.png`), animations: 'disabled' });
    }
  });
}
