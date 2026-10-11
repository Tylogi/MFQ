import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const architecture of ['discrete', 'unified'] as const) {
  test(`physical ${architecture} CUDA memory controls and hierarchy`, async ({ page }, testInfo) => {
    await mockStudioServer(page, { language: 'zh-CN' });
    const separate = architecture === 'discrete';
    const GiB = 2 ** 30;
    await page.route('**/api/v1/runtime/status*', route => route.fulfill({ json: {
      runtime_state: 'ready', backend: 'cuda', memory_architecture: architecture, unified_memory: !separate,
      device_memory_total_bytes: separate ? 24 * GiB : null, host_memory_total_bytes: 128 * GiB,
      active_requests: 0, sampling_defaults: {}, prefix_cache_hot_bytes: GiB, prefix_cache_disk_bytes: GiB,
      prefix_cache_supported: 1, prefix_cache_disk_max_bytes: 100 * GiB, prefix_cache_max_bytes: 2 * GiB,
      prefix_cache_hot_blocks: 1, prefix_cache_disk_blocks: 1, prefix_cache_queries: 2, prefix_cache_hits: 1,
    } }));
    await page.route('**/api/v1/runtime/instances', route => route.fulfill({ json: { data: [{
      id: 'instance-1', model: 'Studio Test Model', state: 'ready', devices: ['cuda:0'], active_sessions: 1,
      queued_requests: 0, context_size: 32768, context_capacity: 262144, qsa_kv_offload_supported: true,
      memory: { resident_weight_bytes: 10 * GiB, kv_bytes: 2 * GiB, prefix_cache_bytes: GiB,
        prefix_cache_limit_bytes: 2 * GiB, prefix_cache_blocks: 1, context_count: 1,
        ram_experts: separate, ram_expert_bytes: 30 * GiB, ram_kv: separate, ram_kv_bytes: 2 * GiB,
        ram_kv_limit_bytes: 4 * GiB, ssd_experts: false, ssd_expert_bytes: 0,
        ssd_ple: true, ssd_ple_bytes: 2 * GiB, ssd_kv: true, ssd_kv_bytes: GiB },
    }] } }));
    await page.route('**/api/v1/runtime/memory-policy', route => route.fulfill({ json: {
      total_limit_bytes: null, model_limit_bytes: null, prefix_limit_bytes: null, effective_total_limit_bytes: null,
    } }));
    await page.goto('/');
    await expect(page.locator('.memory-tier')).toHaveCount(separate ? 7 : 5);
    await expect(page.getByText(separate ? '显存常驻专家与稠密权重' : '常驻专家与稠密权重', { exact: true })).toBeVisible();
    if (separate) {
      await expect(page.getByText('内存流式加载专家', { exact: true })).toBeVisible();
      await expect(page.getByText('内存流式 KV', { exact: true })).toBeVisible();
    }
    await page.goto('/resources');
    await expect(page.locator('.memory-tier')).toHaveCount(separate ? 7 : 5);
    await page.goto('/runtime');
    await expect(page.getByText(separate ? 'VRAM → SSD' : 'RAM → SSD', { exact: true })).toBeVisible();
    await expect(page.getByRole('combobox', { name: separate ? '前缀显存配额 模式' : '前缀 RAM 配额 模式' })).toBeEnabled();
    await page.getByRole('button', { name: '流式稀疏注意力', exact: true }).click();
    await expect(page.getByRole('spinbutton', { name: separate ? 'KV 显存常驻预算' : 'KV 常驻预算', exact: true })).toBeVisible();
    await expect(page.getByRole('spinbutton', { name: '内存流式 KV 预算', exact: true })).toHaveCount(separate ? 1 : 0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.locator('.qsa-kv-settings:not(.kv-quantization-settings)').screenshot({ path: testInfo.outputPath(`${architecture}-kv-settings.png`), animations: 'disabled' });
  });
}
