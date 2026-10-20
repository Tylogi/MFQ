import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['zh-CN', 'en'] as const) {
  test(`per-GPU VRAM allocations and device selection (${language})`, async ({ page }, testInfo) => {
    await mockStudioServer(page, { language });
    const GiB = 2 ** 30;
    const gpu_devices = [0, 2, 3].map(id => ({ id: String(id), name: 'NVIDIA RTX test GPU',
      memory_total_bytes: 24 * GiB, memory_available_bytes: 8 * GiB, unified_memory: false }));
    await page.route('**/api/v1/runtime/status*', route => route.fulfill({ json: {
      runtime_state: 'ready', backend: 'cuda', memory_architecture: 'discrete', gpu_devices,
      host_memory_total_bytes: 64 * GiB, host_memory_available_bytes: 32 * GiB, sampling_defaults: {},
    } }));
    await page.route('**/api/v1/runtime/instances', route => route.fulfill({ json: { data: [{
      id: 'split', model: 'Qwen-Flash-Next', state: 'ready', devices: ['cuda:0', 'cuda:2'],
      active_sessions: 1, queued_requests: 0, started_at: '2026-10-13T00:00:00Z', memory: {
        resident_weight_bytes: 18 * GiB, resident_buffer_bytes: 2 * GiB, kv_bytes: 2 * GiB,
        prefix_cache_bytes: 0, ram_experts: true, ram_expert_bytes: 4 * GiB,
        ram_kv: false, ram_kv_bytes: 0, ssd_experts: false, ssd_ple: false, ssd_kv: false,
        devices: [0, 2].map((id, index) => ({ device_id: String(id),
          resident_weight_bytes: (index ? 10 : 8) * GiB, resident_buffer_bytes: GiB,
          kv_bytes: GiB, prefix_cache_bytes: 0 })),
      },
    }] } }));
    for (const route of ['/', '/resources']) {
      await page.goto(route);
      await expect(page.locator('[data-storage="vram"]')).toHaveCount(3);
      await expect(page.locator('[data-storage="ram"]')).toHaveCount(1);
      await expect(page.locator('[data-gpu-id="0"] h3')).toContainText('GPU 0');
      await expect(page.locator('[data-gpu-id="2"] h3')).toContainText('GPU 2');
      await expect(page.locator('[data-gpu-id="0"] [data-tier="weights"] .memory-tier-heading > span')).toHaveText('9 GiB / 18 GiB');
      await expect(page.locator('[data-gpu-id="2"] [data-tier="weights"] .memory-tier-heading > span')).toHaveText('11 GiB / 20 GiB');
      await expect(page.locator('[data-gpu-id="3"] [data-model-id]')).toHaveCount(0);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    }
    await page.screenshot({ path: testInfo.outputPath('multi-gpu-resources.png'), fullPage: true });
    await page.goto('/models');
    const gpu = page.getByRole('combobox', { name: language === 'zh-CN' ? '计算设备' : 'Compute device' });
    await gpu.selectOption('2');
    await expect(gpu).toHaveValue('2');
    await gpu.selectOption('0');
    await expect(gpu).toHaveValue('0');
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  });
}
