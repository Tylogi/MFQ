import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['zh-CN', 'en'] as const) {
  test(`streamed KV has shared colors, live tracking and no settings context duplicate (${language})`, async ({ page }, testInfo) => {
    const state = await mockStudioServer(page, { language });
    const zh = language === 'zh-CN';
    const GiB = 2 ** 30;
    let size = GiB;
    const models = ['Qwen3.8-Flash-Next-EWQ-MFQ-S4-L-MHC-NINT-LMHead8-MTP8', 'Qwen3.8-Flash-Next-Second-Model'];
    await page.route('**/api/v1/runtime/status*', route => route.fulfill({ json: {
      runtime_state: 'ready', instance_id: 'a', model: models[0], active_requests: 0,
      runtime_memory_effective_budget_bytes: 96 * GiB, sampling_defaults: {},
    } }));
    await page.route('**/api/v1/runtime/instances', route => route.fulfill({ json: { data: models.map((model, index) => ({
      id: index ? 'b' : 'a', model, state: 'ready', devices: ['metal'], active_sessions: 1, queued_requests: 0,
      started_at: `2026-10-09T00:00:0${index}Z`, memory: {
        resident_weight_bytes: 20 * GiB, kv_bytes: GiB, context_count: 1, prefix_cache_blocks: 0,
        prefix_cache_bytes: 0, prefix_cache_limit_bytes: GiB, ssd_experts: false, ssd_expert_bytes: 0,
        ssd_ple: true, ssd_ple_bytes: 2 * GiB, ssd_kv: true, ssd_kv_bytes: index ? 3 * GiB : size,
        streaming_kv_resident_bytes: GiB, streaming_kv_budget_bytes: 2 * GiB, streaming_kv_pending_bytes: 0,
        ssd_kv_read_bytes: index ? 3 * GiB : GiB, ssd_kv_written_bytes: 4 * GiB,
        ssd_kv_reads: 1, ssd_kv_hits: 3,
      },
    })) } }));
    let samples = 0;
    await page.route('**/api/v1/runtime/resources', route => {
      samples++;
      return route.fulfill({ json: { sampled_at: 1, interval_seconds: 2, cpu_utilization_percent: 10,
        cpu_name: 'Apple M5 Max', cpu_cores: 18, gpus: [{ name: 'Apple M5 Max', core_count: 40, utilization_percent: 12 }],
        memory_total_bytes: 128 * GiB, memory_used_bytes: 80 * GiB, memory_available_bytes: 48 * GiB,
        disks: [], weights: models.map((model, index) => ({ instance_id: index ? 'b' : 'a', model,
          expert_read_bytes_per_second: 0, ple_read_bytes_per_second: 2 ** 20, engram_read_bytes_per_second: null,
          kv_read_bytes_per_second: samples === 1 ? 2 ** 20 : 3 * 2 ** 20,
          kv_write_bytes_per_second: 2 * 2 ** 20 })) } });
    });
    await page.goto('/');
    const overview = page.locator('[data-tier="ssd-kv"]');
    await expect(overview).toContainText(zh ? 'SSD 流式 KV' : 'SSD-streamed KV');
    await expect(overview).toContainText('4 GiB');
    await expect(overview).toContainText(zh ? 'RAM 命中率 75%' : 'RAM hit rate 75%');
    const colors = await page.locator('[data-tier="weights"] .memory-tier-track > span').evaluateAll(nodes => nodes.map(node => (node as HTMLElement).style.backgroundColor));
    expect(await overview.locator('.memory-tier-track > span').evaluateAll(nodes => nodes.map(node => (node as HTMLElement).style.backgroundColor))).toEqual(colors);
    await expect(overview.locator('.memory-tier-track > span').first()).toHaveAttribute('style', /width: 25%/);
    await expect(page.locator('[data-tier="kv"]')).toContainText('2 GiB / 56 GiB');
    await overview.scrollIntoViewIfNeeded();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath('overview-streamed-kv.png'), animations: 'disabled', fullPage: true });
    await page.goto('/resources');
    const detail = page.locator('[data-tier="ssd-kv"]');
    await expect(detail.locator('.resource-tier-model')).toHaveCount(2);
    await expect(detail).toContainText(zh ? '常驻 1 GiB / 2 GiB' : 'Resident 1 GiB / 2 GiB');
    await expect(detail).toContainText(zh ? '待写入 0 B' : 'Pending write 0 B');
    const traffic = page.locator('.resource-weight-row').first();
    await expect(traffic).toContainText(zh ? '流式 KV 读取1 MiB/s' : 'Streamed KV read1 MiB/s');
    await expect(traffic).toContainText(zh ? '流式 KV 写入2 MiB/s' : 'Streamed KV write2 MiB/s');
    await expect(traffic).toContainText(zh ? '流式 KV 读取3 MiB/s' : 'Streamed KV read3 MiB/s');
    size = 2 * GiB;
    await page.evaluate(() => document.dispatchEvent(new Event('visibilitychange')));
    await expect(detail.locator('.memory-tier-heading > span')).toHaveText('5 GiB');
    await expect(detail.locator('.memory-tier-track > span').first()).toHaveAttribute('style', /width: 40%/);
    await detail.scrollIntoViewIfNeeded();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath('resources-streamed-kv.png'), animations: 'disabled', fullPage: true });
    await traffic.scrollIntoViewIfNeeded();
    await page.screenshot({ path: testInfo.outputPath('resources-kv-traffic.png'), animations: 'disabled' });
    await page.goto('/settings');
    await expect(page.locator('.model-context-settings')).toHaveCount(0);
    await expect(page.getByText(zh ? '上下文' : 'Context', { exact: true })).toHaveCount(0);
    await expect(page.getByText(zh ? '外观' : 'Appearance', { exact: true })).toBeVisible();
    expect(state.requests.filter(request => /^(POST|PUT|DELETE) /.test(request))).toEqual([]);
  });
}
