import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const theme of ['light', 'dark'] as const) {
  test(`only memory pressure bars and numbers are neutral (${theme})`, async ({ page }, testInfo) => {
    const state = await mockStudioServer(page);
    await page.addInitScript(value => {
      localStorage.setItem('mfq.studio.generation.v1', JSON.stringify({ language: 'en', theme: value }));
    }, theme);
    await page.route('**/api/v1/models', route => route.fulfill({ json: { data: [25, 75, 150].map((size, index) => ({
      id: `pressure-${index}`, name: index === 0 ? 'Studio Test Model' : `Qwen pressure ${index}`,
      architecture: 'qwen3', format: 'mfq', shard_count: 2, total_bytes: size * 2 ** 30,
      estimated_resident_weight_bytes: size * 2 ** 30, tensor_count: 20, record_count: 20,
      dtypes: ['NINTv2'], complete: true, loadable: true, modified_at: '2026-10-10T00:00:00Z',
    })) } }));
    await page.route('**/api/v1/runtime/status*', route => route.fulfill({ json: {
      runtime_state: 'ready', model: 'Studio Test Model', model_type: 'Test', max_context: 8192,
      runtime_memory_headroom_bytes: 100 * 2 ** 30, total_requests: 0, active_requests: 0, sampling_defaults: {},
    } }));
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('/models');
    await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
    await expect(page.locator('.variant-memory-pressure > strong')).toHaveText(['25.0%', '75.0%', '150.0%']);
    for (const path of ['/models', '/model-hub']) {
      if (path !== '/models') await page.goto(path);
      await expect(page.locator('.variant-memory-pressure')).toHaveCount(3);
      const pressure = await page.locator('.variant-memory-pressure').evaluateAll(elements => elements.map(element => {
        const fill = getComputedStyle(element.querySelector('.variant-memory-track > span')!);
        return { bar: fill.backgroundColor, image: fill.backgroundImage, text: getComputedStyle(element.querySelector('strong')!).color };
      }));
      expect(new Set(pressure.map(item => item.bar)).size).toBe(1);
      for (const item of pressure) {
        expect(item.image).toBe('none');
        for (const color of [item.bar, item.text]) {
          const rgb = color.match(/[\d.]+/g)!.slice(0, 3).map(Number);
          expect(Math.max(...rgb) - Math.min(...rgb)).toBeLessThanOrEqual(6);
        }
      }
      await page.screenshot({ path: testInfo.outputPath(`${path.slice(1)}-${theme}.png`), animations: 'disabled', fullPage: true });
    }
    expect(await page.locator('.model-download-page').evaluate(element => getComputedStyle(element).getPropertyValue('--accent').trim()))
      .toBe(theme === 'light' ? '#9f6232' : '#d6aa73');
    expect(errors).toEqual([]);
    expect(state.unexpected).toEqual([]);
  });
}
