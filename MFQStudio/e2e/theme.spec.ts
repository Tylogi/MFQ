import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';
import { analysisArtifacts, analysisFixture } from '../tests/fixtures/checkpointAnalysis';

function contrast(foreground: string, background: string) {
  const luminance = (color: string) => {
    const channels = color.match(/[\d.]+/g)!.slice(0, 3).map(Number).map(value => {
      const channel = value / 255;
      return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
    });
    return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
  };
  const a = luminance(foreground), b = luminance(background);
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

for (const mode of ['light', 'dark', 'system-light', 'system-dark'] as const) {
  test(`copper accent covers controls and charts (${mode})`, async ({ page }, testInfo) => {
    const dark = mode.endsWith('dark');
    const theme = mode.startsWith('system') ? 'system' : mode;
    const expected = dark ? 'rgb(214, 170, 115)' : 'rgb(159, 98, 50)';
    const expectedBar = dark ? 'rgb(153, 157, 158)' : 'rgb(95, 98, 100)';
    const state = await mockStudioServer(page);
    await page.emulateMedia({ colorScheme: dark ? 'dark' : 'light' });
    await page.addInitScript(value => localStorage.setItem('mfq.studio.generation.v1',
      JSON.stringify({ language: 'en', theme: value })), theme);
    await page.route(/\/api\/v1\/models(?:\?.*)?$/, route => route.fulfill({ json: { data: analysisArtifacts } }));
    await page.route('**/api/v1/models/*/analysis', route => route.fulfill({ json: analysisFixture() }));
    await page.route('**/api/v1/runtime/instances', route => route.fulfill({ json: { data: [0, 1, 2, 3].map(index => ({
      id: `instance-${index + 1}`, model: index ? `Model ${index + 1}` : 'Studio Test Model', state: 'ready',
      devices: ['metal'], active_sessions: 0, queued_requests: 0, started_at: `2026-10-10T00:00:0${index}Z`,
      memory: { resident_weight_bytes: 2 ** 30, kv_bytes: 2 ** 28, context_count: 1, prefix_cache_blocks: 0,
        ssd_experts: true, ssd_expert_bytes: 2 ** 30, ssd_ple: true, ssd_ple_bytes: 2 ** 30,
        ssd_kv: true, ssd_kv_bytes: 2 ** 28 },
    })) } }));
    await page.route('**/api/v1/runtime/status*', route => route.fulfill({ json: {
      runtime_state: 'ready', model: 'Studio Test Model', model_type: 'Test', max_context: 8192,
      runtime_memory_effective_budget_bytes: 64 * 2 ** 30, active_requests: 0, sampling_defaults: {},
    } }));
    await page.route('**/api/v1/runtime/resources', route => route.fulfill({ json: {
      sampled_at: 1, interval_seconds: 2, cpu_name: 'Apple M5 Max', cpu_cores: 18, cpu_utilization_percent: 36,
      gpus: [{ name: 'Apple M5 Max', core_count: 40, utilization_percent: 24 }], disks: [], weights: [],
    } }));
    const download = { id: 'theme-download', kind: 'download.modelscope', status: 'running', progress: 0.5,
      payload: { repo_id: 'example/theme-model', destination: '/models' }, cancel_requested: false,
      created_at: '2026-10-10T00:00:00Z', updated_at: '2026-10-10T00:00:00Z' };
    await page.route(/\/api\/v1\/jobs(?:\?.*)?$/, route => route.fulfill({ json: { data: [download] } }));
    await page.route('**/api/v1/jobs/theme-download', route => route.fulfill({ json: download }));
    await page.route('**/api/v1/jobs/theme-download/events/stream*', route => route.fulfill({ status: 503, json: { error: 'offline' } }));
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));

    await page.goto('/settings');
    await expect(page.getByRole('combobox', { name: 'Theme', exact: true })).toBeVisible();
    await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
    const primary = page.locator('main button.primary').first();
    await expect(primary).toBeVisible();
    const buttonColors = await primary.evaluate(element => {
      const style = getComputedStyle(element);
      return { background: style.backgroundColor, foreground: style.color };
    });
    expect(buttonColors.background).toBe(expected);
    expect(contrast(buttonColors.foreground, buttonColors.background)).toBeGreaterThanOrEqual(4.5);
    await primary.focus();
    await expect(primary).toHaveCSS('outline-color', expected);
    const selection = await primary.evaluate(element => {
      const style = getComputedStyle(element, '::selection');
      return { background: style.backgroundColor, foreground: style.color };
    });
    expect(selection.background).toBe(expected);
    expect(contrast(selection.foreground, selection.background)).toBeGreaterThanOrEqual(4.5);
    const vision = page.getByRole('switch', { name: 'Vision input', exact: true });
    await expect(vision).toBeChecked();
    await expect(vision).toHaveCSS('background-color', expected);
    const thumb = vision.locator('.studio-switch-thumb');
    const thumbColor = await thumb.evaluate(element => getComputedStyle(element).backgroundColor);
    expect(contrast(thumbColor, expected)).toBeGreaterThanOrEqual(3);
    await vision.click();
    await expect(vision).not.toBeChecked();
    await expect(thumb).toHaveCSS('background-color', 'rgb(255, 255, 255)');

    await page.goto('/');
    await expect(page.locator('main h1')).toBeVisible();
    const modelColors = Array(4).fill(expectedBar);
    await expect(page.locator('.memory-model-legend i')).toHaveCount(4);
    await expect(page.locator('.memory-tier-track')).toHaveCount(5);
    expect(await page.locator('.memory-model-legend i').evaluateAll(elements =>
      elements.map(element => getComputedStyle(element).backgroundColor))).toEqual(modelColors);
    for (const tier of await page.locator('.memory-tier-track').all()) {
      expect(await tier.locator(':scope > span').evaluateAll(elements =>
        elements.map(element => getComputedStyle(element).backgroundColor))).toEqual(modelColors);
    }
    const chatColors = await page.locator('.runtime-hero-actions .runtime-hero-chat').evaluate(element => {
      const style = getComputedStyle(element);
      return { foreground: style.color, background: style.backgroundColor };
    });
    const channels = chatColors.background.match(/[\d.]+/g)!.map(Number);
    expect(Math.max(...channels) - Math.min(...channels)).toBeLessThanOrEqual(4);
    expect(contrast(chatColors.foreground, chatColors.background)).toBeGreaterThanOrEqual(4.5);
    for (const monogram of await page.locator('.runtime-hero .model-monogram.ready, .overview-model-card .model-monogram.ready').all()) {
      const color = await monogram.evaluate(element => getComputedStyle(element).color);
      const rgb = color.match(/[\d.]+/g)!.map(Number);
      expect(Math.max(...rgb) - Math.min(...rgb)).toBeLessThanOrEqual(4);
      expect(color).toBe(dark ? 'rgb(206, 208, 208)' : 'rgb(69, 71, 72)');
    }
    await expect(page.locator('.runtime-hero .runtime-status-pill')).toHaveCSS('color',
      dark ? 'rgb(87, 209, 132)' : 'rgb(20, 122, 67)');
    await page.screenshot({ path: testInfo.outputPath(`overview-${mode}.png`), fullPage: true, animations: 'disabled' });

    await page.goto('/resources');
    await expect(page.locator('.resource-allocation-panel')).toBeVisible();
    await expect(page.locator('.memory-model-legend i')).toHaveCount(4);
    expect(await page.locator('.memory-model-legend i').evaluateAll(elements =>
      elements.map(element => getComputedStyle(element).backgroundColor))).toEqual(modelColors);
    await expect(page.locator('.resource-utilization-track i').first()).toHaveCSS('background-color', expectedBar);

    await page.goto('/model-hub');
    await expect(page.locator('.download-ring-progress')).toHaveCSS('stroke', expectedBar);
    await page.getByRole('tab', { name: 'Download queue', exact: true }).click();
    const progress = page.getByRole('progressbar', { name: 'Download progress', exact: true });
    await expect(progress).toBeVisible();
    await expect(progress).toHaveAttribute('value', '0.5');
    await expect(progress).toHaveCSS('accent-color', expectedBar);
    await page.screenshot({ path: testInfo.outputPath(`download-bars-${mode}.png`), fullPage: true, animations: 'disabled' });

    await page.goto('/analysis');
    const canvas = page.locator('.expert-heatmap canvas').first();
    await expect(canvas).toBeVisible();
    expect(await canvas.evaluate((element: HTMLCanvasElement) => {
      const pixels = element.getContext('2d')!.getImageData(0, 0, element.width, element.height).data;
      for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i] < pixels[i + 1] || pixels[i + 1] < pixels[i + 2]) return false;
      }
      return true;
    })).toBe(true);
    await expect(page.locator('.attention-qsa').first()).toHaveCSS('background-color', expected);
    await expect(page.locator('.budget-series-moe .budget-line')).toHaveCSS('stroke', expected);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath(`analysis-${mode}.png`), fullPage: true, animations: 'disabled' });
    expect(errors).toEqual([]);
    expect(state.unexpected).toEqual([]);
  });
}
