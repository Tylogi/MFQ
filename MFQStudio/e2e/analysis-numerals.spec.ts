import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';
import { analysisArtifacts, analysisFixture } from '../tests/fixtures/checkpointAnalysis';

for (const language of ['zh-CN', 'en'] as const) {
  test(`analysis reuses download numerals without changing text or layout (${language})`, async ({ page }, testInfo) => {
    const state = await mockStudioServer(page, { language });
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('/model-hub');
    const downloadValue = page.locator('.model-metadata-grid dd').first();
    await expect(downloadValue).toBeVisible();
    const downloadFamily = await downloadValue.evaluate(node => getComputedStyle(node).fontFamily);
    expect(downloadFamily).toContain('MFQ Small Numerals');
    await page.route(/\/api\/v1\/models(?:\?.*)?$/, route => route.fulfill({ json: { data: analysisArtifacts } }));
    await page.route('**/api/v1/models/*/analysis', route => route.fulfill({ json: analysisFixture() }));
    await page.goto('/analysis');
    const fact = page.locator('.analysis-facts dd').nth(1);
    await expect(fact).toBeVisible();
    await expect(fact).toHaveCSS('font-size', testInfo.project.name === 'mobile' ? '15px' : '17px');
    await expect(fact).toHaveCSS('font-weight', '560');
    await page.locator('.analysis-advanced > summary').click();
    await page.locator('.kv-cache-planner > summary').click();
    await page.locator('.analysis-tensor-details > summary').click();
    await page.locator('.layer-budget rect[role="button"]').first().focus();
    for (const selector of [
      '.analysis-facts dd', '.analysis-advanced-body dd', '.analysis-projection-table td',
      '.precision-chart-average', '.heatmap-y-axis', '.layer-budget svg text',
      '.chart-readout', '.kv-cache-planner input', '.analysis-tensor-details td', '.analysis-pagination span',
    ]) {
      const element = page.locator(selector).first();
      await expect(element).toBeVisible();
      expect(await element.evaluate(node => getComputedStyle(node).fontFamily)).toBe(downloadFamily);
    }
    await page.locator('.architecture-expand').click();
    await expect(page.locator('.architecture-dialog')).toBeVisible();
    expect(await page.locator('.architecture-dialog svg text').first().evaluate(node => getComputedStyle(node).fontFamily)).toBe(downloadFamily);
    await page.keyboard.press('Escape');
    const proof = await page.evaluate(async () => {
      const faces = await document.fonts.load('12px "MFQ Small Numerals"', '0123456789');
      const base = getComputedStyle(document.documentElement).fontFamily;
      const ctx = document.createElement('canvas').getContext('2d')!;
      ctx.font = `12px ${base}`;
      const originalText = ctx.measureText('ModelScope 中文').width;
      ctx.font = `12px "MFQ Small Numerals", ${base}`;
      return { count: faces.length, base, originalText, scopedText: ctx.measureText('ModelScope 中文').width,
        widths: [...'0123456789'].map(digit => ctx.measureText(digit).width),
        fits: document.documentElement.scrollWidth <= innerWidth };
    });
    expect(proof.count).toBeGreaterThan(0);
    expect(proof.originalText).toBe(proof.scopedText);
    expect(new Set(proof.widths).size).toBe(1);
    expect(proof.base).not.toContain('MFQ Small Numerals');
    expect(proof.fits).toBe(true);
    expect(errors).toEqual([]);
    expect(state.unexpected).toEqual([]);
  });
}
