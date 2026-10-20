import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['zh-CN', 'en'] as const) {
  test(`service and download small numerals preserve text fonts and other pages (${language})`, async ({ page }, testInfo) => {
    const state = await mockStudioServer(page, { language });
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/api/v1/runtime/memory-policy', route => route.fulfill({ json: {
      total_limit_bytes: null, effective_total_limit_bytes: 107.5 * 2 ** 30,
      model_limit_bytes: null, prefix_limit_bytes: null,
    } }));
    await page.goto('/runtime');
    const available = page.locator('.memory-budget-available');
    await expect(available).toHaveText('107.5 GiB');
    await expect(available).toHaveCSS('font-size', '11.5px');
    expect(await available.evaluate(element => getComputedStyle(element).fontFamily)).toContain('MFQ Small Numerals');
    const label = language === 'zh-CN' ? '前缀 RAM 配额' : 'Prefix RAM allowance';
    await page.getByRole('combobox', { name: `${label} ${language === 'zh-CN' ? '模式' : 'mode'}`, exact: true }).selectOption('manual');
    const input = page.getByRole('spinbutton', { name: `${label} ${language === 'zh-CN' ? '预算上限' : 'budget limit'}`, exact: true });
    await expect(input).toHaveValue('4');
    await expect(input).toHaveCSS('font-size', '12px');
    expect(await input.evaluate(element => getComputedStyle(element).fontFamily)).toContain('MFQ Small Numerals');
    for (const element of await page.locator('.service-view strong, .service-view b').all()) {
      const style = await element.evaluate(node => ({ family: getComputedStyle(node).fontFamily, weight: getComputedStyle(node).fontWeight }));
      if (Number(style.weight) > 500) expect(style.family).not.toContain('MFQ Small Numerals');
    }
    await page.locator('.server-memory-panel').scrollIntoViewIfNeeded();
    await page.locator('.server-memory-panel').screenshot({ path: testInfo.outputPath('service-small-numerals.png') });
    await page.goto('/model-hub');
    for (const selector of ['.model-metadata-grid dd', '.model-variant small', '.detected-memory-pool small', '.download-concurrency select']) {
      await expect(page.locator(selector).first()).toBeVisible();
      expect(await page.locator(selector).first().evaluate(element => getComputedStyle(element).fontFamily)).toContain('MFQ Small Numerals');
    }
    for (const selector of ['.detected-hardware-summary strong', '.detected-memory-pool strong', '.configuration-details strong']) {
      await expect(page.locator(selector).first()).toBeVisible();
      expect(await page.locator(selector).first().evaluate(element => getComputedStyle(element).fontFamily)).not.toContain('MFQ Small Numerals');
    }
    const proof = await page.evaluate(async () => {
      const faces = await document.fonts.load('12px "MFQ Small Numerals"', '0123456789');
      const base = getComputedStyle(document.documentElement).fontFamily;
      const ctx = document.createElement('canvas').getContext('2d')!;
      ctx.font = `12px ${base}`;
      const originalText = ctx.measureText('ModelScope 中文').width;
      ctx.font = `12px "MFQ Small Numerals", ${base}`;
      const scopedText = ctx.measureText('ModelScope 中文').width;
      const widths = [...'0123456789'].map(text => ctx.measureText(text).width);
      return { count: faces.length, originalText, scopedText, widths, base,
        heading: getComputedStyle(document.querySelector('.model-detail-heading h3')!).fontFamily };
    });
    expect(proof.count).toBeGreaterThan(0);
    expect(proof.originalText).toBe(proof.scopedText);
    expect(new Set(proof.widths).size).toBe(1);
    expect(proof.base).not.toContain('MFQ Small Numerals');
    expect(proof.heading).not.toContain('MFQ Small Numerals');
    await page.screenshot({ path: testInfo.outputPath('download-small-numerals.png'), fullPage: true });
    await page.goto('/');
    await expect(page.locator('.metric-tile > strong').first()).toBeVisible();
    expect(await page.locator('.metric-tile > strong').first().evaluate(element => getComputedStyle(element).fontFamily)).not.toContain('MFQ Small Numerals');
    expect(errors).toEqual([]);
    expect(state.unexpected).toEqual([]);
  });
}
