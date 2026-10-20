import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['zh-CN', 'en'] as const) {
  test(`main text uses the Harness system stack without replacing numeric fonts (${language})`, async ({ page }) => {
    await mockStudioServer(page, { language });
    await page.goto('/');
    const heading = page.locator('main h1');
    await expect(heading).toBeVisible();
    const family = await page.locator('html').evaluate(element => getComputedStyle(element).fontFamily);
    expect(family).toMatch(/^-apple-system, (BlinkMacSystemFont|"system-ui"),/);
    expect(family).toContain('PingFang SC');
    expect(family).toContain('Microsoft YaHei');
    expect(family).not.toContain('SF Pro Text');
    expect(family).not.toContain('MFQ Small Numerals');
    await expect(heading).toHaveCSS('font-family', family);
    await expect(page.locator('.sectioned-nav button').first()).toHaveCSS('font-family', family);
    await page.goto('/model-hub');
    await expect(page.locator('.model-detail-heading h3')).toBeVisible();
    await expect(page.locator('.model-detail-heading h3')).toHaveCSS('font-family', family);
    const numeric = page.locator('.download-concurrency select');
    expect(await numeric.evaluate(element => getComputedStyle(element).fontFamily)).toBe(`"MFQ Small Numerals", ${family}`);
    const pressure = page.locator('.variant-memory-pressure > strong').first();
    await expect(pressure).toBeVisible();
    expect(await pressure.evaluate(element => getComputedStyle(element).fontFamily)).toContain('ui-monospace');
  });
}
