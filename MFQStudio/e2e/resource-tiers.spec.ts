import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['zh-CN', 'en'] as const) {
  test(`shared RAM/SSD categories, tier details and pressure typography (${language})`, async ({ page }, testInfo) => {
    const state = await mockStudioServer(page, { language });
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    for (const path of ['/', '/resources']) {
      await page.goto(path);
      await expect(page.locator('[data-memory-group]')).toHaveCount(2);
      await expect(page.locator('[data-memory-group="RAM"] [data-tier]')).toHaveCount(2);
      await expect(page.locator('[data-memory-group="SSD"] [data-tier]')).toHaveCount(3);
      await expect(page.locator('.memory-capacity-note')).toContainText(language === 'zh-CN' ? '扣除系统和其他进程占用' : 'after system and other process usage');
      await expect(page.locator('[data-memory-group="RAM"] h3')).toHaveText(language === 'zh-CN' ? '内存' : 'RAM');
      for (const group of ['RAM', 'SSD']) {
        const heading = page.locator(`[data-memory-group="${group}"] .memory-tier-group-heading`);
        await expect(heading).toHaveCSS('border-bottom-width', '0px');
        const title = (await heading.locator('h3').boundingBox())!;
        const rule = (await heading.locator('i').boundingBox())!;
        expect(rule.x).toBeGreaterThan(title.x + title.width);
        expect(rule.width).toBeGreaterThan(12);
        expect(rule.height).toBe(1);
        expect(Math.abs(rule.y + rule.height / 2 - title.y - title.height / 2)).toBeLessThanOrEqual(1);
      }
      const trigger = page.locator('.memory-heading-title').getByRole('button', { name: language === 'zh-CN' ? '资源分层说明' : 'Memory tier details' });
      const help = page.locator('.studio-tooltip');
      await expect(help).not.toBeVisible();
      await trigger.scrollIntoViewIfNeeded();
      await trigger.hover();
      await expect(help).toBeVisible();
      await expect(help).toContainText(language === 'zh-CN' ? '前缀缓存' : 'prefix caches');
      const bounds = await help.boundingBox();
      expect(bounds!.x).toBeGreaterThanOrEqual(0);
      expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(page.viewportSize()!.width);
      await page.screenshot({ path: testInfo.outputPath(`${path.slice(1) || 'overview'}-help-${language}.png`), fullPage: true });
      await page.keyboard.press('Escape');
      await expect(help).not.toBeVisible();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    }
    const reference = await page.locator('.memory-tier-heading > span').first().evaluate(element => {
      const style = getComputedStyle(element);
      return { family: style.fontFamily, weight: style.fontWeight };
    });
    await page.goto('/model-hub');
    const percentages = page.locator('.model-variant-list .variant-memory-pressure > strong');
    await expect(percentages).toHaveCount(3);
    expect(await percentages.first().evaluate(element => {
      const style = getComputedStyle(element);
      return { family: style.fontFamily, weight: style.fontWeight };
    })).toEqual(reference);
    await expect(page.locator('.model-variant-list')).not.toContainText(language === 'zh-CN' ? '预计权重常驻' : 'est. resident weights');
    const accentColors = await page.evaluate(() => {
      const probe = document.createElement('span');
      probe.style.background = 'var(--accent-soft)'; probe.style.color = 'var(--accent-ink)';
      document.body.append(probe);
      const style = getComputedStyle(probe), colors = { background: style.backgroundColor, color: style.color };
      probe.remove(); return colors;
    });
    await expect(page.locator('.model-variant button').first()).toHaveCSS('background-color', accentColors.background);
    await expect(page.locator('.model-variant button').first()).toHaveCSS('color', accentColors.color);
    const accentColor = await page.locator('.download-circle').evaluate(element => getComputedStyle(element).color);
    await expect(page.locator('.configuration-details.recommended p').first()).toHaveCSS('color', accentColor);
    await page.screenshot({ path: testInfo.outputPath(`download-typography-${language}.png`), fullPage: true });
    expect(errors).toEqual([]);
    expect(state.unexpected).toEqual([]);
  });
}
