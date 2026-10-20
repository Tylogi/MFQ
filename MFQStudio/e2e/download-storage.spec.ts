import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['zh-CN', 'en'] as const) {
  test(`download storage and weight-budget explanation fit the page (${language})`, async ({ page }, testInfo) => {
    const state = await mockStudioServer(page, { language });
    await page.route('**/api/v1/hub/download-directory', route => route.fulfill({ json: {
      directory: null, actual_directory: '/downloads/models', default_directory: '/downloads/models',
      disk_total_bytes: 2048 * 2 ** 30, disk_free_bytes: 200 * 2 ** 30,
    } }));
    await page.goto('/model-hub');
    const disk = page.locator('.download-storage');
    await expect(disk).toContainText('200.0 / 2048.0 GiB');
    await expect(disk).toContainText('9.8%');
    await expect(disk.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '9.765625');
    expect(await disk.locator('.download-storage-summary > span:last-child').evaluate(node => getComputedStyle(node).fontFamily)).toContain('MFQ Small Numerals');
    const directory = await page.locator('.download-directory-control').boundingBox();
    const storage = await disk.boundingBox();
    if (testInfo.project.name === 'desktop') expect(storage!.x).toBeGreaterThanOrEqual(directory!.x + directory!.width);
    const note = page.locator('.model-weight-pressure-note');
    await expect(note).toContainText(language === 'zh-CN' ? '不是磁盘占用' : 'not disk usage');
    await expect(note).toContainText(language === 'zh-CN' ? 'MoE 模型的推理预算统计内存 + 显存，Dense 模型的推理预算仅统计显存。' : 'MoE inference budgets include RAM + VRAM, while Dense inference budgets include only VRAM.');
    await expect(note.locator('br')).toHaveCount(1);
    expect(await note.evaluate(node => node.previousElementSibling?.className)).toBe('repository-line');
    expect(await note.evaluate(node => node.nextElementSibling?.className)).toBe('model-variant-list');
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    expect(state.unexpected).toEqual([]);
  });
}
