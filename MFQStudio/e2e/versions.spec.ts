import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['zh-CN', 'en'] as const) {
  test(`sidebar version manager browses Releases and retains offline notes (${language})`, async ({ page }, testInfo) => {
    await mockStudioServer(page, { language });
    const en = language === 'en';
    const release = { tag_name: 'v0.3.3', name: 'MFQ Studio 0.3.3', body: '# Faster inference\n\n- Packed projection improvements\n- Updated application layout\n\n## Compatibility\n\nExisting models and settings are retained.',
      draft: false, prerelease: false, published_at: '2026-10-10T00:00:00Z', assets: [] };
    let offline = false;
    await page.route('https://api.github.com/repos/Tylogi/TyloQuant/releases?**', route => offline ? route.abort() : route.fulfill({ json: [release,
      { ...release, tag_name: 'v0.3.2', name: 'MFQ Studio 0.3.2', body: 'Previous Release notes' },
      { ...release, tag_name: 'v0.4.0rc1', prerelease: true }, { ...release, tag_name: 'nightly' }] }));
    await page.goto('/applications');
    if (testInfo.project.name === 'mobile') await page.getByRole('button', { name: en ? 'Open sidebar' : '打开侧栏', exact: true }).click();
    const card = page.getByRole('button', { name: en ? 'Open version manager' : '打开版本管理' });
    await expect(card).toContainText('v0.3.2');
    await expect(card).not.toContainText('Studio Test Model');
    await expect(card.locator('img, svg')).toHaveCount(0);
    await card.click();
    await expect(page).toHaveURL(/\/versions$/);
    await expect(page.getByRole('heading', { name: en ? 'Version manager' : '版本管理', exact: true })).toBeVisible();
    await expect(page.locator('.release-version')).toHaveCount(2);
    await expect(page.getByRole('heading', { name: 'Faster inference' })).toBeVisible();
    const previous = page.locator('.release-version').nth(1);
    await previous.locator('summary').click();
    await expect(previous).toContainText('Previous Release notes');
    await expect(page.locator('.experimental-versions')).not.toHaveAttribute('open', '');
    await expect(page.getByRole('button', { name: /Install & restart|安装并重启/ })).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.locator('.workspace').screenshot({ path: testInfo.outputPath('versions.png'), animations: 'disabled' });
    offline = true;
    await page.getByRole('button', { name: en ? 'Check now' : '检查更新', exact: true }).click();
    await expect(page.getByText(en ? 'Could not retrieve online releases' : '未能获取在线版本')).toBeVisible();
    await expect(page.locator('.release-version')).toHaveCount(2);
  });
}
