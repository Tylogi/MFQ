import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['en', 'zh-CN'] as const) {
  test(`download ETA updates without touching the transfer (${language})`, async ({ page }) => {
    await mockStudioServer(page, { language });
    const job = { id: 'eta-download', kind: 'download.modelscope', payload: { repo_id: 'Tylogi/test-MFQ' },
      progress: 0.25, status: 'running', cancel_requested: false,
      created_at: '2026-10-15T00:00:00Z', updated_at: '2026-10-15T00:00:00Z',
      progress_data: { downloaded_bytes: 1024, total_bytes: 8192, bytes_per_second: 1024 } };
    const writes: string[] = [];
    page.on('request', request => { if (request.method() !== 'GET' && request.url().includes('/api/v1/jobs')) writes.push(request.url()); });
    await page.route(/\/api\/v1\/jobs(?:\?.*)?$/, route => route.fulfill({ json: { data: [job] } }));
    await page.route(/\/api\/v1\/jobs\/eta-download\/events\/stream(?:\?.*)?$/, route => route.fulfill({ status: 503, json: {} }));
    await page.goto('/model-hub');
    await page.locator('.download-circle').click();
    const eta = page.getByLabel(language === 'en' ? 'Estimated time remaining' : '预计剩余时间');
    await expect(eta).toContainText(language === 'en' ? '7s remaining' : '预计剩余 7s');
    job.progress_data = { downloaded_bytes: 4096, total_bytes: 8192, bytes_per_second: 2048 };
    job.updated_at = '2026-10-15T00:00:05Z';
    await expect(eta).toContainText(language === 'en' ? '2s remaining' : '预计剩余 2s');
    job.progress_data.downloaded_bytes = 8192;
    job.updated_at = '2026-10-15T00:00:10Z';
    await expect(eta).toContainText(language === 'en' ? 'Finishing download' : '正在收尾');
    expect(writes).toEqual([]);
  });
}
