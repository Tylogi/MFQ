import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

test('updates deleted and added checkpoints on window focus', async ({ page }) => {
  await mockStudioServer(page);
  const artifact = { id: 'local-deletion', name: 'Local deletion fixture', architecture: 'qwen3_5',
    format: 'mfq', shard_count: 1, total_bytes: 1024, tensor_count: 1, record_count: 1,
    dtypes: [], complete: true, loadable: true, modified_at: '2026-10-15T00:00:00Z' };
  let exists = true;
  const scans: URL[] = [];
  await page.route(/\/api\/v1\/models(?:\?.*)?$/, route => {
    scans.push(new URL(route.request().url()));
    return route.fulfill({ json: { data: exists ? [artifact] : [] } });
  });
  await page.goto('/models');
  const checkpoint = page.locator('#local-checkpoints').getByText(artifact.name, { exact: true });
  await expect(checkpoint).toBeVisible();
  exists = false;
  await page.evaluate(() => window.dispatchEvent(new Event('focus')));
  await expect(checkpoint).toHaveCount(0);
  exists = true;
  await page.evaluate(() => window.dispatchEvent(new Event('focus')));
  await expect(checkpoint).toBeVisible();
  expect(scans.length).toBeGreaterThanOrEqual(3);
  expect(scans.every(url => url.searchParams.get('refresh') === 'true')).toBe(true);
});
