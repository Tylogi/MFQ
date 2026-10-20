import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

test('overview shares statistics scope with the compact footer and keeps concurrency live', async ({ page }) => {
  const state = await mockStudioServer(page);
  let polls = 0;
  await page.route('**/api/v1/runtime/serving?*', async route => {
    polls += 1;
    const scope = new URL(route.request().url()).searchParams.get('scope');
    const historical = scope === 'alltime';
    await route.fulfill({ json: { scope, started_at: Date.now() / 1000 - 125,
      totals: { completed_requests: historical ? 5 : 1, prompt_tokens: historical ? 20000 : 4096,
        completion_tokens: historical ? 640 : 128, prefill_tokens: 4096, cached_tokens: 0,
        prefill_tps: historical ? 2000 : 2500, decode_tps: historical ? 80 : 100,
        ttft_ms: 1600, cache_hit_rate: 0 },
      activity: [{ instance_id: 'instance-1', model: 'Studio Test Model', state: 'busy',
        active_requests: 1, queued_requests: 2, sample_age_seconds: 0,
        requests: [{ id: 'request-one', phase: 'prefilling', processed_tokens: 2048,
          generated_tokens: 0, prefill_tps: 2400, decode_tps: 0, elapsed_seconds: 1, updated_at: 1 }] }],
    } });
  });
  await page.goto('/');
  const footer = page.locator('.overview-footer-grid');
  const summary = footer.locator('.overview-session-panel');
  await expect(footer.locator('.overview-endpoint-panel')).toBeVisible();
  await expect(summary.getByRole('heading', { name: 'Session' })).toBeVisible();
  await expect(summary.locator('.overview-session-stats')).toContainText('4,096');
  await expect(summary.getByTitle('Session uptime')).toHaveText('2m 5s');
  await expect(summary.locator('p')).toContainText('failed requests · 1 active');
  await expect(page.locator('.metric-tile').filter({ hasText: 'Prefill' })).toContainText('2,500');
  await expect(page.locator('.metric-tile').filter({ hasText: 'Concurrency' })).toContainText('1');
  await expect(page.locator('.metric-tile').filter({ hasText: 'Concurrency' })).toContainText('2 queued');
  await expect(page.locator('.serving-request')).toContainText('2,048 tokens');
  await expect(page.locator('.serving-request')).toContainText('2,400 tok/s');
  await expect(page.getByText('Processed tokens / total prefill time')).toHaveCount(0);
  await expect(page.getByText('Generated tokens / total decode time')).toHaveCount(0);
  await expect.poll(() => polls).toBeGreaterThan(1);
  await page.getByRole('button', { name: 'All time', exact: true }).click();
  await expect(summary.getByRole('heading', { name: 'All time' })).toBeVisible();
  await expect(summary.locator('.overview-session-stats')).toContainText('20,000');
  await expect(page.locator('.metric-tile').filter({ hasText: 'Prefill' })).toContainText('2,000');
  await expect(page.locator('.metric-tile').filter({ hasText: 'Concurrency' })).toContainText('1');
  expect(state.unexpected).toEqual([]);
});
