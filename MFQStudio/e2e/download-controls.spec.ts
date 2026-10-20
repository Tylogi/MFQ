import { expect, test } from '@playwright/test';
import { mockStudioServer, officialCatalog } from './mockServer';

test('checked prefix control retains its paint and state across focus and a failed poll', async ({ page, context }) => {
  await mockStudioServer(page);
  let calls = 0;
  await page.route('**/api/v1/hub/download-directory', route => {
    calls += 1;
    return calls === 1 ? route.fulfill({ json: {
      directory: null, actual_directory: '/downloads', default_directory: '/downloads', evict_prefix_cache: true,
      disk_total_bytes: 2048 * 2 ** 30, disk_free_bytes: 200 * 2 ** 30,
    } }) : route.fulfill({ status: 503, json: { error: { code: 'unavailable', message: 'offline', retryable: true, details: {} } } });
  });
  await page.goto('/model-hub');
  const checkbox = page.locator('.download-prefix-reclaim input');
  await expect(checkbox).toBeChecked();
  const paint = await checkbox.evaluate(node => {
    const style = getComputedStyle(node);
    return { appearance: style.appearance, background: style.backgroundColor, border: style.borderTopColor };
  });
  expect(paint.appearance).toBe('none');
  expect(paint.background).toBe(paint.border);
  expect(paint.background).not.toBe('rgba(0, 0, 0, 0)');
  const other = await context.newPage();
  await other.bringToFront();
  await page.bringToFront();
  await expect.poll(() => calls).toBeGreaterThan(1);
  await expect(checkbox).toBeChecked();
  expect(await checkbox.evaluate(node => getComputedStyle(node).backgroundColor)).toBe(paint.background);
  await other.close();
});

test('chooses the faster source automatically and displays an existing-file reminder without adding a task', async ({ page }) => {
  await mockStudioServer(page);
  await page.route('**/api/v1/jobs/kinds', route => route.fulfill({ json: { data: [
    { kind: 'download.modelscope', payload_schema: {} }, { kind: 'download.huggingface', payload_schema: {} },
  ] } }));
  const catalog = structuredClone(officialCatalog);
  const first = catalog.data[0];
  const ms = first.sources[0];
  const hf = { ...ms, provider: 'huggingface' as const, url: `https://huggingface.co/${ms.repo_id}` };
  first.sources = [hf, ms]; first.selected_source = hf;
  await page.route('**/api/v1/hub/official*', route => route.fulfill({ json: catalog }));
  await page.route('**/api/v1/hub/download-sources/probe', route => route.fulfill({ json: { data: [
    { provider: 'huggingface', repo_id: ms.repo_id, bytes_per_second: 100 },
    { provider: 'modelscope', repo_id: ms.repo_id, bytes_per_second: 500 },
  ] } }));
  await page.route('**/api/v1/hub/models/modelscope/**', route => route.fulfill({ json: {
    provider: 'modelscope', repo_id: ms.repo_id, revision: 'test', files: [], variants: first.variants,
    downloads: 0, likes: 0, tags: [], architectures: [], modalities: ['text'], gated: false, total_bytes: 1,
  } }));
  await page.route(/\/api\/v1\/jobs$/, route => route.request().method() === 'POST'
    ? route.fulfill({ status: 409, json: { error: { code: 'download_already_exists', message: 'already exists', retryable: false, details: {} } } })
    : route.fallback());
  await page.goto('/model-hub');
  await expect(page.getByRole('option', { name: 'Auto · ModelScope' })).toBeAttached();
  const count = await page.locator('.download-circle-count').count();
  await page.locator('.model-variant button').first().click();
  await expect(page.getByText('The complete files already exist in the destination folder. No download is needed.')).toBeVisible();
  await expect(page.locator('.download-circle-count')).toHaveCount(count);
  await expect(page.locator('.download-flight')).toHaveCount(0);
});
