import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

test('WT2 and imatrix share default layerwise loading and visible memory fallback', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  await page.route('**/api/v1/evaluations/tools', route => route.fulfill({ json: { workspace_root: '/workspace', api_base: 'http://localhost/v1', quality_available: true, reference_available: true, benchmark_available: true, accuracy_available: false } }));
  await page.route('**/api/v1/datasets', route => route.fulfill({ json: { data: [{ id: 'wt2', name: 'WikiText-2 raw · test', kind: 'wikitext2', sha256: 'a'.repeat(64), metadata: {} }] } }));
  await page.route('**/api/v1/quantization/workspace', route => route.fulfill({ json: { import_directory: '/models', export_directory: '/outputs', candidates: ['NINT4', 'NINT8'], official_imatrix_url: null } }));
  await page.route('**/api/v1/quantization/source', route => route.fulfill({ json: { path: '/models/original', architecture: 'qwen3_5', tensors: 320, parameters: 800000000, format: 'hf', full_precision: true, imatrix_supported: true, source_precisions: ['BF16'] } }));
  await page.route('**/api/v1/quantization/loading-plan', route => route.fulfill({ json: { backend: 'metal', resident_allowed: false, available_bytes: 2 ** 30, resident_required_bytes: 4 * 2 ** 30, weights_bytes: 0, workspace_bytes: 0 } }));
  for (const [path, summary, input] of [
    ['/evaluations', 'Generate WT2 logits from original model', 'Original model directory'],
    ['/quantization', 'Generate from original model', 'Source model directory or MFQ file'],
  ]) {
    await page.goto(path);
    await page.getByText(summary, { exact: true }).click();
    await page.getByLabel(input, { exact: true }).fill('/models/original');
    await page.getByRole('button', { name: 'Inspect model', exact: true }).click();
    const layerwise = page.getByRole('checkbox', { name: 'Layerwise', exact: true });
    await expect(layerwise).toBeChecked();
    await expect(layerwise).toBeEnabled();
    await layerwise.click();
    await expect(page.getByRole('status').filter({ hasText: 'Insufficient free memory' })).toBeVisible();
    await expect(layerwise).toBeChecked();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.locator('.source-loading-control').screenshot({ path: testInfo.outputPath(`${path.slice(1)}-memory-fallback.png`) });
    if (path === '/evaluations') await page.locator('.wt2-reference-generator').screenshot({ path: testInfo.outputPath('wt2-reference-generator.png') });
  }
});
