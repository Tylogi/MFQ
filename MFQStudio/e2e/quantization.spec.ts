import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

test('quantization workspace directories, source and candidate controls fit desktop and mobile', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  const groups = { NVQ: ['NVQ1-S', 'NVQ1-L', 'NVQ2J', 'NVQ2J-L', 'NVQ2J-XL', 'NVQ3J', 'NVQ3J-512', 'NVQ3J-L'], NINT: ['NINT4', 'NINT5', 'NINT6', 'NINT8'], 'MXFP4-SQ': ['MXFP4-SQ-1', 'MXFP4-SQ-2', 'MXFP4-SQ-3', 'MXFP4-SQ-F'], 'MXFP8-SQ': [1, 2, 3, 4, 5, 6, 7, 8].map(bits => `MXFP8-SQ-${bits}`), 'FP8-SQ': [1, 2, 3, 4, 5, 6, 7, 8].map(bits => `FP8-SQ-${bits}`) };
  const eligible = [...groups.NVQ, ...groups.NINT, ...groups['FP8-SQ']];
  const settings = { import_directory: '/models', export_directory: '/outputs', candidates: Object.values(groups).flat(), candidate_groups: groups, official_imatrix_url: null };
  await page.route('**/api/v1/quantization/workspace', route => route.fulfill({ json: settings }));
  await page.route('**/api/v1/quantization/files?*', route => route.fulfill({ json: { path: '/models', parent: '/', data: [{ name: 'full-model', path: '/models/full-model', directory: true, byte_size: null }, { name: 'recipe.json', path: '/models/recipe.json', directory: false, byte_size: 1200 }] } }));
  await page.route('**/api/v1/quantization/source', route => route.fulfill({ json: { path: '/models/full-model', architecture: 'qwen4_exp', tensors: 48, parameters: 35000000000, format: 'hf', full_precision: true, source_precisions: ['BF16', 'FP8'], eligible_candidates: eligible } }));
  await page.goto('/quantization');
  await expect(page.getByRole('heading', { name: 'Quantization workspace' })).toBeVisible();
  const candidates = page.getByRole('button', { name: /Candidates/ });
  await expect(candidates).toContainText('12 / 12');
  await expect(page.locator('.qw-candidate-panel')).toBeHidden();
  await page.locator('.qw-directories summary').click();
  await expect(page.getByLabel('Default import directory', { exact: true })).toHaveValue('/models');
  await expect(page.getByLabel('Default export directory', { exact: true })).toHaveValue('/outputs');
  await page.getByRole('button', { name: 'Browse', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Select original model' });
  await expect(dialog.getByRole('button', { name: /recipe.json/ })).toBeVisible();
  await expect(dialog.getByRole('textbox', { name: 'Directory path' })).toHaveValue('/models');
  await page.keyboard.press('Escape');
  await page.getByLabel('Source model directory or MFQ file').fill('/models/full-model');
  await page.getByRole('button', { name: 'Inspect model' }).click();
  await expect(page.getByText('qwen4_exp', { exact: true })).toBeVisible();
  await expect(page.getByLabel('Complete model output path')).toHaveValue('/outputs/models/full-model-quantized.mfq');
  await candidates.click();
  await expect(candidates).toHaveAttribute('aria-expanded', 'true');
  await expect(page.getByRole('checkbox', { name: 'NVQ3J-L', exact: true })).toBeChecked();
  for (const bits of [1, 2, 3]) {
    await expect(page.getByRole('checkbox', { name: `FP8-SQ-${bits}`, exact: true })).toBeChecked();
    await expect(page.getByRole('checkbox', { name: `MXFP8-SQ-${bits}`, exact: true })).toBeDisabled();
  }
  await expect(page.getByRole('checkbox', { name: 'FP8-SQ-8', exact: true })).toBeChecked();
  await expect(page.getByRole('checkbox', { name: 'FP8-SQ-8', exact: true })).toBeVisible();
  await expect(page.getByRole('checkbox', { name: 'MXFP4-SQ-F', exact: true })).toBeDisabled();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  if (testInfo.project.name === 'desktop') {
    const panel = await page.locator('.qw-candidate-panel').boundingBox();
    const toggle = await candidates.boundingBox();
    const sq = await page.locator('[data-family="MXFP4-SQ"]').boundingBox();
    const nvq = await page.locator('[data-family="NVQ"]').boundingBox();
    expect(panel!.width).toBeGreaterThan(toggle!.width * 2);
    expect(sq!.x).toBeLessThan(nvq!.x);
    expect(sq!.width).toBeGreaterThan(200);
  }
  for (const label of await page.locator('.qw-candidate-grid label').all()) {
    expect(await label.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
  }
  await page.locator('.qw-candidate-panel').screenshot({ path: testInfo.outputPath('candidate-groups.png') });
  await page.screenshot({ path: testInfo.outputPath('quantization-workspace.png'), fullPage: true });
  await candidates.click();
  await expect(page.locator('.qw-candidate-panel')).toBeHidden();
  let submissions = 0;
  let rejection = { code: 'quantization_target_above_candidates', target: 12 };
  await page.route('**/api/v1/jobs', async route => {
    if (route.request().method() !== 'POST') return route.fallback();
    submissions++;
    await route.fulfill({ status: 422, json: { error: { code: rejection.code, message: 'Recipe budget validation failed', retryable: false,
      details: { target_bpw: rejection.target, minimum_bpw: 4.5, maximum_bpw: 8.5, source_bpw: 16 } } } });
  });
  await candidates.click();
  await page.getByRole('button', { name: 'Clear', exact: true }).click();
  await page.getByRole('button', { name: 'Generate recipe', exact: true }).click();
  await expect(page.getByRole('alert')).toHaveText('Select at least one source-compatible candidate.');
  expect(submissions).toBe(0);
  await page.getByRole('button', { name: 'Select all', exact: true }).click();
  await candidates.click();
  for (const [code, target, text] of [
    ['quantization_target_above_candidates', 12, 'highest-precision candidate plan 8.5000 bpw'],
    ['quantization_target_below_candidates', 3, 'minimum budget 4.5000 bpw'],
    ['quantization_target_above_source', 24, 'source model average 16.0000 bpw'],
  ] as const) {
    rejection = { code, target };
    await page.getByLabel('Target bpw').fill(String(target));
    await page.getByRole('button', { name: 'Generate recipe', exact: true }).click();
    await expect(page.getByRole('alert')).toContainText(text);
    await expect(page.locator('.qw-section').filter({ has: page.getByRole('heading', { name: 'Generate or import recipe' }) }).getByRole('alert')).toBeVisible();
  }
  expect(submissions).toBe(3);
  await page.getByRole('alert').screenshot({ path: testInfo.outputPath('recipe-boundary-error.png') });
  await page.getByRole('button', { name: 'Calibrated', exact: true }).click();
  await expect(page.getByText(/Calibrated allocation is not selected/)).toBeVisible();
  await expect(page.getByRole('button', { name: /Official imatrix/ })).toBeDisabled();
  await page.getByText('Generate from original model', { exact: true }).click();
  const template = page.getByRole('checkbox', { name: 'Automatically apply model chat template' });
  await expect(template).toBeChecked();
  await expect(page.getByText(/already formatted records are not wrapped again/)).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.locator('.qw-subsection').screenshot({ path: testInfo.outputPath('imatrix-template-choice.png') });
  await template.uncheck();
  await expect(page.getByText(/Automatic chat templates are disabled/)).toBeVisible();
});
