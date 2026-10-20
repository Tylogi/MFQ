import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';
import { analysisArtifacts, analysisFixture } from '../tests/fixtures/checkpointAnalysis';

for (const mode of ['light', 'dark', 'system-light', 'system-dark'] as const) {
  test(`layer budget uses red FFN, blue Attention and purple Total (${mode})`, async ({ page }, testInfo) => {
    const dark = mode.endsWith('dark');
    const theme = mode.startsWith('system') ? 'system' : mode;
    await mockStudioServer(page);
    await page.emulateMedia({ colorScheme: dark ? 'dark' : 'light' });
    await page.addInitScript(value => localStorage.setItem('mfq.studio.generation.v1', JSON.stringify({ language: 'en', theme: value })), theme);
    await page.route(/\/api\/v1\/models(?:\?.*)?$/, route => route.fulfill({ json: { data: [analysisArtifacts[1]] } }));
    await page.route('**/api/v1/models/*/analysis', route => route.fulfill({ json: analysisFixture(true) }));
    await page.goto('/analysis');
    const chart = page.locator('.layer-budget');
    await expect(chart.locator('.budget-legend')).toHaveText('FFNAttentionTotal');
    await chart.locator('rect[role="button"]').first().focus();
    for (const [kind, color] of [
      ['ffn', dark ? 'rgb(223, 138, 135)' : 'rgb(189, 85, 85)'],
      ['attention', dark ? 'rgb(120, 171, 223)' : 'rgb(57, 121, 186)'],
      ['total', dark ? 'rgb(185, 155, 224)' : 'rgb(134, 96, 176)'],
    ]) {
      await expect(chart.locator(`.budget-series-${kind} .budget-line`)).toHaveCSS('stroke', color);
      await expect(chart.locator(`.budget-series-${kind} .budget-dot`).first()).toHaveCSS('fill', color);
      await expect(chart.locator(`.budget-legend .budget-series-${kind} i`)).toHaveCSS('background-color', color);
      await expect(chart.locator(`.budget-readout-series.budget-series-${kind} i`)).toHaveCSS('background-color', color);
    }
    if (testInfo.project.name === 'desktop' && !mode.startsWith('system')) {
      await chart.screenshot({ path: testInfo.outputPath(`layer-budget-${mode}.png`) });
    }
  });
}

test('checkpoint analysis charts and controls fit desktop and mobile', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  await page.route(/\/api\/v1\/models(?:\?.*)?$/, route => route.fulfill({ json: { data: analysisArtifacts } }));
  await page.route('**/api/v1/models/*/analysis', route => {
    const analysis = analysisFixture(route.request().url().includes(analysisArtifacts[1].id));
    if (analysis.expert_count) analysis.tensors.push(...['gate', 'up', 'down'].map(projection => ({
      name: `model.block.0.mlp.shared_expert.${projection}.weight`, shape: [4, 8], category: 'shared_expert', projection,
      layer: 0, parameters: 32, stored_bytes: 24, bpw: 6, format: 'NINTv2',
    })));
    if (analysis.expert_count) for (const category of ['router', 'shared_expert']) {
      const bpw = category === 'router' ? 8 : 16;
      analysis.tensors.push({ name: `model.block.0.${category}.router.weight`, shape: [1, 8], category, projection: 'router',
        layer: 0, parameters: 8, stored_bytes: bpw, bpw, format: category === 'router' ? 'NINTv2' : 'BF16' });
      analysis.projections.push({ name: 'router', category, parameters: 8, stored_bytes: bpw, average_bpw: bpw,
        tensor_count: 1, formats: ['NINTv2'] });
    }
    if (analysis.expert_count) {
      analysis.tensors.push({ name: 'model.block.0.linear_attention.conv.weight', shape: [8, 4], category: 'gdn', projection: 'conv',
        layer: 0, parameters: 32, stored_bytes: 128, bpw: 32, format: 'F32' });
      analysis.projections.push({ name: 'conv', category: 'gdn', parameters: 32, stored_bytes: 128, average_bpw: 32,
        tensor_count: 1, formats: ['F32'] });
    }
    return route.fulfill({ json: analysis });
  });
  await page.goto('/analysis');
  await expect(page.getByRole('heading', { name: 'Model structure' })).toBeVisible();
  await expect(page.locator('.kv-cache-planner')).not.toHaveAttribute('open', '');
  await page.locator('.kv-cache-planner > summary').click();
  await expect(page.getByRole('row', { name: /QSA total cache/ })).toBeVisible();
  await expect(page.getByRole('row', { name: /^Indexer cache/ })).toBeVisible();
  await expect(page.locator('[data-cache-curve]')).toHaveCount(3);
  await page.getByRole('button', { name: 'ctx 8192' }).click();
  await expect(page.getByRole('spinbutton', { name: 'Expected ctx' })).toHaveValue('8192');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('analysis-kv-cache.png'), fullPage: true });
  await page.locator('.kv-cache-planner > summary').click();
  await expect(page.locator('.analysis-advanced')).not.toHaveAttribute('open', '');
  await page.locator('.analysis-advanced > summary').click();
  await expect(page.locator('.analysis-advanced')).toHaveAttribute('open', '');
  await expect(page.getByText('Expert hidden size', { exact: true })).toBeVisible();
  await expect(page.getByText('GR bottleneck dimension', { exact: true })).toBeVisible();
  await expect(page.getByText('Router activation', { exact: true })).toBeVisible();
  await expect(page.getByText('Router weight normalization', { exact: true })).toHaveCount(0);
  await expect(page.locator('.analysis-normalizations').first()).toContainText('GroupedRMSNorm');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('analysis-advanced.png'), fullPage: true });
  await expect(page.locator('.architecture-schedule')).toHaveText('3 × [3 × GDN → QSA]');
  await page.getByRole('button', { name: 'Inspect GDN', exact: true }).click();
  await expect(page.getByRole('img', { name: 'GDN · architecture detail' })).toBeVisible();
  const expand = page.getByRole('button', { name: 'Expand architecture', exact: true });
  await expand.click();
  const dialog = page.getByRole('dialog', { name: 'Architecture details' });
  await expect(dialog).toBeVisible();
  if (testInfo.project.name === 'desktop') expect((await dialog.boundingBox())!.width).toBeGreaterThan(800);
  await dialog.getByRole('tab', { name: 'QSA', exact: true }).click();
  await expect(dialog.getByText('Top-k blocks', { exact: true })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(expand).toBeFocused();
  await page.locator('.analysis-architecture').getByRole('tab', { name: 'Overview', exact: true }).click();
  await expect(page.locator('.expert-heatmap canvas')).toHaveCount(4);
  await expect(page.locator('.expert-heatmap .precision-chart-average')).toHaveCount(4);
  await expect(page.locator('.analysis-visualizations > [role="tabpanel"] > .precision-legend')).toContainText('1.08+ bpw');
  expect(await page.locator('.expert-heatmap canvas').nth(1).evaluate((canvas: HTMLCanvasElement) =>
    Array.from(canvas.getContext('2d')!.getImageData(0, 0, 1, 1).data))).toEqual([216, 216, 216, 255]);
  const shared = page.locator('.analysis-shared-ffn');
  await expect(shared).not.toHaveAttribute('open', '');
  await expect(shared.locator('.projection-bars').first()).toBeHidden();
  await shared.locator('summary').click();
  await expect(shared.locator('.projection-bars')).toHaveCount(4);
  await expect(shared.getByRole('img', { name: 'Shared FFN per-layer bpw bar heatmap' })).toBeVisible();
  await expect(shared.locator('.precision-legend')).toHaveCount(4);
  await shared.locator('summary').click();
  expect(await page.locator('.expert-heatmap canvas').evaluateAll(elements => elements.every(element => {
    const canvas = element as HTMLCanvasElement;
    const pixels = canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height).data;
    for (let i = 0; i < pixels.length; i += 4) {
      if (pixels[i] !== pixels[i + 1] || pixels[i + 1] !== pixels[i + 2]) return false;
    }
    return true;
  }))).toBe(true);
  await expect(page.locator('.layer-budget .budget-line')).toHaveCount(3);
  await expect(page.locator('.budget-legend')).toContainText('FFNAttentionTotal');
  await page.getByRole('button', { name: 'GiB', exact: true }).click();
  await expect(page.getByRole('button', { name: 'GiB', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'bpw', exact: true }).click();
  const canvas = page.locator('.expert-heatmap canvas').first();
  expect(await canvas.evaluate((element: HTMLCanvasElement) => {
    const context = element.getContext('2d')!;
    return Array.from({ length: element.height / 7 }, (_, layer) => {
      const pixels = context.getImageData(0, layer * 7, 1, 7).data;
      return Array.from({ length: 7 }, (_, row) => [0, 1, 2].every(channel => pixels[row * 4 + channel] === pixels[channel])).every(Boolean);
    }).every(Boolean);
  })).toBe(true);
  const bounds = await canvas.boundingBox();
  await canvas.click({ position: { x: bounds!.width * 0.5, y: bounds!.height * 0.5 } });
  await expect(page.locator('.chart-readout').first()).toContainText('bpw');
  await page.screenshot({ path: testInfo.outputPath('analysis-experts.png'), fullPage: true });
  await page.getByRole('tab', { name: 'Attention & other projections' }).click();
  await expect(page.getByRole('combobox', { name: 'Select projection' })).toBeVisible();
  await expect(page.getByRole('combobox', { name: 'Select projection' })).toHaveValue('all');
  await expect(page.getByRole('img', { name: 'All per-layer bpw bar heatmap' })).toBeVisible();
  await expect(page.locator('.other-projection-bars .precision-legend')).toBeVisible();
  await expect(page.locator('.other-projection-bars .precision-legend')).toContainText('1.08+ bpw');
  await expect(page.getByRole('combobox', { name: 'Select projection' }).getByRole('option', { name: 'Router · router', exact: true })).toHaveCount(1);
  await expect(page.getByRole('combobox', { name: 'Select projection' }).getByRole('option', { name: 'Shared expert router · router', exact: true })).toHaveCount(1);
  expect(await page.getByRole('combobox', { name: 'Select projection' }).locator('option').evaluateAll(options =>
    options.every(option => (option as HTMLOptionElement).value === 'shared_expert:router' || !/^(ffn|routed_experts|shared_expert):/.test((option as HTMLOptionElement).value)))).toBe(true);
  for (const projection of ['gdn:conv', 'router:router', 'shared_expert:router']) {
    await page.getByRole('combobox', { name: 'Select projection' }).selectOption(projection);
    await expect(page.locator('.other-projection-bars rect[fill^="rgb"]').first()).toHaveAttribute('fill', 'rgb(45,45,45)');
  }
  await page.getByRole('combobox', { name: 'Select projection' }).selectOption('gdn:qkv');
  await expect(page.locator('.other-projection-bars rect[fill^="rgb"]').first()).toHaveAttribute('fill', 'rgb(125,125,125)');
  await expect(page.getByRole('img', { name: 'qkv per-layer bpw bar heatmap' })).toBeVisible();
  expect(await page.locator('.projection-bars rect[fill^="rgb"]').evaluateAll(elements => elements.every(element => {
    const channels = element.getAttribute('fill')!.match(/\d+/g)!.map(Number);
    return channels[0] === channels[1] && channels[1] === channels[2];
  }))).toBe(true);
  await expect(page.locator('.projection-bars .precision-chart-average')).toHaveText('Average 5.200 bpw');
  await page.getByRole('combobox', { name: 'Select model' }).selectOption(analysisArtifacts[1].id);
  await expect(page.locator('.analysis-fact-architecture')).toContainText('qwen3_5');
  await expect(page.getByRole('combobox', { name: 'Select projection' }).getByRole('option', { name: 'GQA · query' })).toHaveCount(1);
  await expect(page.locator('.analysis-projection-table')).toContainText('GQA');
  await expect(page.locator('.kv-cache-planner')).not.toHaveAttribute('open', '');
  await page.locator('.kv-cache-planner > summary').click();
  await expect(page.getByRole('row', { name: /GQA total cache/ })).toBeVisible();
  await expect(page.getByRole('spinbutton', { name: 'Expected ctx' })).toHaveValue('4096');
  await expect(page.locator('.kv-cache-planner').getByRole('row', { name: /QSA|GDN|PLE|Indexer/ })).toHaveCount(0);
  await expect(page.locator('.architecture-tabs').getByRole('tab', { name: 'GQA', exact: true })).toBeVisible();
  await page.locator('.analysis-advanced > summary').click();
  await expect(page.locator('.analysis-advanced').getByRole('heading', { name: 'GQA', exact: true })).toBeVisible();
  await expect(page.locator('.architecture-tabs').getByRole('tab', { name: 'GR', exact: true })).toHaveCount(0);
  await expect(page.locator('.architecture-tabs').getByRole('tab', { name: 'PLE', exact: true })).toHaveCount(0);
  await page.locator('.analysis-tabs').getByRole('tab', { name: 'FFN', exact: true }).click();
  await expect(page.locator('.dense-ffn-bars .projection-bars')).toHaveCount(4);
  await expect(page.locator('.dense-ffn-bars .precision-legend')).toHaveCount(4);
  await expect(page.getByRole('img', { name: 'FFN per-layer bpw bar heatmap', exact: true })).toBeVisible();
  await expect(page.locator('.analysis-shared-ffn')).toHaveCount(0);
  const budget = page.locator('.layer-budget');
  await expect(budget.locator('.budget-legend')).toHaveText('FFNAttentionTotal');
  await expect(budget).not.toContainText('MoE');
  await budget.locator('rect[role="button"]').first().focus();
  await expect(budget.locator('.budget-series-ffn.budget-readout-series')).toContainText('4.000 bpw');
  await expect(budget.locator('.budget-series-attention.budget-readout-series')).toContainText('5.200 bpw');
  await page.screenshot({ path: testInfo.outputPath('analysis-dense-budget.png'), fullPage: true });
  await page.getByRole('combobox', { name: 'Select model' }).selectOption(analysisArtifacts[2].id);
  await expect(page.getByRole('alert')).toContainText('Model shards are incomplete');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
