/** 验证真实浏览器中的发送、恢复、取消、输入法与弹窗交互，并检查响应式布局。 */
import { expect, test, type Page } from '@playwright/test';
import { mockStudioServer, officialCatalog } from './mockServer';

declare global {
  interface Window {
    revokedPreviews: string[];
  }
}

/** 在浏览器路由内切页，保持根 Provider 与进行中的请求不被整页刷新卸载。 */
async function navigateClient(page: Page, path: string) {
  await page.evaluate((next) => {
    window.history.pushState(null, '', next);
    window.dispatchEvent(new PopStateEvent('popstate'));
  }, path);
}

test('服务器移除对话和可执行文件，设置迁入对话页，模型宽框包含生成参数', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  await page.route('**/api/v1/runtime/models', (route) => route.fulfill({ json: { data: [{ id: 'Test Model' }, { id: 'Another Model' }] } }));
  await page.goto('/runtime');
  await expect(page.getByText('Runtime executable', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('textbox', { name: 'System prompt' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Alias mapping' })).toBeEnabled();
  await expect(page.getByText('/data/mfq/prefix-cache', { exact: true })).toBeVisible();
  await navigateClient(page, '/chat');
  const box = page.locator('.chat-model-summary');
  await expect(box.getByRole('combobox', { name: 'Chat model' })).toBeVisible();
  await expect(box.locator('small')).toBeVisible();
  await expect(box).toContainText('max tokens');
  await expect(box).toContainText('temperature');
  await expect(box).toContainText('streaming');
  if (testInfo.project.name === 'desktop') expect((await box.boundingBox())!.width).toBeGreaterThan(400);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole('button', { name: 'Chat settings' }).click();
  await expect(page.getByRole('dialog', { name: 'Chat settings' })).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'System prompt' })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('chat-settings-and-selector.png'), animations: 'disabled' });
});

test('每个模型独立修改 ctx，点击立即发送重载且不丢失另一个模型', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  const instances = [
    { id: 'flash', model: 'Qwen3.8-Flash-S4-L', state: 'ready', devices: ['metal'], active_sessions: 0,
      queued_requests: 0, context_size: 32768, context_capacity: 131072,
      memory: { resident_weight_bytes: 78.6 * 2 ** 30 } },
    { id: 'dense', model: 'Qwen3.8-27B-S4-M', state: 'ready', devices: ['metal'], active_sessions: 0,
      queued_requests: 0, context_size: 16384, context_capacity: 262144,
      memory: { resident_weight_bytes: 17.7 * 2 ** 30 } },
  ];
  await page.route('**/api/v1/runtime/instances', (route) => route.fulfill({ json: { data: instances } }));
  await page.route('**/api/v1/runtime/models', (route) => route.fulfill({ json: { data: instances.map((item) => ({ id: item.model })) } }));
  await page.route(/\/api\/v1\/runtime\/status(?:\?.*)?$/, (route) => {
    const id = new URL(route.request().url()).searchParams.get('instance_id');
    const item = instances.find((entry) => entry.id === id) ?? instances[0];
    return route.fulfill({ json: { instance_id: item.id, model: item.model,
      runtime_state: 'ready', max_context: item.context_size, context_capacity: item.context_capacity } });
  });
  let complete!: () => void;
  const pending = new Promise<void>((resolve) => { complete = resolve; });
  const reloads: { instance_id: string; context_size: number }[] = [];
  await page.route('**/api/v1/runtime/reload', async (route) => {
    const body = route.request().postDataJSON();
    reloads.push(body);
    await pending;
    const item = instances.find((entry) => entry.id === body.instance_id)!;
    item.context_size = body.context_size;
    await route.fulfill({ json: { model: item.model, max_context: item.context_size } });
  });
  const dialogs: string[] = [];
  page.on('dialog', async (dialog) => { dialogs.push(dialog.message()); await dialog.dismiss(); });
  await page.goto('/runtime');
  const flash = page.getByRole('spinbutton', { name: 'Qwen3.8-Flash-S4-L maximum context' });
  const dense = page.getByRole('spinbutton', { name: 'Qwen3.8-27B-S4-M maximum context' });
  await expect(flash).toHaveValue('32768');
  await expect(dense).toHaveValue('16384');
  const contexts = page.locator('.model-context-settings');
  await expect(contexts.locator('svg')).toHaveCount(0);
  expect(await contexts.evaluate((element) => {
    const heading = element.querySelector('.model-context-heading')!;
    const row = heading.nextElementSibling!;
    return Number.parseFloat(getComputedStyle(heading).paddingTop) >= 16
      && getComputedStyle(row).borderTopWidth === '0px';
  })).toBe(true);
  await flash.fill('8192');
  await dense.fill('65536');
  await page.getByRole('button', { name: 'Reload Qwen3.8-Flash-S4-L' }).click();
  await expect.poll(() => reloads).toEqual([{ instance_id: 'flash', context_size: 8192 }]);
  await expect(page.getByRole('button', { name: 'Reload Qwen3.8-Flash-S4-L' })).toHaveText('Reloading…');
  await expect(page.getByRole('button', { name: 'Reload Qwen3.8-27B-S4-M' })).toBeEnabled();
  await expect(page.locator('.memory-budget-actions')).toContainText('96.3');
  await page.screenshot({ path: testInfo.outputPath('per-model-context.png'), animations: 'disabled' });
  complete();
  await expect(page.getByRole('button', { name: 'Reload Qwen3.8-Flash-S4-L' })).toHaveText('Reload');
  await expect(flash).toHaveValue('8192');
  await expect(dense).toHaveValue('65536');
  expect(dialogs).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)).toBe(false);
});

test('网页运行服务允许编辑端口，无效端口不发送更改', async ({ page }) => {
  await mockStudioServer(page);
  const updates: number[] = [];
  await page.route('**/api/v1/runtime/listener', (route) => {
    if (route.request().method() === 'PUT') updates.push(route.request().postDataJSON().port);
    return route.fulfill({ json: { host: '127.0.0.1', port: 8090, configurable: true } });
  });
  await page.goto('/runtime');
  const port = page.getByRole('spinbutton', { name: 'Port', exact: true });
  await expect(port).toBeEnabled();
  await expect(port).toHaveValue('8090');
  await port.fill('65536');
  await page.getByRole('button', { name: 'Save server settings' }).click();
  await expect(page.getByText('Port must be an integer between 1 and 65535')).toBeVisible();
  expect(updates).toEqual([]);
});

test('三家架构标识贯穿模型页面，保持描线、无边框和靠右布局', async ({ page }, testInfo) => {
  const state = await mockStudioServer(page);
  const models = [
    { name: 'Qwen3.8-Flash-Next-S4-L', architecture: 'qwen4_exp', vendor: 'qwen' },
    { name: 'DeepSeek-V4.1-Flash', architecture: 'deepseek_v4', vendor: 'deepseek' },
    { name: 'GLM-5.3', architecture: 'glm5_next', vendor: 'zai' },
  ];
  const artifacts = models.map((item, index) => ({ ...item, id: `model-${index}`, format: 'mfq',
    shard_count: 1, total_bytes: 1 << 20, tensor_count: 1, record_count: 1, dtypes: [],
    complete: true, loadable: true, modified_at: '2026-01-01' }));
  const instances = models.map((item, index) => ({ id: `instance-${index}`, model: item.name,
    state: 'ready', devices: ['metal'], active_sessions: 0, queued_requests: 0, context_size: 32768 }));
  await page.route('**/api/v1/runtime/models', (route) => route.fulfill({ json: { data: models.map((item) => ({ id: item.name })) } }));
  await page.route('**/api/v1/runtime/instances', (route) => route.fulfill({ json: { data: instances } }));
  await page.route(/\/api\/v1\/runtime\/status(?:\?.*)?$/, (route) => route.fulfill({ json: {
    model: models[0].name, model_type: models[0].architecture, runtime_state: 'ready', instance_id: 'instance-0', max_context: 32768,
  } }));
  await page.route(/\/api\/v1\/models(?:\?.*)?$/, (route) => route.fulfill({ json: { data: artifacts } }));
  await page.route(/\/api\/v1\/hub\/official(?:\?.*)?$/, (route) => route.fulfill({ json: {
    ...officialCatalog, data: models.map((item, index) => ({ ...officialCatalog.data[0], ...item, id: `catalog-${index}` })),
  } }));
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('.runtime-hero-actions [data-model-vendor="qwen"]')).toBeVisible();
  const heroMark = await page.locator('.runtime-hero-actions .model-vendor-mark').boundingBox();
  const heroButton = await page.locator('.runtime-hero-actions button').first().boundingBox();
  expect(heroMark!.x + heroMark!.width).toBeLessThan(heroButton!.x);
  for (const item of models) await expect(page.locator(`.overview-model-grid [data-model-vendor="${item.vendor}"]`)).toBeVisible();
  await navigateClient(page, '/models');
  await expect(page.locator('.loaded-model-panel').getByRole('button', { name: /Use in chat|Current/ })).toHaveCount(0);
  for (const item of models) {
    await expect(page.locator(`.loaded-model-panel [data-model-vendor="${item.vendor}"]`)).toBeVisible();
    await expect(page.locator(`.model-library-panel [data-model-vendor="${item.vendor}"]`)).toBeVisible();
  }
  const row = page.locator('.model-library-panel .model-row').first();
  const nameBox = await row.locator('strong').boundingBox();
  const markBox = await row.locator('.model-vendor-mark').boundingBox();
  expect(markBox!.x).toBeGreaterThan(nameBox!.x + nameBox!.width);
  const rowButton = await row.locator('button').first().boundingBox();
  expect(markBox!.x + markBox!.width).toBeLessThan(rowButton!.x);
  await navigateClient(page, '/runtime');
  await expect(page.locator('.server-model-control')).toBeVisible();
  await expect(page.locator('.server-model-control .model-vendor-mark')).toHaveCount(0);
  await navigateClient(page, '/models');
  await page.screenshot({ path: testInfo.outputPath('model-vendors-local.png'), animations: 'disabled' });
  await navigateClient(page, '/model-hub');
  for (const item of models) await expect(page.locator(`.official-model-card [data-model-vendor="${item.vendor}"]`)).toBeVisible();
  await expect(page.locator('.model-detail-heading [data-model-vendor="qwen"]')).toBeVisible();
  await page.locator('.official-model-card').last().click();
  await expect(page.locator('.model-detail-heading [data-model-vendor="zai"]')).toBeVisible();
  await expect(page.locator('.model-variant .model-vendor-mark')).toHaveCount(0);
  for (const mark of await page.locator('.official-model-card .model-vendor-mark').all()) {
    const appearance = await mark.evaluate((node) => {
      const css = getComputedStyle(node);
      return { fill: css.fill, border: css.borderWidth, background: css.backgroundColor, stroke: css.stroke };
    });
    expect(appearance.fill).toBe('none');
    expect(appearance.border).toBe('0px');
    expect(appearance.background).toBe('rgba(0, 0, 0, 0)');
    const channels = appearance.stroke.match(/\d+/g);
    expect(new Set(channels).size).toBe(1);
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('model-vendors-downloads-light.png'), animations: 'disabled' });
  await page.evaluate(() => {
    document.documentElement.dataset.theme = 'dark';
    document.documentElement.style.colorScheme = 'dark';
  });
  await page.screenshot({ path: testInfo.outputPath('model-vendors-downloads-dark.png'), animations: 'disabled' });
  expect(errors).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

test('推理预算按内存架构分列容量，带宽用小号灰字放在下方', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  await page.goto('/model-hub');
  await expect(page.locator('.detected-hardware-summary strong')).toContainText('128 GiB URAM');
  const unified = page.locator('.detected-memory-pool');
  await expect(unified).toHaveCount(1);
  await expect(unified.locator('strong')).toHaveText('128 GiB URAM');
  await expect(unified.locator('small')).toHaveText('571.8 GiB/s');
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const checkLayout = async () => {
    for (const pool of await page.locator('.detected-memory-pool').all()) {
      const capacity = await pool.locator('strong').boundingBox();
      const bandwidth = await pool.locator('small').boundingBox();
      expect(bandwidth!.y).toBeGreaterThanOrEqual(capacity!.y + capacity!.height);
      expect(bandwidth!.x).toBe(capacity!.x);
      const size = await pool.evaluate((node) => ({
        capacity: getComputedStyle(node.querySelector('strong')!).fontSize,
        bandwidth: getComputedStyle(node.querySelector('small')!).fontSize,
        color: getComputedStyle(node.querySelector('small')!).color,
      }));
      expect(parseFloat(size.bandwidth)).toBeLessThan(parseFloat(size.capacity));
      expect(size.color).toBe(await page.locator('.detected-configuration > div > span').first().evaluate((node) => getComputedStyle(node).color));
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  };
  await checkLayout();
  await page.locator('.detected-configuration').screenshot({ path: testInfo.outputPath('unified-memory.png') });
  await page.route(/\/api\/v1\/hub\/official(?:\?.*)?$/, (route) => route.fulfill({ json: {
    ...officialCatalog, system: { platform: 'Windows', machine: 'AMD64', backend: 'cuda',
      cpu_name: 'AMD Ryzen 5 9600X', gpu_names: ['NVIDIA GeForce RTX 5090'], physical_memory_bytes: 64 * 2 ** 30,
      memory_pools: [{ kind: 'vram', capacity_bytes: 32 * 2 ** 30, bandwidth_bytes_per_second: 1792e9 },
        { kind: 'ram', capacity_bytes: 64 * 2 ** 30, bandwidth_bytes_per_second: 104 * 2 ** 30 }] },
  } }));
  await page.getByRole('button', { name: 'Refresh', exact: true }).click();
  await expect(page.locator('.detected-memory-pool strong')).toHaveText(['32 GiB VRAM', '64 GiB RAM']);
  await expect(page.locator('.detected-memory-pool small')).toHaveText(['1,668.9 GiB/s', '104 GiB/s']);
  await checkLayout();
  await page.locator('.detected-configuration').screenshot({ path: testInfo.outputPath('discrete-memory.png') });
  expect(errors).toEqual([]);
});

test('注册资产显示文件总大小，已载入模型可在生成期间切换且无需登记资产', async ({ page }) => {
  const state = await mockStudioServer(page, { holdResponse: true });
  const nextModel = 'Loaded without registration';
  const assets = [32, 8].map((size, index) => ({ id: `asset-${index}`, name: `Registered checkpoint ${index}`,
    architecture: 'test', format: 'mfq', total_bytes: size * 2 ** 30, shard_count: 1,
    tensor_count: 1, record_count: 1, dtypes: [], complete: true, loadable: true, modified_at: '2026-01-01' }));
  await page.route(/\/api\/v1\/models(?:\?.*)?$/, (route) => route.fulfill({ json: { data: assets } }));
  await page.route('**/api/v1/runtime/models', (route) => route.fulfill({ json: { data: [] } }));
  await page.route('**/api/v1/runtime/instances', (route) => route.fulfill({ json: { data: [
    { id: 'instance-1', model: 'Studio Test Model', state: 'ready', devices: ['cpu'] },
    { id: 'instance-2', model: nextModel, state: 'ready', devices: ['cpu'] },
  ] } }));
  let forks = 0;
  await page.route('**/api/v1/sessions/session-1/fork', (route) => {
    forks += 1;
    expect(route.request().postDataJSON().model).toBe(nextModel);
    return route.fulfill({ json: { id: 'session-2', model: nextModel, mode: 'text', state: 'idle',
      revision: 0, title: 'Regression conversation', created_at: '2026-01-01', updated_at: '2026-01-01', metadata: {} } });
  });
  await page.route('**/api/v1/sessions/session-2/messages', (route) => route.fulfill({ json: { data: [] } }));
  await page.route('**/api/v1/sessions/session-2/responses?*', (route) => route.fulfill({ json: { data: [] } }));
  await page.goto('/models');
  await expect(page.locator('.model-workbench-summary > div').last()).toContainText('Registered model assets size');
  await expect(page.locator('.model-workbench-summary > div').last().locator('strong')).toHaveText('40 GiB');
  await expect(page.getByText('Switching keeps the registered asset')).toHaveCount(0);
  await navigateClient(page, '/chat');
  const selector = page.getByRole('combobox', { name: 'Chat model', exact: true });
  await expect(selector.locator('option')).toHaveText(['Studio Test Model', nextModel]);
  const input = page.getByRole('textbox', { name: 'Message', exact: true });
  await expect(input).toBeEnabled();
  await input.fill('Keep this response on the original model');
  await page.getByRole('button', { name: 'Send' }).click();
  await expect.poll(() => state.submissions).toBe(1);
  await expect(selector).toBeEnabled();
  await selector.selectOption(nextModel);
  await expect(selector).toHaveValue(nextModel);
  expect(forks).toBe(0);
  state.releaseResponse();
  await expect.poll(() => forks).toBe(1);
  await expect(input).toBeEnabled();
  await expect(selector).toHaveValue(nextModel);
  expect(state.submissions).toBe(1);
  expect(state.requests.filter((request) => request.startsWith('POST /api/v1/models'))).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

test('下载来源与九宫格保留统一字体，仅降低文字对比度', async ({ page }) => {
  await mockStudioServer(page);
  await page.goto('/model-hub');
  await expect(page.locator('.model-source-picker select')).toBeVisible();
  for (const theme of ['light', 'dark']) {
    await page.evaluate((value) => {
      document.documentElement.dataset.theme = value;
      document.documentElement.style.colorScheme = value;
    }, theme);
    await expect.poll(() => page.evaluate(() => {
      const color = getComputedStyle(document.querySelector('.model-detail-panel > p')!).color;
      return Array.from(document.querySelectorAll('.model-source-picker select, .model-metadata-grid dd')).every((node) => getComputedStyle(node).color === color);
    })).toBe(true);
    const styles = await page.evaluate(() => {
      const body = getComputedStyle(document.body);
      const detail = getComputedStyle(document.querySelector('.model-detail-panel > p')!);
      return Array.from(document.querySelectorAll('.model-source-picker select, .model-metadata-grid dd')).map((node) => {
        const css = getComputedStyle(node);
        return { font: css.fontFamily, expectedFont: body.fontFamily, weight: css.fontWeight,
          color: css.color, expectedColor: detail.color, primaryColor: body.color };
      });
    });
    for (const css of styles) {
      expect(css.font).toBe(css.expectedFont);
      expect(css.weight).toBe('400');
      expect(css.color).toBe(css.expectedColor);
      expect(css.color).not.toBe(css.primaryColor);
    }
  }
});

test('运行资源按模型分段，四个槽共享颜色，端点包含 v1', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  const instances = [1, 2, 3, 4].map((index) => ({
    id: `resource-${index}`, model: `Resource Model ${index}`, state: 'ready', devices: ['metal'],
    active_sessions: 0, queued_requests: 0, started_at: `2026-01-01T00:00:0${index}Z`,
    memory: { resident_weight_bytes: index * 2 ** 30, kv_bytes: index * 2 ** 20,
      context_count: index, prefix_cache_blocks: index * 2, ssd_experts: true,
      ssd_expert_bytes: index * 2 ** 30, ssd_ple: true, ssd_ple_bytes: index * 2 ** 30 },
  }));
  await page.route('**/api/v1/runtime/instances', (route) => route.fulfill({ json: { data: instances } }));
  await page.route(/\/api\/v1\/runtime\/status(?:\?.*)?$/, (route) => route.fulfill({ json: {
    runtime_state: 'ready', model: 'Resource Model 1', instance_id: 'resource-1',
    runtime_memory_budget_bytes: 32 * 2 ** 30, runtime_memory_effective_budget_bytes: 20 * 2 ** 30,
  } }));
  await page.goto('/');
  await expect(page.getByText('Resource hierarchy', { exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Runtime resources', exact: true })).toBeVisible();
  await expect(page.locator('.memory-model-legend i')).toHaveCount(4);
  await expect(page.getByText('10 contexts · 20 cache blocks')).toBeVisible();
  const colors = await page.locator('.memory-model-legend i').evaluateAll((dots) => dots.map((dot) => getComputedStyle(dot).backgroundColor));
  expect(new Set(colors).size).toBe(4);
  for (const tier of ['weights', 'kv', 'experts', 'ple']) {
    const bars = page.locator(`[data-tier="${tier}"] .memory-tier-track > span`);
    await expect(bars).toHaveCount(4);
    expect(await bars.evaluateAll((segments) => segments.map((segment) => getComputedStyle(segment).backgroundColor))).toEqual(colors);
    const width = parseFloat(await bars.first().evaluate((segment) => (segment as HTMLElement).style.width));
    expect(width).toBeCloseTo(tier === 'weights' ? 5 : tier === 'kv' ? 1 / 1024 / 10 * 100 : 10);
  }
  await expect(page.locator('[data-tier="weights"] .memory-tier-heading > span')).toHaveText('10 GiB / 20 GiB');
  await expect(page.locator('[data-tier="kv"] .memory-tier-heading > span')).toHaveText('10 MiB / 10 GiB');
  await expect(page.getByText(/Colors show each model/)).toHaveCount(0);
  await expect(page.locator('.overview-endpoint-panel code')).toHaveText('http://127.0.0.1:8090/v1');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.locator('.overview-memory-panel').screenshot({ path: testInfo.outputPath('resource-hierarchy.png') });
});

test('加载条显示实际任务进度，部分缺失的资源明细保留已知数值', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  const job = { id: 'loading-job', kind: 'model.load', payload: { model: 'Loading Model' },
    progress: 0.25, status: 'running', cancel_requested: false,
    created_at: '2026-01-01', updated_at: '2026-01-01' };
  const instances = [{ id: 'loading', model: 'Loading Model', state: 'loading', devices: ['metal'],
    active_sessions: 0, queued_requests: 0, context_size: 32768 }];
  let advance!: () => void;
  const progressEvent = new Promise<void>((resolve) => { advance = resolve; });
  await page.route(/\/api\/v1\/jobs(?:\?.*)?$/, (route) => route.fulfill({ json: { data: [job] } }));
  await page.route('**/api/v1/jobs/loading-job/events/stream', async (route) => {
    await progressEvent;
    job.progress = 0.75;
    await route.fulfill({ contentType: 'text/event-stream', body: `data: ${JSON.stringify({
      job_id: job.id, sequence: 1, type: 'progress', level: 'info', progress: 0.75,
      data: {}, created_at: job.updated_at,
    })}\n\n` }).catch(() => undefined);
  });
  await page.route('**/api/v1/runtime/instances', (route) => route.fulfill({ json: { data: instances } }));
  await page.route(/\/api\/v1\/runtime\/status(?:\?.*)?$/, (route) => route.fulfill({ json: {
    runtime_state: 'loading', model: 'Loading Model', instance_id: 'loading',
  } }));
  await page.route(/\/api\/v1\/models(?:\?.*)?$/, (route) => route.fulfill({ json: { data: [{
    id: 'artifact', name: 'Loading Model', architecture: 'qwen35', loadable: true,
    total_bytes: 1024, shard_count: 1,
  }] } }));
  await page.goto('/');
  await expect(page.locator('.runtime-hero progress')).toHaveAttribute('value', '0.25');
  advance();
  await expect(page.locator('.runtime-hero progress')).toHaveAttribute('value', '0.75');
  await navigateClient(page, '/models');
  await expect(page.locator('.loaded-model-panel progress')).toHaveAttribute('value', '0.75');
  await expect(page.locator('.model-library-panel progress')).toHaveAttribute('value', '0.75');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('model-loading-progress.png') });
  job.status = 'succeeded';
  job.progress = 1;
  await page.route('**/api/v1/runtime/instances', (route) => route.fulfill({ json: { data: [
    { id: 'known', model: 'Known Model', state: 'ready', devices: ['metal'], active_sessions: 0, queued_requests: 0, memory: {
      resident_weight_bytes: 2 ** 30, kv_bytes: 0, context_count: 0, prefix_cache_blocks: 0,
      ssd_experts: false, ssd_expert_bytes: 0, ssd_ple: false, ssd_ple_bytes: 0,
    } }, { id: 'unknown', model: 'Unknown Model', state: 'ready', devices: ['metal'], active_sessions: 0, queued_requests: 0, memory: null },
  ] } }));
  await page.route(/\/api\/v1\/runtime\/status(?:\?.*)?$/, (route) => route.fulfill({ json: {
    runtime_state: 'ready', model: 'Known Model', runtime_memory_effective_budget_bytes: 4 * 2 ** 30,
  } }));
  await page.goto('/');
  await expect(page.locator('[data-tier="weights"] .memory-tier-heading > span')).toHaveText('≥ 1 GiB / 4 GiB');
  await expect(page.locator('[data-tier="weights"] .memory-tier-notice')).toContainText('Unknown Model');
  await expect(page.locator('[data-tier="weights"] .memory-tier-track > span')).toHaveCount(1);
  await page.unrouteAll({ behavior: 'ignoreErrors' });
});

for (const kind of ['model.load', 'download.modelscope']) {
  test(`${kind} 事件流中断后切页返回仍同步进度`, async ({ page }) => {
    await mockStudioServer(page);
    const loading = kind === 'model.load';
    const job = { id: 'resume-job', kind, payload: loading ? { model: 'Loading Model' } : { repo_id: 'Tylogi/test-MFQ' },
      progress: 0.25, status: 'running', cancel_requested: false,
      created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z' };
    await page.route(/\/api\/v1\/jobs(?:\?.*)?$/, (route) => route.fulfill({ json: { data: [job] } }));
    await page.route(/\/api\/v1\/jobs\/resume-job\/events\/stream(?:\?.*)?$/, (route) => route.fulfill({ status: 503, json: { error: 'stream offline' } }));
    if (loading) {
      await page.route('**/api/v1/runtime/instances', (route) => route.fulfill({ json: { data: [{
        id: 'loading', model: 'Loading Model', state: 'loading', devices: ['metal'], active_sessions: 0, queued_requests: 0,
      }] } }));
      await page.route(/\/api\/v1\/runtime\/status(?:\?.*)?$/, (route) => route.fulfill({ json: {
        runtime_state: 'loading', model: 'Loading Model', instance_id: 'loading',
      } }));
    }
    const path = loading ? '/' : '/model-hub';
    await page.goto(path);
    if (!loading) await page.locator('.download-circle').click();
    const progress = () => page.locator(loading ? '.runtime-hero progress' : '.download-queue-item progress');
    await expect(progress()).toHaveAttribute('value', '0.25');
    await navigateClient(page, '/chat');
    job.progress = 0.75;
    job.updated_at = '2026-01-01T00:00:05Z';
    await navigateClient(page, path);
    if (!loading) await page.locator('.download-circle').click();
    await expect(progress()).toHaveAttribute('value', '0.75');
    job.progress = 0.9;
    job.updated_at = '2026-01-01T00:00:06Z';
    await expect(progress()).toHaveAttribute('value', '0.9');
  });
}

test('页面按需请求自己的资源，概览不预载其他业务列表', async ({ page }) => {
  const state = await mockStudioServer(page);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Overview', exact: true })).toBeVisible();
  for (const path of ['sessions', 'datasets', 'evaluations', 'models', 'runtime/logs', 'runtime/profiles', 'mcp/servers']) {
    expect(state.requests).not.toContain(`GET /api/v1/${path}`);
  }
  const routes = [
    ['/evaluations', '/api/v1/evaluations'],
    ['/model-hub', null],
    ['/quantization', null],
    ['/settings', '/api/v1/presets'],
    ['/runtime', '/api/v1/mcp/servers'],
    ['/resources', '/api/v1/runtime/profiles'],
    ['/logs', '/api/v1/runtime/logs'],
  ] as const;
  for (const [path, endpoint] of routes) {
    await navigateClient(page, path);
    await expect(page.locator('main h1')).toBeVisible();
    if (endpoint) await expect.poll(() => state.requests.includes(`GET ${endpoint}`)).toBe(true);
  }
  expect(errors).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

test('模型下载留在本页，飞入圆圈后打开第三个队列标签，量化工作台为空', async ({ page }, testInfo) => {
  const state = await mockStudioServer(page);
  const jobs: unknown[] = [{ id: 'load', kind: 'model.load', status: 'succeeded', progress: 1,
    payload: { repo_id: 'Not a download' }, cancel_requested: false, created_at: '2026-01-01', updated_at: '2026-01-01' }];
  await page.route('**/api/v1/jobs/kinds', (route) => route.fulfill({ json: { data: [{ kind: 'download.modelscope', payload_schema: {} }] } }));
  await page.route(/\/api\/v1\/jobs(?:\?.*)?$/, async (route) => {
    if (route.request().method() === 'GET') return route.fulfill({ json: { data: jobs } });
    const { kind, payload } = route.request().postDataJSON();
    expect(kind).toBe('download.modelscope');
    const job = { id: 'download', kind, payload, status: 'succeeded', progress: 1,
      cancel_requested: false, created_at: '2026-01-01', updated_at: '2026-01-01' };
    jobs.unshift(job);
    return route.fulfill({ json: job });
  });
  await page.goto('/model-hub');
  await expect(page.getByRole('img', { name: 'Apple · METAL' })).toBeVisible();
  await expect(page.getByText('Apple M5 Max · 18 CPU / 40 GPU · 128 GiB URAM')).toBeVisible();
  await expect(page.getByRole('tab')).toHaveText(['Official', 'Community', 'Download queue']);
  await page.getByRole('button', { name: 'Download', exact: true }).first().click();
  await expect(page).toHaveURL('/model-hub');
  await expect(page.getByRole('tab', { name: 'Official' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('.download-flight')).toBeVisible();
  expect(await page.locator('.download-flight').evaluate((node) => getComputedStyle(node).animationName)).toBe('download-fly');
  expect(await page.locator('.download-circle').evaluate((node) => {
    const rect = node.getBoundingClientRect();
    return document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2)?.closest('.download-circle') === node;
  })).toBe(true);
  await page.getByRole('button', { name: 'Download queue', exact: true }).click();
  await expect(page.getByRole('tab', { name: 'Download queue' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('.download-queue-item')).toHaveCount(1);
  await expect(page.locator('.download-queue-item')).toContainText('example/studio-layout-test');
  await expect(page.locator('.download-queue-item')).toContainText('100%');
  await expect(page.getByText('Not a download')).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('download-queue.png'), animations: 'disabled' });
  await navigateClient(page, '/quantization');
  await expect(page.locator('.quantization-empty-panel')).toHaveCount(4);
  await expect(page.locator('.quantization-empty-grid')).toHaveText('');
  await expect(page.locator('.quantization-empty-grid input, .quantization-empty-grid button')).toHaveCount(0);
  expect(state.requests).not.toContain('GET /api/v1/artifacts/lineage');
  expect(state.unexpected).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath('empty-quantization.png'), animations: 'disabled' });
});

test('生成期间离开聊天页后返回仍完成同一次请求，草稿按会话保留', async ({ page }) => {
  const state = await mockStudioServer(page, { holdResponse: true });
  await page.goto('/chat');
  const input = page.getByRole('textbox', { name: 'Message', exact: true });
  await expect(input).toBeEnabled();
  await input.fill('Continue in background');
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await expect.poll(() => state.submissions).toBe(1);
  await navigateClient(page, '/evaluations');
  await expect(page.getByRole('heading', { name: 'Evaluations', exact: true })).toBeVisible();
  state.releaseResponse();
  await navigateClient(page, '/chat');
  await expect(page.locator('.message-assistant strong')).toHaveText('formatted content');
  await expect(input).toBeEnabled();
  expect(state.submissions).toBe(1);
  expect(state.cancellations).toBe(0);
  await input.fill('Draft survives navigation');
  await navigateClient(page, '/');
  await expect(page.getByRole('heading', { name: 'Overview', exact: true })).toBeVisible();
  await navigateClient(page, '/chat');
  await expect(input).toHaveValue('Draft survives navigation');
});

test('待发送附件切页保留，移除时释放预览 URL', async ({ page }) => {
  await mockStudioServer(page);
  await page.goto('/chat');
  await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toBeEnabled();
  await page.evaluate(() => {
    const revoke = URL.revokeObjectURL.bind(URL);
    window.revokedPreviews = [];
    URL.revokeObjectURL = (url) => {
      window.revokedPreviews.push(url);
      revoke(url);
    };
  });
  await page.locator('input[type="file"]').setInputFiles({
    name: 'example.png',
    mimeType: 'image/png',
    buffer: Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/hS8AAAAASUVORK5CYII=',
      'base64',
    ),
  });
  await expect(page.getByText('example.png')).toBeVisible();
  const preview = await page.locator('.attachment-chip img').getAttribute('src');
  await navigateClient(page, '/settings');
  await expect(page.getByRole('heading', { name: 'Settings', exact: true })).toBeVisible();
  await navigateClient(page, '/chat');
  await expect(page.getByText('example.png')).toBeVisible();
  expect(await page.locator('.attachment-chip img').getAttribute('src')).toBe(preview);
  await page.getByRole('button', { name: 'Remove attachment' }).click();
  await expect(page.getByText('example.png')).toHaveCount(0);
  expect(await page.evaluate(() => window.revokedPreviews)).toContain(preview);
});

test('生成 POST 被拒绝时保留草稿与附件，重试成功后清空输入', async ({ page }) => {
  const state = await mockStudioServer(page, { rejectFirstSubmission: true });
  await page.goto('/chat');
  const input = page.getByRole('textbox', { name: 'Message', exact: true });
  await expect(input).toBeEnabled();
  await input.fill('Retry this request');
  await page.locator('input[type="file"]').setInputFiles({
    name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('note'),
  });
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Request rejected');
  await expect(input).toHaveValue('Retry this request');
  await expect(page.locator('.attachment-chip')).toContainText('notes.txt');
  await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await expect(page.locator('.message-assistant strong')).toHaveText('formatted content');
  await expect(input).toHaveValue('');
  await expect(page.locator('.attachment-chip')).toHaveCount(0);
  expect(state.submissions).toBe(2);
  expect(state.unexpected).toEqual([]);
});

test('发送流式回答并完成历史同步', async ({ page }, testInfo) => {
  const state = await mockStudioServer(page);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/chat');
  const input = page.getByRole('textbox', { name: 'Message', exact: true });
  await expect(input).toBeEnabled();
  await input.fill('Explain this project');
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await expect(page.locator('.message-assistant strong')).toHaveText('formatted content');
  await expect(input).toBeEnabled();
  await expect(page.locator('.message-assistant')).toHaveCount(1);
  await expect.poll(() => state.submissions).toBe(1);
  expect(state.unexpected).toEqual([]);
  expect(errors).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  await page.screenshot({ path: testInfo.outputPath('chat.png'), fullPage: true });
});

test('输入法确认不会误发，Shift Enter 保留换行', async ({ page }) => {
  const state = await mockStudioServer(page);
  await page.goto('/chat');
  const input = page.getByRole('textbox', { name: 'Message', exact: true });
  await expect(input).toBeEnabled();
  await input.fill('中文候选');
  await input.dispatchEvent('compositionstart');
  await input.dispatchEvent('keydown', {
    key: 'Enter',
    code: 'Enter',
    isComposing: true,
    keyCode: 229,
  });
  await input.dispatchEvent('compositionend');
  await expect(input).toHaveValue('中文候选');
  expect(state.submissions).toBe(0);
  await input.press('Shift+Enter');
  await expect(input).toHaveValue('中文候选\n');
  expect(state.submissions).toBe(0);
});

test('历史同步失败保留回答且恢复时不重复生成', async ({ page }) => {
  const state = await mockStudioServer(page, { failFirstSync: true });
  await page.goto('/chat');
  await page.getByRole('textbox', { name: 'Message', exact: true }).fill('Keep this response');
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  const retry = page.getByRole('button', { name: 'Synchronize response', exact: true });
  await expect(retry).toBeVisible();
  await expect(page.getByText('formatted content', { exact: true })).toBeVisible();
  await retry.click();
  await expect(retry).toBeHidden();
  await expect(page.locator('.message-assistant')).toHaveCount(1);
  expect(state.submissions).toBe(1);
});

test('停止挂起生成后恢复输入', async ({ page }) => {
  const state = await mockStudioServer(page, { waitForCancel: true });
  await page.goto('/chat');
  const input = page.getByRole('textbox', { name: 'Message', exact: true });
  await input.fill('Stop this request');
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await expect.poll(() => state.submissions).toBe(1);
  await page.getByRole('button', { name: 'Stop generation', exact: true }).click();
  await expect(input).toBeEnabled();
  expect(state.cancellations).toBe(1);
  expect(state.submissions).toBe(1);
});

test('路由导航及模型目录弹窗键盘焦点', async ({ page }, testInfo) => {
  const state = await mockStudioServer(page);
  await page.goto('/models');
  const addModel = page.getByRole('button', { name: 'Add model', exact: true }).first();
  await expect(addModel).toBeEnabled();
  await addModel.click();
  const dialog = page.getByRole('dialog', { name: 'Choose model folder' });
  await expect(dialog).toBeVisible();
  await page.keyboard.press('Tab');
  expect(await dialog.evaluate((element) => element.contains(document.activeElement))).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('model-dialog.png'), fullPage: true });
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(addModel).toBeFocused();
  if (testInfo.project.name === 'mobile')
    await page.getByRole('button', { name: 'Open sidebar', exact: true }).click();
  await page.getByRole('button', { name: 'Overview', exact: true }).click();
  await expect(page).toHaveURL('/');
  await expect(page.getByRole('heading', { name: 'Overview', exact: true })).toBeVisible();
  await expect(page.locator('.sidebar')).not.toHaveClass(/open/);
  await expect(page.getByRole('main')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  expect(state.unexpected).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath('overview.png'), fullPage: true, animations: 'disabled' });
});

test('明暗主题的页面内容保持在工作区内且可滚动访问', async ({ page }, testInfo) => {
  const state = await mockStudioServer(page);
  const routes = [['/', 'Overview'], ['/models', 'Models'], ['/settings', 'Settings'],
    ['/model-hub', 'Model downloads']] as const;
  for (const theme of ['light', 'dark']) {
    await page.goto('/settings');
    await expect(page.getByRole('heading', { name: 'Settings', exact: true })).toBeVisible();
    const inherit = page.getByRole('switch', { name: 'Use model or architecture defaults' });
    await expect(inherit).toBeChecked();
    await inherit.click();
    await expect(page.locator('#settings-system-prompt')).toBeEnabled();
    await inherit.click();
    await expect(page.locator('#settings-system-prompt')).toBeDisabled();
    await page.getByRole('combobox', { name: 'Theme', exact: true }).selectOption(theme);
    await page.getByRole('button', { name: 'Apply settings', exact: true }).first().click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
    for (const [path, title] of routes) {
      await navigateClient(page, path);
      await expect(page.getByRole('heading', { name: title, exact: true })).toBeVisible();
      if (path === '/model-hub') {
        await expect(page.locator('.model-detail-panel')).toBeVisible();
        await expect(page.locator('.model-variant')).toHaveCount(3);
      }
      const layout = await page.locator('.dashboard-view').evaluate((element) => ({
        width: element.clientWidth,
        scrollWidth: element.scrollWidth,
        height: element.clientHeight,
        bottom: element.getBoundingClientRect().bottom,
        viewport: window.innerHeight,
      }));
      expect(layout.scrollWidth).toBeLessThanOrEqual(layout.width + 1);
      expect(layout.height).toBeGreaterThan(0);
      expect(layout.bottom).toBeLessThanOrEqual(layout.viewport + 1);
      await page.screenshot({ path: testInfo.outputPath(`${title.replaceAll(' ', '-')}-${theme}.png`), fullPage: true, animations: 'disabled' });
    }
  }
  expect(state.unexpected).toEqual([]);
});
