/** 验证真实浏览器中的发送、恢复、取消、输入法与弹窗交互，并检查响应式布局。 */
import { expect, test, type Page } from '@playwright/test';
import { mockStudioServer } from './mockServer';

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
    ['/quantization', '/api/v1/jobs/kinds'],
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
  await expect(page.getByRole('main')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  expect(state.unexpected).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath('overview.png'), fullPage: true });
});
