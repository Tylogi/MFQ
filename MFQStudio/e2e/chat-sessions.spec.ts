import { expect, test, type Page } from '@playwright/test';
import { mockStudioServer } from './mockServer';
import type { Session } from '../src/shared/api/types';

async function mockConversations(page: Page, language: 'zh-CN' | 'en', count: number) {
  await mockStudioServer(page, { language });
  let sessions: Session[] = Array.from({ length: count }, (_, index) => ({
    id: `session-${index + 1}`, model: 'Studio Test Model', mode: 'text', state: 'idle', revision: 0,
    title: `Conversation ${index + 1} with a longer title for layout checks`,
    runtime_instance_id: 'instance-1', created_at: '2026-09-23T00:00:00Z', updated_at: '2026-09-23T00:00:00Z', metadata: {},
  }));
  const state = { deletions: 0, renames: 0 };
  await page.route('**/api/v1/sessions**', async route => {
    const url = new URL(route.request().url());
    const method = route.request().method();
    if (url.pathname === '/api/v1/sessions' && method === 'GET') {
      const offset = Number(url.searchParams.get('offset') || 0);
      return route.fulfill({ json: { data: sessions.slice(offset, offset + 200) } });
    }
    const match = url.pathname.match(/^\/api\/v1\/sessions\/([^/]+)(?:\/(messages|responses))?$/);
    if (!match) return route.fallback();
    if (match[2] && method === 'GET') return route.fulfill({ json: { data: [] } });
    const session = sessions.find(item => item.id === match[1]);
    if (!session) return route.fulfill({ status: 404, json: { error: { message: 'Unknown conversation' } } });
    if (method === 'DELETE') {
      sessions = sessions.filter(item => item.id !== session.id);
      state.deletions += 1;
      return route.fulfill({ status: 204 });
    }
    if (method === 'PATCH') {
      Object.assign(session, route.request().postDataJSON(), { revision: session.revision + 1 });
      state.renames += 1;
    }
    return route.fulfill({ json: session });
  });
  return state;
}

async function openConversations(page: Page, en: boolean) {
  await expect(page.locator('.chat-screen-header')).toBeVisible();
  const expand = page.getByRole('button', { name: en ? 'Expand conversations' : '展开会话列表', exact: true });
  if (await expand.isVisible()) await expand.click();
  await expect(page.getByRole('complementary', { name: en ? 'Conversations' : '会话列表' })).toBeVisible();
}

test('chat context keeps its identity when the module is re-evaluated', async ({ page }) => {
  const client = await page.request.get('/@vite/client');
  test.skip(!client.ok() || !client.headers()['content-type']?.includes('javascript'), 'HMR identity requires the Vite development server.');
  await mockConversations(page, 'en', 2);
  await page.goto('/chat');
  await expect(page.locator('.chat-session-row')).toHaveCount(2);
  const context = await page.evaluate(async () => {
    const path = '/src/features/chat/ChatProvider.tsx';
    const clientPath = '/@vite/client';
    const previous = await import(path);
    const client = await import(clientPath);
    const before = client.createHotContext(path).data.chatContext;
    const updated = await import(`${path}?context-regression=${Date.now()}`);
    const after = client.createHotContext(path).data.chatContext;
    return { exists: Boolean(before?.Provider), same: before === after, reEvaluated: previous.useChat !== updated.useChat };
  });
  expect(context).toEqual({ exists: true, same: true, reEvaluated: true });
  await expect(page.locator('.failure-page')).toHaveCount(0);
});

for (const language of ['zh-CN', 'en'] as const) {
  test(`conversation collapse, rename and confirmed bulk deletion (${language})`, async ({ page }) => {
    const en = language === 'en';
    const state = await mockConversations(page, language, 2);
    await page.goto('/chat');
    await openConversations(page, en);
    const sidebar = page.locator('.chat-session-sidebar');
    await expect(sidebar.locator('.chat-session-row')).toHaveCount(2);
    const collapse = sidebar.getByRole('button', { name: en ? 'Collapse conversations' : '收起会话列表' });
    await expect(collapse).toHaveAttribute('aria-expanded', 'true');
    await expect(collapse.locator('svg')).toHaveCount(1);
    await collapse.click();
    await expect(sidebar).toBeHidden();
    await openConversations(page, en);
    await sidebar.getByRole('button', { name: /Rename chat:|重命名对话：/ }).first().click();
    const name = sidebar.getByRole('textbox', { name: en ? 'Conversation name' : '对话名称' });
    await expect(name).toBeFocused();
    await name.fill('Renamed conversation');
    await name.press('Enter');
    await expect(sidebar.locator('.chat-session-select-button').first()).toContainText('Renamed conversation');
    await expect(page.locator('.chat-screen-title h1')).toHaveText('Renamed conversation');
    expect(state.renames).toBe(1);
    await page.reload();
    await openConversations(page, en);
    await expect(sidebar.locator('.chat-session-select-button').first()).toContainText('Renamed conversation');
    await sidebar.getByRole('button', { name: /Rename chat:|重命名对话：/ }).first().click();
    await name.fill('Cancelled name');
    await name.press('Escape');
    await expect(name).toHaveCount(0);
    expect(state.renames).toBe(1);
    const deleteAll = sidebar.getByRole('button', { name: en ? 'Delete all conversations' : '删除全部对话', exact: true });
    page.once('dialog', dialog => dialog.dismiss());
    await deleteAll.click();
    await expect(sidebar.locator('.chat-session-row')).toHaveCount(2);
    expect(state.deletions).toBe(0);
    page.once('dialog', dialog => dialog.accept());
    await deleteAll.click();
    await expect(sidebar.locator('.chat-session-row')).toHaveCount(0);
    await expect(sidebar.getByText(en ? 'No conversations yet' : '暂无会话')).toBeVisible();
    expect(state.deletions).toBe(2);
    await expect(deleteAll).toBeDisabled();
    await expect(sidebar.getByRole('button', { name: en ? 'New chat' : '新建会话', exact: true })).toBeEnabled();
  });

  test(`conversation list scrolls independently with a fixed header (${language})`, async ({ page }, testInfo) => {
    const en = language === 'en';
    await mockConversations(page, language, 60);
    await page.goto('/chat');
    await openConversations(page, en);
    const list = page.getByRole('list', { name: en ? 'Conversation history' : '对话记录' });
    await expect(list.getByRole('listitem')).toHaveCount(60);
    expect(await list.evaluate(element => ({ overflow: getComputedStyle(element).overflowY,
      scrollable: element.scrollHeight > element.clientHeight, height: element.clientHeight })))
      .toMatchObject({ overflow: 'auto', scrollable: true });
    const header = page.locator('.chat-session-sidebar-header');
    const before = await header.boundingBox();
    await list.focus();
    await page.keyboard.press('PageDown');
    await expect.poll(() => list.evaluate(element => element.scrollTop)).toBeGreaterThan(0);
    await list.getByRole('listitem').last().scrollIntoViewIfNeeded();
    await expect(list.getByRole('listitem').last()).toBeInViewport();
    expect((await header.boundingBox())?.y).toBe(before?.y);
    await expect(header).toBeInViewport();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.locator('.chat-view').screenshot({ path: testInfo.outputPath('conversation-list.png'), animations: 'disabled' });
  });
}
