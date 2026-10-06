/** Verify history pagination and scroll preservation against real browser layout on desktop and mobile. */
import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

test('scrolls through both histories and keeps the reading position when logs arrive', async ({
  page,
}, testInfo) => {
  await mockStudioServer(page);
  const logs = Array.from({ length: 115 }, (_, index) => ({
    sequence: index + 1,
    created_at: '2026-10-06T00:00:00Z',
    level: 'info',
    fields: {},
    message: `History event ${index + 1}: model worker activity`,
  }));
  const requests = logs.map(({ sequence }) => ({
    sequence,
    captured_at: '2026-10-06T00:00:00Z',
    values: {
      last_request: {
        id: `history-request-${sequence}`,
        completed_at: 1791244800 + sequence,
        prompt_tokens: 12,
        completion_tokens: 30,
        decode_tps: 25,
      },
    },
  }));
  await page.route('**/api/v1/runtime/logs?*', (route) => {
    const params = new URL(route.request().url()).searchParams;
    const data = logs.filter(
      (entry) =>
        entry.sequence > Number(params.get('after') ?? 0) &&
        entry.sequence < Number(params.get('before') ?? Infinity),
    );
    data.sort((a, b) =>
      params.get('order') === 'asc' ? a.sequence - b.sequence : b.sequence - a.sequence,
    );
    return route.fulfill({ json: { data: data.slice(0, Number(params.get('limit'))) } });
  });
  await page.route('**/api/v1/runtime/requests?*', (route) => {
    const params = new URL(route.request().url()).searchParams;
    const data = requests.filter(
      (entry) =>
        entry.sequence > Number(params.get('after') ?? 0) &&
        entry.sequence < Number(params.get('before') ?? Infinity),
    );
    data.sort((a, b) =>
      params.get('order') === 'asc' ? a.sequence - b.sequence : b.sequence - a.sequence,
    );
    return route.fulfill({ json: { data: data.slice(0, Number(params.get('limit'))) } });
  });
  for (const [channel, rows] of [['logs', logs], ['requests', requests]] as const) {
    await page.route(`**/api/v1/runtime/${channel}/stream?*`, (route) => {
      const after = Number(new URL(route.request().url()).searchParams.get('after') ?? 0);
      return route.fulfill({
        contentType: 'text/event-stream',
        body: rows.filter((row) => row.sequence > after)
          .map((row) => `id: ${row.sequence}\nevent: ${channel}\ndata: ${JSON.stringify(row)}\n\n`).join('') || ': heartbeat\n\n',
      });
    });
  }
  await page.goto('/logs');
  const logPanel = page.getByRole('region', { name: 'Runtime logs' });
  const requestPanel = page.getByRole('region', { name: 'Recent requests' });
  await expect(logPanel.locator('[data-history-id]')).toHaveCount(50);
  await expect(requestPanel.locator('[data-history-id]')).toHaveCount(50);
  expect(await logPanel.evaluate((node) => node.scrollHeight > node.clientHeight)).toBe(true);

  await logPanel.evaluate((node) => {
    node.scrollTop = 200;
  });
  const anchor = await logPanel.evaluate((node) => {
    const row = Array.from(node.querySelectorAll<HTMLElement>('[data-history-id]')).find(
      (entry) => entry.getBoundingClientRect().bottom > node.getBoundingClientRect().top,
    )!;
    return {
      id: row.dataset.historyId!,
      offset: row.getBoundingClientRect().top - node.getBoundingClientRect().top,
    };
  });
  logs.push({ ...logs[0], sequence: 116, message: 'A new live event' });
  await expect(logPanel.locator('[data-history-id]')).toHaveCount(51);
  const nextOffset = await logPanel.evaluate((node, id) => {
    const row = node.querySelector<HTMLElement>(`[data-history-id="${id}"]`)!;
    return row.getBoundingClientRect().top - node.getBoundingClientRect().top;
  }, anchor.id);
  expect(Math.abs(nextOffset - anchor.offset)).toBeLessThan(2);

  for (const panel of [logPanel, requestPanel]) {
    await panel.evaluate((node) => {
      node.scrollTop = node.scrollHeight;
    });
    await expect(panel.locator('[data-history-id]')).toHaveCount(panel === logPanel ? 101 : 100);
    await panel.evaluate((node) => {
      node.scrollTop = node.scrollHeight;
    });
    await expect(panel.getByText('No older records')).toBeVisible();
    await expect(panel.locator('[data-history-id]')).toHaveCount(panel === logPanel ? 116 : 115);
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('runtime-history.png'), fullPage: true });
});
