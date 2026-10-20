import { expect, test, type Page } from '@playwright/test';
import { mockStudioServer } from './mockServer';

async function windowsBridge(page: Page) {
  await page.addInitScript(() => {
    const state = { maximized: false, minimized: false, closed: false, theme: null as string | null };
    let nextId = 0;
    const callbacks = new Map<number, (event: unknown) => void>();
    Object.assign(window, {
      __MFQ_WINDOW_TEST__: state,
      __TAURI_EVENT_PLUGIN_INTERNALS__: { unregisterListener: () => {} },
      __TAURI_INTERNALS__: {
        metadata: { currentWindow: { label: 'main' }, currentWebview: { label: 'main' } },
        transformCallback(callback: (event: unknown) => void) { callbacks.set(++nextId, callback); return nextId; },
        async invoke(command: string, args: Record<string, unknown> = {}) {
          switch (command) {
            case 'studio_custom_titlebar': return true;
            case 'studio_status': return {
              config: { mode: 'local', remote_url: location.origin, local_service_port: 5187 },
              service_url: location.origin, reachable: true, managed_pid: null,
            };
            case 'studio_credential_get': return '';
            case 'studio_update_status': return {
              current_version: '0.4.0', current_release: false, platform_supported: false,
              automatic_check: false, automatic_download: false, update_available: false,
              releases_page: 'https://github.com/Tylogi/TyloQuant/releases', releases: [], installed_versions: [],
            };
            case 'plugin:window|is_maximized': return state.maximized;
            case 'plugin:window|is_focused': return true;
            case 'plugin:window|minimize': state.minimized = true; return;
            case 'plugin:window|toggle_maximize': state.maximized = !state.maximized; return;
            case 'plugin:window|close': state.closed = true; return;
            case 'plugin:window|set_theme': state.theme = args.value as string | null; return;
            case 'plugin:event|listen': return ++nextId;
            case 'plugin:event|unlisten': return;
            default: throw new Error(`Unexpected desktop command: ${command}`);
          }
        },
      },
    });
  });
}

for (const theme of ['light', 'dark'] as const) {
  test(`Windows frame follows the ${theme} panel and leaves the workspace within the viewport`, async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await mockStudioServer(page, { theme });
    await windowsBridge(page);
    await page.goto('/');
    const titlebar = page.getByRole('banner', { name: 'Window title bar' });
    await expect(titlebar).toBeVisible();
    await expect(page.locator('.app-shell')).toBeVisible();
    await expect(titlebar).toHaveCSS('background-color', theme === 'dark' ? 'rgb(28, 29, 33)' : 'rgb(238, 240, 243)');
    await expect(page.locator('.app-shell')).toHaveCSS('height', '864px');
    await expect.poll(() => page.evaluate(() => Reflect.get(window, '__MFQ_WINDOW_TEST__').theme)).toBe(theme);
    await page.screenshot({ path: `artifacts/windows-frame-${theme}.png` });
    await page.getByRole('button', { name: 'Maximize', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Restore window' })).toBeVisible();
    await page.getByRole('button', { name: 'Restore window' }).click();
    await expect(page.getByRole('button', { name: 'Maximize', exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Minimize', exact: true }).click();
    expect(await page.evaluate(() => Reflect.get(window, '__MFQ_WINDOW_TEST__').minimized)).toBe(true);
    await page.getByRole('button', { name: 'Close window' }).click();
    expect(await page.evaluate(() => Reflect.get(window, '__MFQ_WINDOW_TEST__').closed)).toBe(true);
  });
}

test('browser windows keep the original layout without desktop controls', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await mockStudioServer(page);
  await page.goto('/');
  await expect(page.locator('.app-shell')).toBeVisible();
  await expect(page.locator('.desktop-titlebar')).toHaveCount(0);
  await expect(page.locator('.app-shell')).toHaveCSS('height', '900px');
});
