import { beforeEach, expect, it, vi } from 'vitest';
import { browserUpdateStatus, cachedBrowserStatus, compareVersions, releaseCatalog, setBrowserAutomatic } from './releases';

beforeEach(() => { localStorage.clear(); });
const release = { tag_name: 'v0.3.3', name: 'Release title', body: 'Version notes', published_at: '2026-10-10T00:00:00Z', draft: false, prerelease: false };

it('only lists Releases and sorts semantic versions, not lexical strings', () => {
  const catalog = releaseCatalog([release, { ...release, tag_name: 'v0.10.0' }, { ...release, tag_name: 'v0.4.0rc1' },
    { ...release, tag_name: 'v0.4.0a1' }, { ...release, tag_name: 'v0.4.0', prerelease: true },
    { ...release, tag_name: 'v0.5.0', draft: true }, { ...release, tag_name: 'nightly' }]);
  expect(catalog.map(item => item.version)).toEqual(['0.10.0', '0.3.3']);
  expect(catalog[1].notes).toBe('Version notes');
  expect(compareVersions('0.3.3', '0.3.2+dev.12345678.modified')).toBeGreaterThan(0);
});

it('caches online releases and retains them when a forced check fails', async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => [release] });
  vi.stubGlobal('fetch', fetch);
  expect((await browserUpdateStatus(true)).releases).toHaveLength(1);
  await browserUpdateStatus(false);
  expect(fetch).toHaveBeenCalledOnce();
  fetch.mockRejectedValueOnce(new Error('offline'));
  const offline = await browserUpdateStatus(true);
  expect(offline.releases[0].version).toBe('0.3.3');
  expect(offline.error).toBe('offline');
});

it('automatic checks can be disabled without disabling manual checks', async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => [release] });
  vi.stubGlobal('fetch', fetch);
  setBrowserAutomatic(false);
  await browserUpdateStatus(false);
  expect(fetch).not.toHaveBeenCalled();
  await browserUpdateStatus(true);
  expect(fetch).toHaveBeenCalledOnce();
  expect(cachedBrowserStatus().automatic_check).toBe(false);
});

it('a corrupt cache cannot hide the running build identity', () => {
  localStorage.setItem('mfq-studio-release-cache', '{broken');
  expect(cachedBrowserStatus().current_version).toContain('0.3.2');
  localStorage.setItem('mfq-studio-release-cache', JSON.stringify({ releases: [null, {}, { tag: 'nightly' }] }));
  expect(cachedBrowserStatus().releases).toEqual([]);
});
