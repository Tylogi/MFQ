/** Verify SDK endpoints follow explicit services and the current browser origin. */
import { afterEach, expect, it, vi } from 'vitest';
import { openAIEndpoint } from './endpoint';
import { setApiBaseUrl } from '../../shared/api/client';

afterEach(() => {
  setApiBaseUrl('');
  vi.unstubAllGlobals();
});

it.each([
  [null, `${window.location.origin}/v1`],
  ['http://localhost:8090/', 'http://localhost:8090/v1'],
  ['https://example.com/v1/', 'https://example.com/v1'],
  ['https://example.com/mfq///', 'https://example.com/mfq/v1'],
  ['https://example.com/mfq/v1?x=1#settings', 'https://example.com/mfq/v1'],
  ['http://[::1]:8090', 'http://[::1]:8090/v1'],
])('produces an SDK base URL from %s without duplicate /v1', (source, expected) => {
  expect(openAIEndpoint(source)).toBe(expected);
});

it('uses the browser origin for same-origin remote deployments', () => {
  vi.stubGlobal('window', { location: { origin: 'https://studio.example:9443' } });
  expect(openAIEndpoint('')).toBe('https://studio.example:9443/v1');
});

it('uses the active explicit service when no URL is supplied', () => {
  setApiBaseUrl('https://service.example/mfq');
  expect(openAIEndpoint()).toBe('https://service.example/mfq/v1');
});
