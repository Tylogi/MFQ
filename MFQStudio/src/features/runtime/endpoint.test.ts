import { expect, it } from 'vitest';
import { openAIEndpoint } from './endpoint';

it.each([
  [null, 'http://127.0.0.1:8090/v1'],
  ['http://localhost:8090/', 'http://localhost:8090/v1'],
  ['https://example.com/v1/', 'https://example.com/v1'],
  ['https://example.com/mfq///', 'https://example.com/mfq/v1'],
  ['https://example.com/mfq/v1?x=1#settings', 'https://example.com/mfq/v1'],
  ['http://[::1]:8090', 'http://[::1]:8090/v1'],
])('produces an SDK base URL from %s without duplicate /v1', (source, expected) => {
  expect(openAIEndpoint(source)).toBe(expected);
});
