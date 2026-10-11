import { expect, it } from 'vitest';
import { anthropicEndpoint, openAIEndpoint } from './endpoint';

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

it('keeps OpenAI /v1 while Anthropic uses the actual separate port and full request path', () => {
  expect(openAIEndpoint('http://127.0.0.1:9000/v1/')).toBe('http://127.0.0.1:9000/v1');
  expect(anthropicEndpoint('http://127.0.0.1:9000/v1/', 9001)).toBe('http://127.0.0.1:9001/v1/messages');
  expect(anthropicEndpoint('https://remote.example/mfq/v1?key=discard#fragment', 9443)).toBe('https://remote.example:9443/mfq/v1/messages');
  expect(anthropicEndpoint('http://[::1]:8090', 8091)).toBe('http://[::1]:8091/v1/messages');
});

it.each([null, undefined, 0, 65536, 1.5])('never invents an endpoint for an invalid or unavailable port: %s', (port) => {
  expect(anthropicEndpoint('', port)).toBeNull();
});

it.each([
  [null, 8091, 'http://127.0.0.1:8091/v1/messages'],
  ['http://127.0.0.1:9000/v1/', 9001, 'http://127.0.0.1:9001/v1/messages'],
  ['https://remote.example/mfq/v1?key=discard#fragment', 9443, 'https://remote.example:9443/mfq/v1/messages'],
  ['http://[::1]:8090', 8091, 'http://[::1]:8091/v1/messages'],
])('also exposes the complete Anthropic request URL for %s', (service, port, expected) => {
  expect(anthropicEndpoint(service, port)).toBe(expected);
});
