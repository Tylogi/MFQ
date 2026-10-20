import { expect, it } from 'vitest';
import { ApiError } from '../shared/api/client';
import { errorMessage } from './formatters';

const apiError = (code: string, message = '[Errno 48] address already in use', details: Record<string, unknown> = {}) => new ApiError(409, { error: { code, message, details, retryable: false } });

it.each(['listener_port_conflict', 'listener_port_in_use', 'invalid_download_directory', 'invalid_model_alias', 'model_alias_conflict', 'resident_budget_exhausted', 'resident_budget_unsupported', 'authentication_required', 'qsa_kv_configuration_invalid'])('localizes expected input errors without exposing %s or system diagnostics', code => {
  const error = apiError(code);
  for (const tr of [(zh: string, _en: string) => zh, (_zh: string, en: string) => en]) {
    const message = errorMessage(error, tr);
    expect(message).not.toContain(code);
    expect(message).not.toContain('Errno');
    expect(message.length).toBeGreaterThan(10);
  }
});

it('recognizes the legacy port collision and explains port range errors', () => {
  const tr = (_zh: string, en: string) => en;
  expect(errorMessage(apiError('listener_change_failed'), tr)).toBe('Port is already in use. Choose another port.');
  expect(errorMessage(apiError('invalid_request', 'request validation failed', { errors: [{ loc: ['body', 'port'] }] }), tr)).toBe('Port must be an integer between 1 and 65535.');
});

it('preserves diagnostic codes for unexpected failures', () => {
  expect(errorMessage(apiError('unexpected_kernel_error', 'allocation failed'), (_zh, en) => en)).toBe('unexpected_kernel_error: allocation failed');
  expect(errorMessage(new Error('unexpected problem'))).toBe('unexpected problem');
});

it('preserves native desktop string and message-object rejections', () => {
  const message = 'The Anthropic API port 8091 is already in use.';
  expect(errorMessage(message)).toBe(message);
  expect(errorMessage({ message })).toBe(message);
  for (const value of [null, undefined, '', '  ', {}, { message: 1 }]) {
    expect(errorMessage(value)).toBe('Unknown error');
  }
});

it('explains automatic KV capacity failure with the requested SSD alternative', () => {
  expect(errorMessage(apiError('kv_memory_insufficient'), (zh) => zh)).toBe(
    '可用显存和内存不足，请选择自动（启用SSD KV），这可能会拖慢长上下文时的速度并造成磁盘占用，或者降低上下文长度。');
});
