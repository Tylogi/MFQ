/** Verify native prefill metric priority, fallback, and missing values without relying on implementation strings. */
import { expect, it } from 'vitest';
import { displayPrefillMetric, preferPositiveMetric } from '../src/features/runtime/metrics';
import { textParts } from '../src/features/chat/messageParts';
import type { Message } from '../src/shared/api/types';

it('verifies runtimeMetrics test behavior 1', () => {
  expect(displayPrefillMetric({
    ttft_ms: 200, prefill_tokens: 40, prompt_tokens: 100,
    model_prefill_ms: 100, prefill_ms: 50, prefill_tps: 999,
    complete_prefill_ms: 2000, complete_prefill_tps: 50,
  })).toEqual({ milliseconds: 50, tokensPerSecond: 999 });
});

it('verifies runtimeMetrics test behavior 2', () => {
  expect(displayPrefillMetric({ prompt_tokens: 30, model_prefill_ms: 100, prefill_ms: 50 }))
    .toEqual({ milliseconds: 50, tokensPerSecond: 600 });
  expect(displayPrefillMetric({ prefill_tokens: 20, ttft_ms: 0, model_prefill_ms: 100 }))
    .toEqual({ milliseconds: 100, tokensPerSecond: 200 });
});

it('verifies runtimeMetrics test behavior 3', () => {
  expect(displayPrefillMetric()).toEqual({ milliseconds: undefined, tokensPerSecond: undefined });
  expect(displayPrefillMetric({ ttft_ms: 100 })).toEqual({ milliseconds: undefined, tokensPerSecond: undefined });
  expect(displayPrefillMetric({ ttft_ms: Infinity, prefill_tps: 12 }))
    .toEqual({ milliseconds: undefined, tokensPerSecond: 12 });
  expect(displayPrefillMetric({ prefill_tps: -1 })).toEqual({ milliseconds: undefined, tokensPerSecond: undefined });
});

it('verifies runtimeMetrics test behavior 4', () => {
  expect(preferPositiveMetric(100, 2000)).toBe(100);
  expect(preferPositiveMetric(0, 2000)).toBe(2000);
  expect(preferPositiveMetric(NaN, undefined)).toBeUndefined();
});

it('verifies runtimeMetrics test behavior 5', () => {
  const message = { parts: [
    { type: 'reasoning', text: 'Reasoning one' },
    { type: 'text', text: 'Body text' },
    { type: 'reasoning', text: 'Reasoning two' },
    { type: 'transcript', text: 'Transcript' },
  ] } as Message;
  expect(textParts(message)).toEqual({ text: 'Body textTranscript', reasoning: 'Reasoning oneReasoning two' });
});
