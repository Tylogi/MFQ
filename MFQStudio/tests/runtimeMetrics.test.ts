/** 验证原生预填充指标的优先级、回退和缺失值，不依赖实现字符串。 */
import { expect, it } from 'vitest';
import { displayPrefillMetric, preferPositiveMetric } from '../src/features/runtime/metrics';
import { textParts } from '../src/features/chat/messageParts';
import type { Message } from '../src/shared/api/types';

it('优先使用运行时上报的预填充耗时与速度，不把 TTFT 当作预填充', () => {
  expect(displayPrefillMetric({
    ttft_ms: 200, prefill_tokens: 40, prompt_tokens: 100,
    model_prefill_ms: 100, prefill_ms: 50, prefill_tps: 999,
    complete_prefill_ms: 2000, complete_prefill_tps: 50,
  })).toEqual({ milliseconds: 50, tokensPerSecond: 999 });
});

it('运行时指标缺失时按语言耗时、模型耗时回退，并可使用 prompt token', () => {
  expect(displayPrefillMetric({ prompt_tokens: 30, model_prefill_ms: 100, prefill_ms: 50 }))
    .toEqual({ milliseconds: 50, tokensPerSecond: 600 });
  expect(displayPrefillMetric({ prefill_tokens: 20, ttft_ms: 0, model_prefill_ms: 100 }))
    .toEqual({ milliseconds: 100, tokensPerSecond: 200 });
});

it('缺失和非正指标不生成虚假速度，只有报告速度时保留它', () => {
  expect(displayPrefillMetric()).toEqual({ milliseconds: undefined, tokensPerSecond: undefined });
  expect(displayPrefillMetric({ ttft_ms: 100 })).toEqual({ milliseconds: undefined, tokensPerSecond: undefined });
  expect(displayPrefillMetric({ ttft_ms: Infinity, prefill_tps: 12 }))
    .toEqual({ milliseconds: undefined, tokensPerSecond: 12 });
  expect(displayPrefillMetric({ prefill_tps: -1 })).toEqual({ milliseconds: undefined, tokensPerSecond: undefined });
});

it('首 token 指标使用正数主值，缺失时使用有效回退', () => {
  expect(preferPositiveMetric(100, 2000)).toBe(100);
  expect(preferPositiveMetric(0, 2000)).toBe(2000);
  expect(preferPositiveMetric(NaN, undefined)).toBeUndefined();
});

it('消息正文与 reasoning 独立合并，不把思考混入可编辑正文', () => {
  const message = { parts: [
    { type: 'reasoning', text: '思考一' },
    { type: 'text', text: '正文' },
    { type: 'reasoning', text: '思考二' },
    { type: 'transcript', text: '转录' },
  ] } as Message;
  expect(textParts(message)).toEqual({ text: '正文转录', reasoning: '思考一思考二' });
});
