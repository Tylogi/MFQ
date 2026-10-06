/** Verify output budgets use protocol validity rather than an arbitrary model-independent ceiling. */
import { expect, it } from 'vitest';
import { isValidMaxTokens } from './configuration';

it.each([1, 4096, 131072])('accepts positive integer output budgets: %s', (value) => {
  expect(isValidMaxTokens(value)).toBe(true);
});

it.each([0, -1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1])('rejects invalid output budgets: %s', (value) => {
  expect(isValidMaxTokens(value)).toBe(false);
});
