/** Verify archive pagination reaches sessions beyond the first server page. */
import { afterEach, expect, it, vi } from 'vitest';
import type { Session } from '../types';
import { SESSION_PAGE_SIZE, sessionsApi } from './sessions';

afterEach(() => vi.restoreAllMocks());

it('exports every session page and deduplicates overlapping pages', async () => {
  const first = Array.from({ length: SESSION_PAGE_SIZE }, (_, index) => ({ id: String(index) } as Session));
  const final = { id: 'older' } as Session;
  const read = vi.spyOn(sessionsApi, 'listSessions').mockResolvedValueOnce(first).mockResolvedValueOnce([first[0], final]);
  expect(await sessionsApi.listAllSessions()).toEqual([...first, final]);
  expect(read.mock.calls).toEqual([[0], [SESSION_PAGE_SIZE]]);
});
