/** Verify session-store reset, stale-request isolation, and synchronous-update boundaries. */
import { beforeEach, expect, it } from 'vitest';
import type { Message, ResponseResource, Session } from '../../../shared/api/types';
import { useConversationStore } from './conversationStore';

const first = { id: 'a', model: 'model-a', title: 'A', mode: 'text', revision: 0 } as Session;
const second = { ...first, id: 'b', title: 'B' };
const message = {
  id: 'reply',
  role: 'assistant',
  parts: [],
  parent_id: null,
  created_at: '',
} as Message;
const response = { output_message_id: 'reply', id: 'response' } as ResponseResource;

beforeEach(() => useConversationStore.getState().reset());

it('moves an older conversation ahead after activity is synchronized without changing the active chat', () => {
  const store = useConversationStore.getState();
  const recent = { ...first, updated_at: '2026-10-06T10:00:00Z' };
  const older = { ...second, updated_at: '2026-10-05T10:00:00Z' };
  store.loadSessions(store.epoch, [recent, older]);
  store.setActiveId(older.id);
  store.applySynchronized({ ...older, updated_at: '2026-10-06T11:00:00Z' }, [message], [response]);
  expect(useConversationStore.getState().sessions.map((session) => session.id)).toEqual(['b', 'a']);
  expect(useConversationStore.getState().activeId).toBe('b');
  expect(useConversationStore.getState().messages).toEqual([message]);
});

it('preserves server page order without mutating input or promoting read-only selection', () => {
  const store = useConversationStore.getState();
  const older = { ...first, updated_at: '2026-10-05T10:00:00Z' };
  const recent = { ...second, updated_at: '2026-10-06T10:00:00Z' };
  const input = [recent, older];
  store.setSessions(input);
  expect(input).toEqual([recent, older]);
  store.setActiveId(older.id);
  expect(useConversationStore.getState().sessions).toEqual([recent, older]);
  store.setSessions((sessions) => [...sessions, { ...older, id: 'c', updated_at: '2026-10-04T10:00:00Z' }]);
  expect(useConversationStore.getState().sessions.map((session) => session.id)).toEqual(['b', 'a', 'c']);
});

it('uses the server ID tie-breaker and keeps unloaded sessions out of the loaded page', () => {
  const store = useConversationStore.getState();
  const recent = { ...first, updated_at: '2026-10-06T10:00:00Z' };
  const older = { ...second, updated_at: '2026-10-05T10:00:00Z' };
  const input = [recent, older];
  store.setSessions(input);
  store.applySynchronized({ ...older, updated_at: recent.updated_at }, [], []);
  expect(useConversationStore.getState().sessions.map((session) => session.id)).toEqual(['b', 'a']);
  expect(input).toEqual([recent, older]);
  store.applySynchronized({ ...recent, id: 'unloaded' }, [], []);
  expect(useConversationStore.getState().sessions).toHaveLength(2);
});

it('verifies conversationStore test behavior 1', () => {
  const store = useConversationStore.getState();
  store.loadSessions(store.epoch, [first, second]);
  store.applyHistory(store.epoch, first.id, [message], [response]);
  expect(useConversationStore.getState().historyLoadedId).toBe(first.id);
  store.setActiveId(second.id);
  expect(useConversationStore.getState()).toMatchObject({
    activeId: second.id,
    messages: [],
    responses: {},
    historyLoadedId: null,
  });
});

it('verifies conversationStore test behavior 2', () => {
  const store = useConversationStore.getState();
  const epoch = store.epoch;
  expect(store.loadSessions(epoch, [first])).toBe(true);
  store.applyHistory(epoch, 'a', [message], [response]);
  expect(useConversationStore.getState().historyLoadedId).toBe('a');
  store.reset();
  expect(store.loadSessions(epoch, [second])).toBe(false);
  store.applyHistory(epoch, 'a', [message], [response]);
  expect(useConversationStore.getState()).toMatchObject({
    sessions: [],
    activeId: null,
    messages: [],
    responses: {},
    historyLoadedId: null,
  });
});

it('verifies conversationStore test behavior 3', () => {
  const store = useConversationStore.getState();
  const epoch = store.epoch;
  store.loadSessions(epoch, [first, second]);
  store.setActiveId('b');
  store.applyHistory(epoch, 'a', [message], [response]);
  expect(useConversationStore.getState().messages).toEqual([]);
  store.applyHistory(epoch, 'b', [message], [response]);
  expect(useConversationStore.getState().responses.reply).toBe(response);
  expect(useConversationStore.getState().historyLoadedId).toBe('b');
});

it('verifies conversationStore test behavior 4', () => {
  const store = useConversationStore.getState();
  store.loadSessions(store.epoch, [first, second]);
  store.applySynchronized({ ...second, revision: 2 }, [message], [response]);
  expect(useConversationStore.getState().sessions[1].revision).toBe(2);
  expect(useConversationStore.getState().messages).toEqual([]);
  store.applySynchronized({ ...first, revision: 3 }, [message], [response]);
  expect(useConversationStore.getState()).toMatchObject({
    messages: [message],
    responses: { reply: response },
    historyLoadedId: null,
  });
});
