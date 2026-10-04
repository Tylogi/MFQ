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
