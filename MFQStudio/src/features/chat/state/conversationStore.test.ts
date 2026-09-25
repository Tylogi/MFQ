/** 验证会话 store 的重置、过期请求隔离与同步更新边界。 */
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

it('替换活动会话时立即清空历史、响应和就绪标记', () => {
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

it('重置所有会话数据并拒绝旧连接的列表与历史回写', () => {
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

it('切换会话后拒绝迟到的旧历史，即使连接版本没有变化', () => {
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

it('同步更新仅替换目标会话，并只为当前会话写入消息与响应', () => {
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
