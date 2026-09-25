/** 验证聊天页局部状态不影响共享会话及生成控制器。 */
import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router';
import { useChat } from '../ChatProvider';
import { useConversationSelector } from '../state/conversationStore';
import { useChatPageState } from './useChatPageState';

vi.mock('../ChatProvider', () => ({ useChat: vi.fn() }));
vi.mock('../state/conversationStore', () => ({ useConversationSelector: vi.fn() }));
vi.mock('./useChatAutoScroll', () => ({ useChatAutoScroll: () => ({
  scrollerRef: { current: null }, following: true,
  handleScroll: vi.fn(), scrollToBottom: vi.fn(),
}) }));

const selectSession = vi.fn();
const createSession = vi.fn().mockResolvedValue(undefined);
const saveEdit = vi.fn().mockResolvedValue(undefined);
const state = { activeId: 'first', sessions: [{ id: 'first', mode: 'text' }] };

beforeEach(() => {
  state.activeId = 'first';
  selectSession.mockClear();
  createSession.mockClear();
  saveEdit.mockClear();
  vi.stubGlobal('matchMedia', vi.fn((query: string) => ({
    matches: query === '(max-width: 680px)',
    addEventListener: vi.fn(), removeEventListener: vi.fn(),
  })));
  vi.mocked(useConversationSelector).mockImplementation((selector) => selector(state as never));
  vi.mocked(useChat).mockReturnValue({
    busy: false,
    conversation: { selectSession, createSession },
    inference: { setSelectedModel: vi.fn() },
    messageActions: { saveEdit },
  } as unknown as ReturnType<typeof useChat>);
});

it('移动端切换会话收起侧栏并清空旧编辑草稿', async () => {
  const { result, rerender } = renderHook(useChatPageState, {
    wrapper: ({ children }) => <MemoryRouter>{children}</MemoryRouter>,
  });
  act(() => {
    result.current.setChatSessionsOpen(true);
    result.current.setEditDraft({ messageId: 'message-1', text: 'draft' });
    result.current.selectSession('second');
  });
  expect(selectSession).toHaveBeenCalledWith('second');
  expect(result.current.chatSessionsOpen).toBe(false);
  state.activeId = 'second';
  rerender();
  await waitFor(() => expect(result.current.editDraft).toBeNull());
});
