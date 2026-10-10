/** 验证侧栏删除入口只转发目标会话，不触发会话切换。 */
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it, vi } from 'vitest';
import type { ChatPageState } from '../hooks/useChatPageState';
import { ChatSessionSidebar } from './ChatSessionSidebar';

vi.mock('../../settings/SettingsProvider', () => ({
  useSettings: () => ({ tr: (_zh: string, en: string) => en }),
}));
vi.mock('../state/conversationStore', () => ({
  useConversationSelector: (selector: (state: unknown) => unknown) => selector({
    sessions: [
      { id: 'a', title: 'First', model: 'model-a' },
      { id: 'b', title: 'Second', model: 'model-b' },
    ],
  }),
}));

it('删除按钮独立于选中按钮，并在忙碌时禁用', async () => {
  const deleteConversation = vi.fn().mockResolvedValue(undefined);
  const selectSession = vi.fn();
  const page = {
    activeId: 'a',
    chatSessionsOpen: true,
    selectSession,
    createSession: vi.fn(),
    chat: {
      busy: false,
      deleteConversation,
      conversation: { transitioning: false, modelAvailable: true },
    },
  } as unknown as ChatPageState;
  const { rerender } = render(<ChatSessionSidebar page={page} />);
  await userEvent.click(screen.getByRole('button', { name: 'Delete chat: Second' }));
  expect(deleteConversation).toHaveBeenCalledWith('b');
  expect(selectSession).not.toHaveBeenCalled();
  page.chat.busy = true;
  rerender(<ChatSessionSidebar page={page} />);
  expect(screen.getByRole('button', { name: 'Delete chat: Second' })).toBeDisabled();
});

function fixture() {
  return {
    activeId: 'a', chatSessionsOpen: true, setChatSessionsOpen: vi.fn(),
    selectSession: vi.fn(), createSession: vi.fn(),
    chat: { busy: false, deleteConversation: vi.fn(), deleteAllConversations: vi.fn(),
      conversation: { transitioning: false, modelAvailable: true, renameSession: vi.fn().mockResolvedValue(true) } },
  } as unknown as ChatPageState;
}

it('三角标收起列表，全部删除按钮不触发会话切换', async () => {
  const page = fixture();
  render(<ChatSessionSidebar page={page} />);
  const disclosure = screen.getByRole('button', { name: 'Collapse conversations' });
  expect(disclosure).toHaveAttribute('aria-expanded', 'true');
  expect(disclosure.querySelector('svg')).not.toBeNull();
  await userEvent.click(disclosure);
  expect(page.setChatSessionsOpen).toHaveBeenCalledWith(false);
  await userEvent.click(screen.getByRole('button', { name: 'Delete all conversations' }));
  expect(page.chat.deleteAllConversations).toHaveBeenCalledOnce();
  expect(page.selectSession).not.toHaveBeenCalled();
});

it('重命名就地编辑，回车保存，不触发切换或删除', async () => {
  const page = fixture();
  render(<ChatSessionSidebar page={page} />);
  await userEvent.click(screen.getByRole('button', { name: 'Rename chat: Second' }));
  const input = screen.getByRole('textbox', { name: 'Conversation name' });
  expect(input).toHaveFocus();
  expect(input).toHaveValue('Second');
  await userEvent.clear(input);
  await userEvent.type(input, 'New name{Enter}');
  expect(page.chat.conversation.renameSession).toHaveBeenCalledWith('b', 'New name');
  expect(screen.queryByRole('textbox', { name: 'Conversation name' })).not.toBeInTheDocument();
  expect(page.selectSession).not.toHaveBeenCalled();
  expect(page.chat.deleteConversation).not.toHaveBeenCalled();
});

it('空名称不能保存，Esc取消，不发送更新', async () => {
  const page = fixture();
  render(<ChatSessionSidebar page={page} />);
  await userEvent.click(screen.getByRole('button', { name: 'Rename chat: First' }));
  await userEvent.clear(screen.getByRole('textbox', { name: 'Conversation name' }));
  expect(screen.getByRole('button', { name: 'Save name' })).toBeDisabled();
  await userEvent.keyboard('{Escape}');
  expect(screen.queryByRole('textbox', { name: 'Conversation name' })).not.toBeInTheDocument();
  expect(page.chat.conversation.renameSession).not.toHaveBeenCalled();
});

it('保存失败保留编辑内容，忙碌时所有写操作禁用', async () => {
  const page = fixture();
  vi.mocked(page.chat.conversation.renameSession).mockResolvedValueOnce(false);
  const { rerender } = render(<ChatSessionSidebar page={page} />);
  await userEvent.click(screen.getByRole('button', { name: 'Rename chat: First' }));
  await userEvent.type(screen.getByRole('textbox', { name: 'Conversation name' }), 'Retry{Enter}');
  expect(screen.getByRole('textbox', { name: 'Conversation name' })).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Cancel rename' }));
  page.chat.busy = true;
  rerender(<ChatSessionSidebar page={page} />);
  expect(screen.getByRole('button', { name: 'Delete all conversations' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Rename chat: First' })).toBeDisabled();
});
