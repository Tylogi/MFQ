/** Verify that the sidebar delete action forwards only the target session and does not switch sessions. */
import { i18n } from '../../../i18n';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it, vi } from 'vitest';
import type { ChatPageState } from '../hooks/useChatPageState';
import { ChatSessionSidebar } from './ChatSessionSidebar';

vi.mock('../../settings/SettingsProvider', () => ({
  useSettings: () => ({ t: i18n.getFixedT('en') }),
}));
vi.mock('../state/conversationStore', () => ({
  useConversationSelector: (selector: (state: unknown) => unknown) => selector({
    sessions: [
      { id: 'a', title: 'First', model: 'model-a' },
      { id: 'b', title: 'Second', model: 'model-b' },
    ],
  }),
}));

it('verifies ChatSessionSidebar test behavior 1', async () => {
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

it('loads older conversations near the bottom without a load-more button and stops when loading or exhausted', () => {
  const loadMoreSessions = vi.fn();
  const page = {
    activeId: 'a', chatSessionsOpen: true, selectSession: vi.fn(), createSession: vi.fn(),
    chat: {
      busy: false, recoveryNeeded: false, deleteConversation: vi.fn(),
      conversation: { modelAvailable: true, hasMoreSessions: true, loadingSessions: false, loadMoreSessions },
    },
  } as unknown as ChatPageState;
  const { rerender } = render(<ChatSessionSidebar page={page} />);
  const list = screen.getByRole('region', { name: 'Conversation history' });
  Object.defineProperties(list, {
    scrollHeight: { configurable: true, value: 1500 },
    clientHeight: { configurable: true, value: 500 },
  });
  expect(screen.queryByRole('button', { name: 'Load more' })).not.toBeInTheDocument();
  fireEvent.scroll(list, { target: { scrollTop: 400 } });
  expect(loadMoreSessions).not.toHaveBeenCalled();
  fireEvent.scroll(list, { target: { scrollTop: 950 } });
  expect(loadMoreSessions).toHaveBeenCalledOnce();
  page.chat.conversation.loadingSessions = true;
  rerender(<ChatSessionSidebar page={page} />);
  expect(screen.getByRole('status')).toHaveTextContent('Loading…');
  fireEvent.scroll(list);
  expect(loadMoreSessions).toHaveBeenCalledOnce();
  page.chat.conversation.loadingSessions = false;
  page.chat.conversation.hasMoreSessions = false;
  rerender(<ChatSessionSidebar page={page} />);
  fireEvent.scroll(list);
  expect(loadMoreSessions).toHaveBeenCalledOnce();
  expect(screen.getByText('No older conversations')).toBeInTheDocument();
});
