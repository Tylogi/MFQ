/** Verify that the sidebar delete action forwards only the target session and does not switch sessions. */
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
