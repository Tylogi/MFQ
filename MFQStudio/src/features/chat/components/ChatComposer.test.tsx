/** Verify input-draft isolation, parent render boundaries, and input preservation when the business action does not accept it. */
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, expect, it, vi } from 'vitest';
import { ChatComposer } from './ChatComposer';
import { useDraftStore } from '../state/draftStore';

vi.mock('../ChatAttachmentsProvider', () => ({
  useChatAttachmentActions: () => ({ selectAttachments: vi.fn(), removeAttachment: vi.fn() }),
  useChatAttachmentList: () => [],
}));

const props = {
  sessionId: 'session-a',
  ready: true,
  busy: false,
  recoveryNeeded: false,
  phase: 'idle' as const,
  placeholder: '',
  attachmentAccept: '',
  toolbar: null,
  tr: (_zh: string, en: string) => en,
  onSend: vi.fn(async (_text: string, _accepted: () => void) => undefined),
  onStop: vi.fn(async () => undefined),
  onError: vi.fn(),
};

beforeEach(() => {
  useDraftStore.setState({ drafts: {} });
  vi.clearAllMocks();
});

it('verifies ChatComposer test behavior 1', async () => {
  const user = userEvent.setup();
  let renders = 0;
  function Host({ sessionId }: { sessionId: string }) {
    renders += 1;
    return <ChatComposer {...props} sessionId={sessionId} />;
  }
  const view = render(<Host sessionId="session-a" />);
  await user.type(screen.getByRole('textbox'), 'first draft');
  expect(renders).toBe(1);
  view.rerender(<Host sessionId="session-b" />);
  expect(screen.getByRole('textbox')).toHaveValue('');
  await user.type(screen.getByRole('textbox'), 'second draft');
  view.rerender(<Host sessionId="session-a" />);
  expect(screen.getByRole('textbox')).toHaveValue('first draft');
});

it('verifies ChatComposer test behavior 2', async () => {
  const user = userEvent.setup();
  const failure = new Error('upload failed');
  const send = vi
    .fn()
    .mockRejectedValueOnce(failure)
    .mockImplementationOnce(async (_text, accepted) => accepted());
  render(<ChatComposer {...props} onSend={send} />);
  await user.type(screen.getByRole('textbox'), 'keep me');
  await user.click(screen.getByRole('button', { name: 'Send' }));
  expect(screen.getByRole('textbox')).toHaveValue('keep me');
  expect(props.onError).toHaveBeenCalledWith(failure);
  await user.click(screen.getByRole('button', { name: 'Send' }));
  expect(screen.getByRole('textbox')).toHaveValue('');
});
