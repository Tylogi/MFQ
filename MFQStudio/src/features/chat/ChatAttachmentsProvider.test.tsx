/** Verify that attachment updates notify only the input area and previews are released on session or service changes. */
import { act, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { mediaApi } from '../../shared/api/resources/media';
import {
  ChatAttachmentsProvider,
  useChatAttachmentActions,
  useChatAttachmentList,
} from './ChatAttachmentsProvider';
import { useConversationStore } from './state/conversationStore';

const runtime = vi.hoisted(() => ({ connectionRevision: 1 }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => runtime }));

beforeEach(() => {
  vi.restoreAllMocks();
  useConversationStore.getState().reset();
  useConversationStore.getState().setActiveId('session-a');
  runtime.connectionRevision = 1;
  URL.createObjectURL = vi.fn(() => 'blob:preview');
  URL.revokeObjectURL = vi.fn();
});

it('verifies ChatAttachmentsProvider test behavior 1', async () => {
  vi.spyOn(mediaApi, 'uploadMedia').mockRejectedValueOnce(new Error('upload failed'));
  let actions!: ReturnType<typeof useChatAttachmentActions>;
  function Consumer() {
    actions = useChatAttachmentActions();
    const attachments = useChatAttachmentList();
    return <span>{attachments.length}</span>;
  }
  render(
    <ChatAttachmentsProvider>
      <Consumer />
    </ChatAttachmentsProvider>,
  );
  const file = new File(['text'], 'notes.txt', { type: 'text/plain' });
  act(() => actions.selectAttachments([file] as unknown as FileList));
  await act(async () => {
    await expect(actions.uploadAttachments()).rejects.toThrow('upload failed');
  });
  expect(actions.getAttachments()[0].file).toBe(file);
  expect(screen.getByText('1')).toBeInTheDocument();
});

it('verifies ChatAttachmentsProvider test behavior 2', () => {
  let actionRenders = 0;
  let actions!: ReturnType<typeof useChatAttachmentActions>;
  function Actions() {
    actions = useChatAttachmentActions();
    actionRenders += 1;
    return null;
  }
  function Tray() {
    const attachments = useChatAttachmentList();
    return <span>{attachments.length}</span>;
  }
  const view = render(
    <ChatAttachmentsProvider>
      <Actions />
      <Tray />
    </ChatAttachmentsProvider>,
  );
  const image = new File(['image'], 'example.png', { type: 'image/png' });
  act(() => actions.selectAttachments([image] as unknown as FileList));
  expect(screen.getByText('1')).toBeInTheDocument();
  expect(actionRenders).toBe(1);
  expect(actions.getAttachments()[0].file).toBe(image);

  act(() => useConversationStore.getState().setActiveId('session-b'));
  expect(screen.getByText('0')).toBeInTheDocument();
  expect(URL.revokeObjectURL).toHaveBeenCalledTimes(1);

  act(() => actions.selectAttachments([image] as unknown as FileList));
  runtime.connectionRevision = 2;
  view.rerender(
    <ChatAttachmentsProvider>
      <Actions />
      <Tray />
    </ChatAttachmentsProvider>,
  );
  expect(screen.getByText('0')).toBeInTheDocument();
  expect(URL.revokeObjectURL).toHaveBeenCalledTimes(2);
});
