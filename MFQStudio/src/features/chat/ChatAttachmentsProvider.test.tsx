/** Verify that attachment updates notify only the input area and previews are released on session or service changes. */
import { act, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { mediaApi } from '../../shared/api/resources/media';
import {
  ChatAttachmentsProvider,
  useChatAttachmentActions,
  useChatAttachmentList,
  useChatAttachmentError,
} from './ChatAttachmentsProvider';
import { useConversationStore } from './state/conversationStore';

const runtime = vi.hoisted(() => ({ connectionRevision: 1 }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => runtime }));
vi.mock('../settings/SettingsProvider', () => ({
  useSettings: () => ({ tr: (_zh: string, en: string) => en }),
}));

beforeEach(() => {
  vi.restoreAllMocks();
  useConversationStore.getState().reset();
  useConversationStore.getState().setActiveId('session-a');
  runtime.connectionRevision = 1;
  URL.createObjectURL = vi.fn(() => 'blob:preview');
  URL.revokeObjectURL = vi.fn();
});

it('keeps eight attachments and reports overflow, including consecutive selections before rendering', () => {
  let actions!: ReturnType<typeof useChatAttachmentActions>;
  function Consumer() {
    actions = useChatAttachmentActions();
    const error = useChatAttachmentError();
    return <span>{error?.message}</span>;
  }
  render(<ChatAttachmentsProvider><Consumer /></ChatAttachmentsProvider>);
  const files = Array.from({ length: 10 }, (_, index) => new File(['text'], `${index}.txt`, { type: 'text/plain' }));
  act(() => {
    actions.selectAttachments(files.slice(0, 6) as unknown as FileList);
    actions.selectAttachments(files.slice(6) as unknown as FileList);
  });
  expect(actions.getAttachments().map((item) => item.file)).toEqual(files.slice(0, 8));
  expect(screen.getByText('Up to 8 attachments per message; 2 files were not added.')).toBeInTheDocument();
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
