/** 验证输入草稿隔离、父级渲染边界以及业务未接受时保留输入。 */
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

it('每次键入只更新输入组件，草稿按会话保存并在返回时恢复', async () => {
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

it('提交失败不会丢失输入；业务接受后才清空草稿', async () => {
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
