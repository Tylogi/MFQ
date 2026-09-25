/** 验证消息编辑的提交时机、重新生成的回退目标及失败后的历史保护。 */
import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { sessionsApi } from '../../../shared/api/resources/sessions';
import type { Message, Session } from '../../../shared/api/types';
import { useMessageActions } from './useMessageActions';
import type { useConversationSessions } from './useConversationSessions';

const session = { id: 'session-1', model: 'model-a', revision: 1 } as Session;
const message = {
  id: 'message-1',
  role: 'user',
  parts: [{ type: 'text', text: 'original' }],
} as Message;

afterEach(() => vi.restoreAllMocks());

it('编辑保留媒体和文档，截断后续历史并使用回退版本继续生成', async () => {
  const media = { type: 'image', media: { id: 'image-1' } };
  const document = { type: 'document', media: { id: 'document-1' }, name: 'notes.txt' };
  const edited = { ...message, parts: [...message.parts, media, document] } as Message;
  const previous = { ...message, id: 'previous' };
  const later = { ...message, id: 'later' };
  const rewound = { ...session, revision: 9 };
  vi.spyOn(sessionsApi, 'rewindSession').mockResolvedValue(rewound);
  const generate = vi.fn().mockResolvedValue(undefined);
  const setMessages = vi.fn();
  const setSessions = vi.fn();
  const setResponses = vi.fn();
  const conversation = { active: session, activeIdRef: { current: session.id }, messages: [previous, edited, later], setMessages, setSessions, setResponses, setError: vi.fn() } as unknown as ReturnType<typeof useConversationSessions>;
  const { result } = renderHook(() => useMessageActions({ conversation, blocked: false, generate, setBusy: vi.fn() }));
  await act(async () => expect(await result.current.saveEdit(edited, '  revised  ', vi.fn())).toBe(true));
  const parts = [{ type: 'text', text: 'revised' }, media, document];
  expect(sessionsApi.rewindSession).toHaveBeenCalledExactlyOnceWith(session.id, session.revision, edited.id, false);
  expect(setMessages).toHaveBeenCalledExactlyOnceWith([previous, { ...edited, parts }]);
  expect(setSessions.mock.calls[0][0]([session])).toEqual([rewound]);
  expect(setResponses.mock.calls[0][0]({ previous: 'keep', later: 'drop' })).toEqual({ previous: 'keep' });
  expect(generate).toHaveBeenCalledExactlyOnceWith(rewound, parts, false);
});

it('重新生成回退到目标回答前的用户消息，保留附件并使用新会话版本', async () => {
  const user = {
    ...message,
    parts: [...message.parts, { type: 'image', media: { id: 'image-1' } }],
  } as Message;
  const assistant = { ...message, id: 'assistant-1', role: 'assistant' } as Message;
  const laterUser = { ...message, id: 'later-user' };
  const rewound = { ...session, revision: 2 };
  vi.spyOn(sessionsApi, 'rewindSession').mockResolvedValue(rewound);
  const setMessages = vi.fn();
  const generate = vi.fn().mockResolvedValue(undefined);
  const conversation = {
    active: session,
    activeIdRef: { current: session.id },
    messages: [user, assistant, laterUser],
    setMessages,
    setSessions: vi.fn(),
    setResponses: vi.fn(),
    setError: vi.fn(),
  } as unknown as ReturnType<typeof useConversationSessions>;
  const { result } = renderHook(() => useMessageActions({
    conversation, blocked: false, generate, setBusy: vi.fn(),
  }));
  await act(async () => result.current.regenerate(assistant));
  expect(sessionsApi.rewindSession).toHaveBeenCalledExactlyOnceWith(
    session.id, session.revision, user.id, false,
  );
  expect(setMessages).toHaveBeenCalledExactlyOnceWith([user]);
  expect(generate).toHaveBeenCalledExactlyOnceWith(rewound, user.parts, false);
});

it('重新生成回退失败时不修改历史也不发起生成', async () => {
  vi.spyOn(sessionsApi, 'rewindSession').mockRejectedValue(new Error('revision conflict'));
  const assistant = { ...message, id: 'assistant-1', role: 'assistant' } as Message;
  const setMessages = vi.fn();
  const setError = vi.fn();
  const generate = vi.fn();
  const conversation = {
    active: session,
    activeIdRef: { current: session.id },
    messages: [message, assistant],
    setMessages,
    setSessions: vi.fn(),
    setResponses: vi.fn(),
    setError,
  } as unknown as ReturnType<typeof useConversationSessions>;
  const { result } = renderHook(() => useMessageActions({
    conversation, blocked: false, generate, setBusy: vi.fn(),
  }));
  await act(async () => result.current.regenerate(assistant));
  expect(setMessages).not.toHaveBeenCalled();
  expect(generate).not.toHaveBeenCalled();
  expect(setError).toHaveBeenCalledWith('revision conflict');
});

it('回退成功时立即结束编辑，不等待生成流完成', async () => {
  vi.spyOn(sessionsApi, 'rewindSession').mockResolvedValue({ ...session, revision: 2 });
  const onCommitted = vi.fn();
  let finishGeneration!: () => void;
  const generate = vi.fn(
    () =>
      new Promise<void>((resolve) => {
        finishGeneration = resolve;
      }),
  );
  const conversation = {
    active: session,
    activeIdRef: { current: session.id },
    messages: [message],
    setMessages: vi.fn(),
    setSessions: vi.fn(),
    setResponses: vi.fn(),
    setError: vi.fn(),
  } as unknown as ReturnType<typeof useConversationSessions>;
  const { result } = renderHook(() =>
    useMessageActions({
      conversation,
      blocked: false,
      generate,
      setBusy: vi.fn(),
    }),
  );

  let pending!: Promise<boolean>;
  act(() => {
    pending = result.current.saveEdit(message, 'revised', onCommitted);
  });
  await waitFor(() => expect(onCommitted).toHaveBeenCalledOnce());
  expect(generate).toHaveBeenCalledOnce();
  await act(async () => {
    finishGeneration();
    expect(await pending).toBe(true);
  });
});

it('回退失败时不清除编辑草稿', async () => {
  vi.spyOn(sessionsApi, 'rewindSession').mockRejectedValue(new Error('rewind failed'));
  const onCommitted = vi.fn();
  const setError = vi.fn();
  const conversation = {
    active: session,
    activeIdRef: { current: session.id },
    messages: [message],
    setMessages: vi.fn(),
    setSessions: vi.fn(),
    setResponses: vi.fn(),
    setError,
  } as unknown as ReturnType<typeof useConversationSessions>;
  const { result } = renderHook(() =>
    useMessageActions({
      conversation,
      blocked: false,
      generate: vi.fn(),
      setBusy: vi.fn(),
    }),
  );

  await act(async () => {
    expect(await result.current.saveEdit(message, 'revised', onCommitted)).toBe(false);
  });
  expect(onCommitted).not.toHaveBeenCalled();
  expect(setError).toHaveBeenCalledWith('rewind failed');
});
