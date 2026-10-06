/** Verify message-edit commit timing, regeneration rewind targets, and history preservation after failures. */
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

it('verifies useMessageActions test behavior 1', async () => {
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

it('verifies useMessageActions test behavior 2', async () => {
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

it('verifies useMessageActions test behavior 3', async () => {
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

it('verifies useMessageActions test behavior 4', async () => {
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

it('verifies useMessageActions test behavior 5', async () => {
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
