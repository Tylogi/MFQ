/** Verify lazy session loading, history race isolation, and protection against model changes during generation. */
import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { SESSION_PAGE_SIZE, sessionsApi } from '../../../shared/api/resources/sessions';
import type { Session, Message, RuntimeInstance } from '../../../shared/api/types';
import { useConversationStore } from '../state/conversationStore';
import { useConversationSessions } from './useConversationSessions';

const runtime = vi.hoisted(() => ({
  ready: true,
  connectionRevision: 1,
  selectedModel: 'model-a',
  setSelectedModel: vi.fn(),
  models: [{ id: 'model-a' }, { id: 'model-b' }],
  instances: [] as RuntimeInstance[],
}));
vi.mock('../../../app/RuntimeProvider', () => ({ useRuntime: () => runtime }));
const first = { id: 'a', model: 'model-a', title: 'A', mode: 'text', revision: 0 } as Session;
const second = { ...first, id: 'b', title: 'B' };

beforeEach(() => {
  useConversationStore.getState().reset();
  runtime.ready = true;
  runtime.connectionRevision = 1;
  runtime.selectedModel = 'model-a';
  runtime.models = [{ id: 'model-a' }, { id: 'model-b' }];
  runtime.instances = [];
  runtime.setSelectedModel.mockClear();
  vi.spyOn(sessionsApi, 'listSessions').mockResolvedValue([first, second]);
  vi.spyOn(sessionsApi, 'listMessages').mockResolvedValue([]);
  vi.spyOn(sessionsApi, 'listResponses').mockResolvedValue([]);
  vi.spyOn(sessionsApi, 'forkSession').mockResolvedValue({ ...first, id: 'fork', model: 'model-b' });
  vi.spyOn(sessionsApi, 'deleteSession').mockResolvedValue(undefined);
});

it('loads older sessions without switching the active chat and ignores duplicate records', async () => {
  const page = Array.from({ length: SESSION_PAGE_SIZE }, (_, index) => ({ ...first, id: `session-${index}` }));
  vi.mocked(sessionsApi.listSessions).mockResolvedValueOnce(page).mockResolvedValueOnce([page[0], second]);
  const { result } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.hasMoreSessions).toBe(true));
  await act(async () => result.current.loadMoreSessions());
  expect(sessionsApi.listSessions).toHaveBeenLastCalledWith(SESSION_PAGE_SIZE);
  expect(result.current.sessions).toHaveLength(SESSION_PAGE_SIZE + 1);
  expect(result.current.activeId).toBe('session-0');
  expect(result.current.hasMoreSessions).toBe(false);
});

it('discards pending older pages after switching servers and prevents duplicate loads', async () => {
  const page = Array.from({ length: SESSION_PAGE_SIZE }, (_, index) => ({ ...first, id: `session-${index}` }));
  let resolvePage!: (sessions: Session[]) => void;
  vi.mocked(sessionsApi.listSessions)
    .mockResolvedValueOnce(page)
    .mockImplementationOnce(() => new Promise((resolve) => { resolvePage = resolve; }))
    .mockResolvedValueOnce([second]);
  const { result, rerender } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.hasMoreSessions).toBe(true));
  let loading!: Promise<void>;
  act(() => { loading = result.current.loadMoreSessions(); void result.current.loadMoreSessions(); });
  expect(sessionsApi.listSessions).toHaveBeenCalledTimes(2);
  runtime.connectionRevision += 1;
  rerender();
  await waitFor(() => expect(result.current.sessions).toEqual([second]));
  await act(async () => { resolvePage([{ ...first, id: 'stale' }]); await loading; });
  expect(result.current.sessions).toEqual([second]);
  expect(result.current.loadingSessions).toBe(false);
});

it('verifies useConversationSessions test behavior 1', async () => {
  let resolveOld!: (sessions: Session[]) => void;
  vi.mocked(sessionsApi.listSessions)
    .mockImplementationOnce(() => new Promise((resolve) => { resolveOld = resolve; }))
    .mockResolvedValueOnce([second]);
  const { result, rerender } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(sessionsApi.listSessions).toHaveBeenCalledOnce());
  runtime.connectionRevision = 2;
  rerender();
  await waitFor(() => expect(result.current.activeId).toBe('b'));
  await act(async () => resolveOld([first]));
  expect(result.current.sessions).toEqual([second]);
});

it('verifies useConversationSessions test behavior 2', async () => {
  let resolveCreate!: (session: Session) => void;
  vi.spyOn(sessionsApi, 'createSession').mockImplementationOnce(() =>
    new Promise((resolve) => { resolveCreate = resolve; }),
  );
  const { result, rerender } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  let creation!: Promise<void>;
  act(() => { creation = result.current.createSession(); });
  runtime.connectionRevision = 2;
  rerender();
  await waitFor(() => expect(sessionsApi.listSessions).toHaveBeenCalledTimes(2));
  await act(async () => { resolveCreate({ ...first, id: 'obsolete' }); await creation; });
  expect(result.current.sessions.some((session) => session.id === 'obsolete')).toBe(false);
});

it('verifies useConversationSessions test behavior 3', async () => {
  const { result, rerender } = renderHook(
    ({ enabled }) => useConversationSessions(enabled, false),
    { initialProps: { enabled: false } },
  );
  expect(sessionsApi.listSessions).not.toHaveBeenCalled();
  expect(result.current.conversationReady).toBe(false);
  rerender({ enabled: true });
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  expect(sessionsApi.listSessions).toHaveBeenCalledOnce();
});

it('verifies useConversationSessions test behavior 4', async () => {
  let resolveOld!: (messages: Message[]) => void;
  vi.mocked(sessionsApi.listMessages).mockImplementation((id) =>
    id === 'a'
      ? new Promise((resolve) => {
          resolveOld = resolve;
        })
      : Promise.resolve([
          { id: 'b-message', role: 'user', parts: [], parent_id: null, created_at: '' },
        ]),
  );
  const { result } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.activeId).toBe('a'));
  await act(async () => result.current.selectSession('b'));
  await waitFor(() => expect(result.current.messages[0]?.id).toBe('b-message'));
  await act(async () =>
    resolveOld([{ id: 'a-message', role: 'user', parts: [], parent_id: null, created_at: '' }]),
  );
  expect(result.current.activeId).toBe('b');
  expect(result.current.messages[0]?.id).toBe('b-message');
});

it('keeps the active session and its recovery state when navigation is blocked', async () => {
  const create = vi.spyOn(sessionsApi, 'createSession');
  create.mockClear();
  const { result } = renderHook(() => useConversationSessions(true, true));
  await waitFor(() => expect(result.current.activeId).toBe('a'));
  act(() => result.current.selectSession('b'));
  await act(async () => result.current.createSession());
  expect(result.current.activeId).toBe('a');
  expect(create).not.toHaveBeenCalled();
});

it('does not fork history when runtime reconciliation selects another available model', async () => {
  const { result, rerender } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  runtime.selectedModel = 'model-b';
  rerender();
  expect(sessionsApi.forkSession).not.toHaveBeenCalled();
  expect(result.current.sessions).toEqual([first, second]);
  expect(result.current.activeId).toBe('a');
});

it('forks once when the user explicitly changes the chat model', async () => {
  const { result, rerender } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  act(() => result.current.changeSessionModel('model-b'));
  runtime.selectedModel = 'model-b';
  rerender();
  await waitFor(() => expect(result.current.activeId).toBe('fork'));
  expect(sessionsApi.forkSession).toHaveBeenCalledExactlyOnceWith('a', null, true, 'A', 'model-b');
});

it('opens history for an unloaded model without duplicating the conversation', async () => {
  const historical = { ...second, model: 'unloaded-model' };
  vi.mocked(sessionsApi.listSessions).mockResolvedValue([first, historical]);
  const { result, rerender } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  act(() => result.current.selectSession('b'));
  runtime.selectedModel = 'model-b';
  rerender();
  await waitFor(() => expect(useConversationStore.getState().historyLoadedId).toBe('b'));
  expect(result.current.sessions).toEqual([first, historical]);
  expect(result.current.activeId).toBe('b');
  expect(sessionsApi.forkSession).not.toHaveBeenCalled();
});

it('verifies useConversationSessions test behavior 7', async () => {
  const { result } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  await act(async () => expect(await result.current.deleteSession('a')).toBe(true));
  expect(sessionsApi.deleteSession).toHaveBeenCalledWith('a');
  expect(result.current.sessions).toEqual([second]);
  expect(result.current.activeId).toBe('b');
  expect(runtime.setSelectedModel).toHaveBeenCalledWith(second.model);
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
});

it('verifies useConversationSessions test behavior 8', async () => {
  const { result } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  runtime.setSelectedModel.mockClear();
  await act(async () => expect(await result.current.deleteSession('b')).toBe(true));
  expect(result.current.activeId).toBe('a');
  expect(runtime.setSelectedModel).not.toHaveBeenCalled();
  await act(async () => expect(await result.current.deleteSession('a')).toBe(true));
  expect(result.current.sessions).toEqual([]);
  expect(result.current.activeId).toBeNull();
  expect(result.current.messages).toEqual([]);
});

it('verifies useConversationSessions test behavior 9', async () => {
  vi.mocked(sessionsApi.deleteSession).mockRejectedValueOnce(new Error('delete failed'));
  const { result } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  await act(async () => expect(await result.current.deleteSession('a')).toBe(false));
  expect(result.current.sessions).toEqual([first, second]);
  expect(result.current.activeId).toBe('a');
  expect(result.current.error).toContain('delete failed');
});

it('verifies useConversationSessions test behavior 10', async () => {
  let resolveDelete!: () => void;
  vi.mocked(sessionsApi.deleteSession).mockImplementationOnce(() =>
    new Promise((resolve) => { resolveDelete = resolve; }),
  );
  const { result, rerender } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  let deletion!: Promise<boolean>;
  act(() => { deletion = result.current.deleteSession('a'); });
  runtime.connectionRevision = 2;
  vi.mocked(sessionsApi.listSessions).mockResolvedValueOnce([second]);
  rerender();
  await waitFor(() => expect(result.current.activeId).toBe('b'));
  await act(async () => { resolveDelete(); expect(await deletion).toBe(false); });
  expect(result.current.sessions).toEqual([second]);
});
