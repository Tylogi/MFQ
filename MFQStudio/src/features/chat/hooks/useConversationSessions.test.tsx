/** 验证会话模块的惰性加载、历史竞态隔离及生成期间模型切换保护。 */
import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { sessionsApi } from '../../../shared/api/resources/sessions';
import type { Session, Message, RuntimeInstance } from '../../../shared/api/types';
import { useConversationStore } from '../state/conversationStore';
import { useDraftStore } from '../state/draftStore';
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
  useDraftStore.setState({ drafts: {} });
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
  vi.spyOn(sessionsApi, 'updateSession').mockImplementation(async (id, update) => ({ ...first, id, ...update } as Session));
});

it('连接版本变化后丢弃旧列表请求并加载新连接的会话', async () => {
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

it('创建会话期间连接重置不会将旧创建结果写入新列表', async () => {
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

it('未访问聊天不请求会话，访问后等待历史就绪才启用输入', async () => {
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

it('切换会话后迟到的旧历史不能覆盖当前消息', async () => {
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

it('后台生成期间不派生新模型会话，完成后再执行模型切换', async () => {
  const { result, rerender } = renderHook(({ busy }) => useConversationSessions(true, busy), {
    initialProps: { busy: true },
  });
  await waitFor(() => expect(result.current.activeId).toBe('a'));
  runtime.selectedModel = 'model-b';
  rerender({ busy: true });
  expect(sessionsApi.forkSession).not.toHaveBeenCalled();
  rerender({ busy: false });
  await waitFor(() => expect(result.current.activeId).toBe('fork'));
  expect(sessionsApi.forkSession).toHaveBeenCalledOnce();
});

it('模型只在已载入实例里，也能切换并用于会话，无需资产注册', async () => {
  runtime.models = [];
  runtime.instances = [{ id: 'loaded-b', model: 'model-b', state: 'ready' }] as RuntimeInstance[];
  runtime.selectedModel = 'model-b';
  const { result } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  expect(result.current.active?.model).toBe('model-b');
  expect(sessionsApi.forkSession).toHaveBeenCalledWith('a', null, true, 'A', 'model-b');
});

it('删除当前会话后切换到剩余会话并清除旧历史', async () => {
  const { result } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  await act(async () => expect(await result.current.deleteSession('a')).toBe(true));
  expect(sessionsApi.deleteSession).toHaveBeenCalledWith('a');
  expect(result.current.sessions).toEqual([second]);
  expect(result.current.activeId).toBe('b');
  expect(runtime.setSelectedModel).toHaveBeenCalledWith(second.model);
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
});

it('删除非当前会话不改变选中项，删除最后一条后进入空状态', async () => {
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

it('删除失败保留原会话并展示错误', async () => {
  vi.mocked(sessionsApi.deleteSession).mockRejectedValueOnce(new Error('delete failed'));
  const { result } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  await act(async () => expect(await result.current.deleteSession('a')).toBe(false));
  expect(result.current.sessions).toEqual([first, second]);
  expect(result.current.activeId).toBe('a');
  expect(result.current.error).toContain('delete failed');
});

it('删除期间连接切换不回写旧连接的结果', async () => {
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

it('删除全部会话后清除历史和对应草稿，不新建空会话', async () => {
  useDraftStore.setState({ drafts: { a: 'draft A', b: 'draft B', unrelated: 'keep' } });
  const create = vi.spyOn(sessionsApi, 'createSession');
  const { result } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  await act(async () => expect(await result.current.deleteAllSessions()).toEqual(['a', 'b']));
  expect(sessionsApi.deleteSession).toHaveBeenCalledTimes(2);
  expect(result.current.sessions).toEqual([]);
  expect(result.current.activeId).toBeNull();
  expect(result.current.messages).toEqual([]);
  expect(result.current.responses).toEqual({});
  expect(useDraftStore.getState().drafts).toEqual({ unrelated: 'keep' });
  expect(create).not.toHaveBeenCalled();
});

it('批量删除覆盖第200条之后的会话，先取完分页再删除', async () => {
  const page = Array.from({ length: 200 }, (_, index) => ({ ...first, id: `chat-${index}` }));
  const last = { ...second, id: 'older-chat' };
  vi.mocked(sessionsApi.listSessions).mockResolvedValueOnce(page).mockResolvedValueOnce(page).mockResolvedValueOnce([last]);
  const { result } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.activeId).toBe('chat-0'));
  await act(async () => expect(await result.current.deleteAllSessions()).toHaveLength(201));
  expect(sessionsApi.listSessions).toHaveBeenNthCalledWith(2, 0);
  expect(sessionsApi.listSessions).toHaveBeenNthCalledWith(3, 200);
  expect(sessionsApi.deleteSession).toHaveBeenCalledTimes(201);
  expect(sessionsApi.deleteSession).toHaveBeenLastCalledWith('older-chat');
  expect(vi.mocked(sessionsApi.listSessions).mock.invocationCallOrder[2]).toBeLessThan(vi.mocked(sessionsApi.deleteSession).mock.invocationCallOrder[0]);
  expect(result.current.sessions).toEqual([]);
});

it('批量删除部分失败只移除成功项，并保留剩余会话和草稿', async () => {
  useDraftStore.setState({ drafts: { a: 'A', b: 'B' } });
  vi.mocked(sessionsApi.deleteSession).mockResolvedValueOnce(undefined).mockRejectedValueOnce(new Error('response in progress'));
  const { result } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  await act(async () => expect(await result.current.deleteAllSessions()).toEqual(['a']));
  expect(result.current.sessions).toEqual([second]);
  expect(result.current.activeId).toBe('b');
  expect(result.current.error).toContain('response in progress');
  expect(useDraftStore.getState().drafts).toEqual({ b: 'B' });
});

it('批量读取失败时不开始删除', async () => {
  vi.mocked(sessionsApi.listSessions).mockResolvedValueOnce([first, second]).mockRejectedValueOnce(new Error('offline'));
  const { result } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  await act(async () => expect(await result.current.deleteAllSessions()).toEqual([]));
  expect(sessionsApi.deleteSession).not.toHaveBeenCalled();
  expect(result.current.sessions).toEqual([first, second]);
  expect(result.current.error).toContain('offline');
});

it('批量删除期间禁用发送，切换连接后停止后续删除且不污染新列表', async () => {
  let resolveDelete!: () => void;
  vi.mocked(sessionsApi.deleteSession).mockImplementationOnce(() => new Promise((resolve) => { resolveDelete = resolve; }));
  const { result, rerender } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  let deletion!: Promise<string[]>;
  act(() => { deletion = result.current.deleteAllSessions(); });
  await waitFor(() => expect(sessionsApi.deleteSession).toHaveBeenCalledOnce());
  expect(result.current.conversationReady).toBe(false);
  runtime.connectionRevision = 2;
  vi.mocked(sessionsApi.listSessions).mockResolvedValueOnce([second]);
  rerender();
  await waitFor(() => expect(result.current.activeId).toBe('b'));
  await act(async () => { resolveDelete(); expect(await deletion).toEqual([]); });
  expect(sessionsApi.deleteSession).toHaveBeenCalledOnce();
  expect(result.current.sessions).toEqual([second]);
  expect(result.current.transitioning).toBe(false);
});

it('重命名沿用同一会话和历史并保存服务器返回的版本', async () => {
  const message = { id: 'message-a', role: 'user', parts: [], parent_id: null, created_at: '' } as Message;
  vi.mocked(sessionsApi.listMessages).mockResolvedValue([message]);
  vi.mocked(sessionsApi.updateSession).mockResolvedValueOnce({ ...first, title: 'Renamed', revision: 1 });
  const { result } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  await act(async () => expect(await result.current.renameSession('a', '  Renamed  ')).toBe(true));
  expect(sessionsApi.updateSession).toHaveBeenCalledWith('a', { title: 'Renamed' });
  expect(result.current.active?.title).toBe('Renamed');
  expect(result.current.active?.revision).toBe(1);
  expect(result.current.activeId).toBe('a');
  expect(result.current.messages).toEqual([message]);
  expect(sessionsApi.forkSession).not.toHaveBeenCalled();
});

it('无效名称不请求，重命名失败保留原名称', async () => {
  vi.mocked(sessionsApi.updateSession).mockRejectedValueOnce(new Error('rename failed'));
  const { result } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  await act(async () => {
    expect(await result.current.renameSession('a', '  ')).toBe(false);
    expect(await result.current.renameSession('a', 'x'.repeat(513))).toBe(false);
    expect(await result.current.renameSession('a', 'A')).toBe(true);
  });
  expect(sessionsApi.updateSession).not.toHaveBeenCalled();
  await act(async () => expect(await result.current.renameSession('a', 'Renamed')).toBe(false));
  expect(result.current.active?.title).toBe('A');
  expect(result.current.error).toContain('rename failed');
});

it('生成期间不删除全部或重命名', async () => {
  const { result } = renderHook(() => useConversationSessions(true, true));
  await waitFor(() => expect(result.current.activeId).toBe('a'));
  await act(async () => {
    expect(await result.current.deleteAllSessions()).toEqual([]);
    expect(await result.current.renameSession('a', 'New')).toBe(false);
  });
  expect(sessionsApi.deleteSession).not.toHaveBeenCalled();
  expect(sessionsApi.updateSession).not.toHaveBeenCalled();
});

it('连接变化后不回写旧重命名结果', async () => {
  let resolveRename!: (session: Session) => void;
  vi.mocked(sessionsApi.updateSession).mockImplementationOnce(() => new Promise((resolve) => { resolveRename = resolve; }));
  const { result, rerender } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  let rename!: Promise<boolean>;
  act(() => { rename = result.current.renameSession('a', 'Old server'); });
  runtime.connectionRevision = 2;
  vi.mocked(sessionsApi.listSessions).mockResolvedValueOnce([second]);
  rerender();
  await waitFor(() => expect(result.current.activeId).toBe('b'));
  await act(async () => { resolveRename({ ...first, title: 'Old server' }); expect(await rename).toBe(false); });
  expect(result.current.sessions).toEqual([second]);
});
