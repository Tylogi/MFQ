/** 将原 Python 会话检查迁为真实 Provider、附件转换和生成控制器的行为测试。 */
import { act, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { MemoryRouter } from 'react-router';
import { beforeEach, expect, it, vi } from 'vitest';
import { ChatProvider, useChat } from '../src/features/chat/ChatProvider';
import { useChatAttachmentActions } from '../src/features/chat/ChatAttachmentsProvider';
import { useConversationStore } from '../src/features/chat/state/conversationStore';
import { GenerationController } from '../src/features/chat/state/generationController';
import { DEFAULT_SETTINGS } from '../src/features/settings/configuration';
import { sessionsApi } from '../src/shared/api/resources/sessions';
import { connectionsApi } from '../src/shared/api/resources/connections';
import { mediaApi } from '../src/shared/api/resources/media';
import type { Message, Session, StreamRequest, ResponseResource } from '../src/shared/api/types';

const mocks = vi.hoisted(() => ({
  runtime: { ready: true, connectionRevision: 1, selectedModel: 'model-a', models: [{ id: 'model-a' }], instances: [], runtime: null, realtime: null, capabilities: null, setSelectedModel: vi.fn(), refreshRuntime: vi.fn() },
  start: vi.fn(), reset: vi.fn(), prompt: '', excludeReasoning: false,
}));
vi.mock('../src/app/RuntimeProvider', () => ({ useRuntime: () => mocks.runtime }));
vi.mock('../src/features/settings/SettingsProvider', () => ({
  useSettings: () => ({ settings: { ...DEFAULT_SETTINGS, inheritModelDefaults: false, systemPrompt: mocks.prompt, excludeReasoning: mocks.excludeReasoning }, tr: (_zh: string, en: string) => en }),
}));
vi.mock('../src/features/chat/hooks/useChatGeneration', () => {
  const controller = { start: mocks.start, reset: mocks.reset, getPhase: () => 'idle', getSnapshot: () => ({ recoveryNeeded: false }) };
  return { useChatGeneration: () => ({ controller, phase: 'idle' }) };
});
vi.mock('../src/studio', () => ({ studioConfirm: vi.fn().mockResolvedValue(true) }));

const session = { id: 'session-a', model: 'model-a', mode: 'text', revision: 7 } as Session;
const previous = { id: 'previous', role: 'assistant', parts: [{ type: 'text', text: 'old answer' }], parent_id: null, created_at: '' } as Message;

/** 控制异步边界，验证完成前的状态与调用顺序。 */
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

/** 挂载真实聊天领域并等待历史就绪。 */
async function mountChat() {
  const wrapper = ({ children }: { children: ReactNode }) => <MemoryRouter initialEntries={['/chat']}><ChatProvider>{children}</ChatProvider></MemoryRouter>;
  const hook = renderHook(() => ({ chat: useChat(), attachments: useChatAttachmentActions() }), { wrapper });
  await waitFor(() => expect(hook.result.current.chat.conversation.conversationReady).toBe(true));
  return hook;
}

beforeEach(() => {
  vi.clearAllMocks();
  mocks.prompt = '';
  mocks.excludeReasoning = false;
  mocks.runtime.ready = true;
  mocks.start.mockResolvedValue(undefined);
  useConversationStore.getState().reset();
  vi.spyOn(sessionsApi, 'listSessions').mockResolvedValue([session]);
  vi.spyOn(sessionsApi, 'listMessages').mockResolvedValue([previous]);
  vi.spyOn(sessionsApi, 'listResponses').mockResolvedValue([]);
  vi.spyOn(connectionsApi, 'mcpTools').mockResolvedValue({ data: [], errors: {} });
});

it('运行时未就绪时不加载会话或发送，就绪并加载历史后才开放输入', async () => {
  mocks.runtime.ready = false;
  const wrapper = ({ children }: { children: ReactNode }) => <MemoryRouter initialEntries={['/chat']}><ChatProvider>{children}</ChatProvider></MemoryRouter>;
  const { result, rerender } = renderHook(() => useChat(), { wrapper });
  expect(sessionsApi.listSessions).not.toHaveBeenCalled();
  expect(result.current.conversation.conversationReady).toBe(false);
  await act(async () => result.current.send('blocked', vi.fn()));
  expect(mocks.start).not.toHaveBeenCalled();
  mocks.runtime.ready = true;
  rerender();
  await waitFor(() => expect(result.current.conversation.conversationReady).toBe(true));
  expect(sessionsApi.listSessions).toHaveBeenCalledOnce();
});

it('清空先创建再删除，成功后才切换并清空历史', async () => {
  const created = deferred<Session>();
  const deleted = deferred<void>();
  const replacement = { ...session, id: 'replacement' };
  vi.spyOn(sessionsApi, 'createSession').mockReturnValue(created.promise);
  vi.spyOn(sessionsApi, 'deleteSession').mockReturnValue(deleted.promise);
  const { result } = await mountChat();
  let clearing!: Promise<void>;
  act(() => { clearing = result.current.chat.clearActiveConversation(); });
  await waitFor(() => expect(sessionsApi.createSession).toHaveBeenCalledWith(session.model, session.mode));
  expect(sessionsApi.deleteSession).not.toHaveBeenCalled();
  await act(async () => created.resolve(replacement));
  expect(sessionsApi.deleteSession).toHaveBeenCalledWith(session.id);
  expect(useConversationStore.getState().activeId).toBe(session.id);
  vi.mocked(sessionsApi.listMessages).mockReturnValue(new Promise(() => {}));
  await act(async () => { deleted.resolve(); await clearing; });
  expect(useConversationStore.getState()).toMatchObject({ activeId: replacement.id, sessions: [replacement], messages: [], responses: {}, historyLoadedId: null });
});

it.each([
  { prompt: '', excludeReasoning: false },
  { prompt: '  用户自定义提示  ', excludeReasoning: true },
])('发送仅使用显式提示 $prompt，排除思考=$excludeReasoning，响应前追加乐观消息并采用默认 token 限制', async ({ prompt, excludeReasoning }) => {
  mocks.prompt = prompt;
  mocks.excludeReasoning = excludeReasoning;
  const pending = deferred<void>();
  mocks.start.mockReturnValue(pending.promise);
  const { result } = await mountChat();
  const accepted = vi.fn();
  let sending!: Promise<void>;
  act(() => { sending = result.current.chat.send('hello', accepted); });
  await waitFor(() => expect(mocks.start).toHaveBeenCalledOnce());
  expect(mocks.start.mock.calls[0][1].include_reasoning_history).toBe(!excludeReasoning);
  expect(mocks.start).toHaveBeenCalledWith(session.id, expect.objectContaining({ expected_revision: 7, system_prompt: prompt.trim(), sampling: expect.objectContaining({ max_tokens: 4096 }), input: [{ type: 'text', text: 'hello' }] }), expect.any(Function));
  expect(useConversationStore.getState().messages).toEqual([previous, expect.objectContaining({ role: 'user', parent_id: previous.id, parts: [{ type: 'text', text: 'hello' }] })]);
  expect(accepted).not.toHaveBeenCalled();
  await act(async () => { mocks.start.mock.calls[0][2](); pending.resolve(); await sending; });
  expect(accepted).toHaveBeenCalledOnce();
});

it('图片预览上传后以带尺寸的类型化片段发送，接受后释放预览', async () => {
  const close = vi.fn();
  vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue({ width: 32, height: 24, close }));
  const revoke = vi.fn();
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, writable: true, value: vi.fn(() => 'blob:preview') });
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, writable: true, value: revoke });
  const media = { id: 'image-1' };
  vi.spyOn(mediaApi, 'uploadMedia').mockResolvedValue({ media } as Awaited<ReturnType<typeof mediaApi.uploadMedia>>);
  const { result } = await mountChat();
  const file = new File(['image'], 'photo.png', { type: 'image/png' });
  act(() => result.current.attachments.selectAttachments([file] as unknown as FileList));
  expect(result.current.attachments.getAttachments()[0]).toMatchObject({ file, kind: 'image', previewUrl: 'blob:preview' });
  await act(async () => result.current.chat.send('describe', vi.fn()));
  expect(mediaApi.uploadMedia).toHaveBeenCalledExactlyOnceWith(file);
  expect(close).toHaveBeenCalledOnce();
  expect(mocks.start.mock.calls[0][1].input).toEqual([{ type: 'image', media, width: 32, height: 24 }, { type: 'text', text: 'describe' }]);
  expect(result.current.attachments.getAttachments()).toHaveLength(1);
  act(() => mocks.start.mock.calls[0][2]());
  expect(result.current.attachments.getAttachments()).toEqual([]);
  expect(revoke).toHaveBeenCalledWith('blob:preview');
});

it('取消请求完成之前保持流连接，完成之后才中止读取', async () => {
  const cancellation = deferred<void>();
  let streamSignal!: AbortSignal;
  const cancel = vi.fn(() => cancellation.promise);
  const controller = new GenerationController({ onSynchronized: vi.fn(), onSessionState: vi.fn() }, {
    stream: (_id, _request, _receive, signal) => new Promise<void>((_resolve, reject) => {
      streamSignal = signal;
      signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true });
    }),
    cancel,
    synchronize: async () => ({ session: { ...session, state: 'idle' }, messages: [], responses: [] }),
  });
  const running = controller.start(session.id, { request_id: 'request-a' } as StreamRequest);
  const stopping = controller.stop();
  expect(cancel).toHaveBeenCalledWith(session.id, expect.any(AbortSignal));
  expect(controller.getPhase()).toBe('stopping');
  expect(streamSignal.aborted).toBe(false);
  cancellation.resolve();
  await stopping;
  expect(streamSignal.aborted).toBe(true);
  await running;
  expect(controller.getPhase()).toBe('cancelled');
});

it('同步按请求和输出消息双重定位，不能误认其他回答', async () => {
  const history = {
    session: { ...session, state: 'idle' } as Session,
    messages: [previous],
    responses: [{ request_id: 'other', status: 'completed', output_message_id: previous.id }, { request_id: 'request-a', status: 'completed', output_message_id: 'target' }] as ResponseResource[],
  };
  const synchronized = vi.fn();
  const controller = new GenerationController({ onSynchronized: synchronized, onSessionState: vi.fn() }, {
    stream: async () => {}, cancel: async () => {}, synchronize: async () => history,
  });
  await controller.start(session.id, { request_id: 'request-a' } as StreamRequest);
  expect(controller.getSnapshot().recoveryNeeded).toBe(true);
  expect(synchronized).not.toHaveBeenCalled();
  history.messages.push({ ...previous, id: 'target' });
  await controller.retrySynchronization();
  expect(controller.getPhase()).toBe('completed');
  expect(synchronized).toHaveBeenCalledExactlyOnceWith(history);
  expect(history.messages).toHaveLength(2);
});
