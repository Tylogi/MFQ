/**
 * ChatProvider 领域状态缓存与 Context 重渲染拦截测试。
 */

import { act, render, renderHook, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Profiler, type ProfilerOnRenderCallback, type ReactNode } from 'react';
import { MemoryRouter } from 'react-router';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ChatProvider, useChat } from './ChatProvider';
import { useConversationStore } from './state/conversationStore';
import { useDraftStore } from './state/draftStore';
import { setVoiceLevel } from '../voice/voiceLevelStore';
import { ChatComposer } from './components/ChatComposer';
import { ChatToolbar } from './components/ChatToolbar';
import { SavedMessageList } from './SavedMessageList';
import type { JobResource, Message } from '../../shared/api/types';
import { TooltipProvider } from '../../shared/ui/Tooltip';
import { useJobStore } from '../../stores/jobStore';

const mockRuntime = {
  runtime: null,
  models: [],
  instances: [],
  jobs: [],
  capabilities: null,
  realtime: null,
  voiceComponent: null,
  studio: null,
  selectedModel: '',
  setSelectedModel: vi.fn(),
  refreshRuntime: vi.fn().mockResolvedValue(undefined),
  addJob: vi.fn(),
  reloadService: vi.fn().mockResolvedValue(undefined),
  connectionRevision: 1,
  ready: true,
  loading: false,
  error: null,
};

vi.mock('../../app/RuntimeProvider', () => ({
  useRuntime: () => mockRuntime,
}));

const mockSettings = {
  inheritModelDefaults: true,
  systemPrompt: '',
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  repetitionPenalty: 1.1,
  maxTokens: 2048,
};
const mockUpdateSettings = vi.fn();
const mockTr = (_zh: string, en: string) => en;

vi.mock('../settings/SettingsProvider', () => ({
  useSettings: () => ({
    settings: mockSettings,
    updateSettings: mockUpdateSettings,
    tr: mockTr,
  }),
}));

vi.mock('../../shared/api/resources/sessions', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../shared/api/resources/sessions')>();
  return {
    ...actual,
    sessionsApi: {
      ...actual.sessionsApi,
      listSessions: vi.fn().mockResolvedValue([]),
      listMessages: vi.fn().mockResolvedValue([]),
      listResponses: vi.fn().mockResolvedValue([]),
    },
  };
});
vi.mock('../../shared/api/resources/connections', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../shared/api/resources/connections')>();
  return {
    ...actual,
    connectionsApi: {
      ...actual.connectionsApi,
      mcpTools: vi.fn().mockResolvedValue({ data: [] }),
    },
  };
});

vi.mock('../../studio', () => ({
  studioConfirm: vi.fn().mockResolvedValue(true),
}));

describe('ChatProvider Context 重渲染拦截与缓存', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useConversationStore.getState().reset();
    useDraftStore.setState({ drafts: {} });
    setVoiceLevel(0);
    useJobStore.getState().setJobs([]);
  });

  it('Profiler 计数表明草稿与音量更新不会提交无关历史列表', async () => {
    const commits = { toolbar: 0, history: 0 };
    const onRender: ProfilerOnRenderCallback = (id, phase) => {
      if (phase !== 'mount' && (id === 'toolbar' || id === 'history')) commits[id] += 1;
    };
    function History() {
      const messages = useConversationStore((state) => state.messages);
      const { busy } = useChat();
      return (
        <SavedMessageList
          messages={messages}
          responses={{}}
          mcpTools={[]}
          busy={busy}
          tr={mockTr}
          editDraft={null}
          setEditDraft={vi.fn()}
          actions={{ saveEdit: vi.fn(), copyMessage: vi.fn(), regenerate: vi.fn(), executeToolCalls: vi.fn() }}
        />
      );
    }
    function Surface() {
      const { busy, generationPhase, recoveryNeeded } = useChat();
      return (
        <>
          <Profiler id="history" onRender={onRender}><History /></Profiler>
          <ChatComposer
            sessionId="profile-session"
            ready
            busy={busy}
            recoveryNeeded={recoveryNeeded}
            phase={generationPhase}
            placeholder=""
            attachmentAccept=""
            tr={mockTr}
            onSend={vi.fn()}
            onStop={vi.fn()}
            onError={vi.fn()}
            toolbar={<Profiler id="toolbar" onRender={onRender}><ChatToolbar /></Profiler>}
          />
        </>
      );
    }
    render(
      <MemoryRouter initialEntries={['/chat']}>
        <TooltipProvider><ChatProvider><Surface /></ChatProvider></TooltipProvider>
      </MemoryRouter>,
    );
    await act(async () => { await Promise.resolve(); });
    const baseline = { ...commits };
    await userEvent.setup().type(screen.getByRole('textbox', { name: 'Message' }), 'draft');
    expect(commits).toEqual(baseline);

    const message = {
      id: 'history-1', role: 'user', parts: [{ type: 'text', text: 'saved' }], created_at: '2026-09-23T00:00:00Z',
    } as Message;
    act(() => useConversationStore.getState().setMessages([message]));
    expect(commits.history).toBeGreaterThan(baseline.history);
    expect(commits.toolbar).toBe(baseline.toolbar);

    const afterHistory = { ...commits };
    for (let index = 1; index <= 10; index += 1)
      act(() => setVoiceLevel(index / 10));
    expect(commits.toolbar - afterHistory.toolbar).toBe(10);
    expect(commits.history).toBe(afterHistory.history);

    const afterVoice = { ...commits };
    for (let index = 1; index <= 10; index += 1)
      act(() => useJobStore.getState().setJobs([{
        id: 'job-a', kind: 'quantization', status: 'running', payload: {}, progress: index / 10,
      } as JobResource]));
    expect(commits).toEqual(afterVoice);
  });

  it('消息写入 store 时不会更新只订阅聊天命令的 Context 消费者', async () => {
    let renders = 0;
    function Commands() {
      useChat();
      renders += 1;
      return null;
    }
    render(
      <MemoryRouter initialEntries={['/chat']}>
        <ChatProvider><Commands /></ChatProvider>
      </MemoryRouter>,
    );
    await act(async () => {
      await Promise.resolve();
    });
    const before = renders;
    act(() => useConversationStore.getState().setMessages([]));
    expect(renders).toBe(before);
  });

  it('ChatProvider 提供 useChat 上下文且导出的业务动作引用保持稳定', () => {
    const wrapper = ({ children }: { children: ReactNode }) => (
      <MemoryRouter initialEntries={['/chat']}>
        <ChatProvider>{children}</ChatProvider>
      </MemoryRouter>
    );

    const { result, rerender } = renderHook(() => useChat(), { wrapper });

    expect(result.current).toBeDefined();
    const initialSend = result.current.send;
    const initialClear = result.current.clearActiveConversation;
    const initialSelectMode = result.current.selectInteractionMode;
    const initialToggleVoice = result.current.toggleVoice;

    // 重新渲染 Provider 外层
    rerender();

    // 动作方法引用应当保持一致，避免子组件非必要重渲染
    expect(result.current.send).toBe(initialSend);
    expect(result.current.clearActiveConversation).toBe(initialClear);
    expect(result.current.selectInteractionMode).toBe(initialSelectMode);
    expect(result.current.toggleVoice).toBe(initialToggleVoice);
  });
});
