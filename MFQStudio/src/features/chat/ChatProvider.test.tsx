/**
* Tests for ChatProvider domain-state caching and prevention of unnecessary Context rerenders.
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

describe('describes ChatProvider test behavior 1', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useConversationStore.getState().reset();
    useDraftStore.setState({ drafts: {} });
    setVoiceLevel(0);
    useJobStore.getState().setJobs([]);
  });

  it('verifies ChatProvider test behavior 2', async () => {
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

  it('verifies ChatProvider test behavior 3', async () => {
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

  it('verifies ChatProvider test behavior 4', () => {
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
// Rerender the outer Provider.
    rerender();
// Action method references should remain stable to avoid unnecessary child-component rerenders.
    expect(result.current.send).toBe(initialSend);
    expect(result.current.clearActiveConversation).toBe(initialClear);
    expect(result.current.selectInteractionMode).toBe(initialSelectMode);
    expect(result.current.toggleVoice).toBe(initialToggleVoice);
  });
});
