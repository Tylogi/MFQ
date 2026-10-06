/** Compose chat sessions, message actions, attachments, and voice lifecycle, preserving generation across pages. */
import { useChatAttachments } from './hooks/useChatAttachments';
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { useLocation } from 'react-router';
import { sessionsApi } from '../../shared/api/resources/sessions';
import { runtimeApi } from '../../shared/api/resources/runtime';
import type { ContentPart, Session, SessionMode } from '../../shared/api/types';
import { studioConfirm } from '../../studio';
import { useRuntime } from '../../app/RuntimeProvider';
import { errorMessage } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';
import { useConversationSessions } from './hooks/useConversationSessions';
import { useChatGeneration } from './hooks/useChatGeneration';
import {
  ChatAttachmentsProvider,
  useChatAttachmentActions,
  useChatAttachmentError,
} from './ChatAttachmentsProvider';
import { useChatInference } from './hooks/useChatInference';
import { useMessageActions } from './hooks/useMessageActions';
import { useVoiceConversation } from '../voice/useVoiceConversation';
import { isGenerationBusy } from './state/generationController';
import { conversationActions } from './state/conversationStore';
import { ChatToolsProvider, useChatTools } from './ChatToolsProvider';
/** Load data lazily on chat access; generation and voice controllers persist when navigating to other pages. */
function useChatDomain() {
  const location = useLocation();
  const [visited, setVisited] = useState(location.pathname === '/chat');
  useEffect(() => {
    if (location.pathname === '/chat') setVisited(true);
  }, [location.pathname]);
  const { controller: generation, phase: generationPhase } = useChatGeneration({
    onSessionState: (id, state, revision) =>
      conversationActions.setSessions((current) =>
        current.map((session) => (session.id === id ? { ...session, state, revision } : session)),
      ),
    onSynchronized: ({ session, messages, responses }) =>
      conversationActions.applySynchronized(session, messages, responses),
  });
  const recoveryNeeded = generation.getSnapshot().recoveryNeeded;
  const conversation = useConversationSessions(
    visited || location.pathname === '/chat',
    isGenerationBusy(generationPhase) || recoveryNeeded,
  );
  const {
    active,
    activeId,
    activeIdRef,
    setSessions,
    setMessages,
    setResponses,
    setError,
    setActiveId,
  } = conversation;
  const { settings, tr } = useSettings();
  const runtimeContext = useRuntime();
  const { connectionRevision, ready, refreshRuntime, voiceComponent } = runtimeContext;
  const inference = useChatInference(active?.mode ?? 'text');
  const voice = useVoiceConversation(activeId, connectionRevision, setError);
  useEffect(() => {
    const controller = voice.voiceRef.current;
    if (controller && !controller.active)
      void controller.setFullDuplex(active?.mode === 'full_duplex');
  }, [active?.mode, connectionRevision, voice.voiceRef]);
  const attachments = useChatAttachmentActions();
  const attachmentError = useChatAttachmentError();
  const [operationBusy, setBusy] = useState(false);
  const [voiceComponentBusy, setVoiceComponentBusy] = useState(false);
  const { mcpTools, selectedTools, error: toolsError } = useChatTools();
  const revisionRef = useRef(connectionRevision);
  revisionRef.current = connectionRevision;
  const busy = operationBusy || conversation.transitioning || isGenerationBusy(generationPhase);
  useEffect(() => {
    generation.reset();
    setBusy(false);
  }, [activeId, connectionRevision, generation]);
  useEffect(() => {
    if (toolsError) setError(toolsError);
  }, [toolsError, setError]);
  useEffect(() => {
    if (attachmentError) setError(attachmentError);
  }, [attachmentError, setError]);

  const selectedToolsRef = useRef(selectedTools);
  selectedToolsRef.current = selectedTools;
  const mcpToolsRef = useRef(mcpTools);
  mcpToolsRef.current = mcpTools;
  const inferenceRef = useRef(inference);
  inferenceRef.current = inference;
  const voiceRef = voice.voiceRef;
  const setVoiceMessages = voice.setVoiceMessages;
  const removeSessionVoiceHistory = voice.removeSessionVoiceHistory;
  const trRef = useRef(tr);
  trRef.current = tr;
/** Build real-time configuration for the current voice connection using resolved model defaults. */
  const realtimeSessionConfig = useCallback(
    (sessionId: string) => {
      const value = inferenceRef.current.effectiveSettings;
      return {
        sessionId,
        systemPrompt: value.systemPrompt.trim(),
        temperature: value.temperature,
        topP: value.topP,
        topK: value.topK,
        repetitionPenalty: value.repetitionPenalty,
      };
    },
    [],
  );
/** Generate from text or tool results, with UI snapshots and request identity managed by a dedicated controller. */
  const generate = useCallback(
    async (
      session: Session,
      input: ContentPart[],
      optimistic = true,
      role: 'user' | 'tool' = 'user',
      onAccepted?: () => void,
    ) => {
      if (
        activeIdRef.current !== session.id ||
        isGenerationBusy(generation.getPhase()) ||
        generation.getSnapshot().recoveryNeeded
      )
        return;
      setError(null);
      if (optimistic)
        setMessages((current) => [
          ...current,
          {
            id: crypto.randomUUID(),
            role: 'user',
            parts: input,
            parent_id: current.at(-1)?.id ?? null,
            created_at: new Date().toISOString(),
          },
        ]);
      const currentSelected = selectedToolsRef.current;
      const currentMcpTools = mcpToolsRef.current;
      const inference = inferenceRef.current;
      await generation.start(session.id, {
        request_id: crypto.randomUUID(),
        expected_revision: session.revision,
        input,
        input_role: role,
        sampling: inference.sampling,
        system_prompt: inference.effectiveSettings.systemPrompt.trim(),
        include_reasoning_history: !inference.effectiveSettings.excludeReasoning,
        tools: currentMcpTools
          .filter((tool) => currentSelected.includes(tool.qualified_name))
          .map((tool) => ({
            type: 'function' as const,
            function: {
              name: tool.qualified_name,
              description: tool.description,
              parameters: tool.input_schema,
            },
          })),
        tool_choice: currentSelected.length ? 'auto' : 'none',
        stream: true,
      }, onAccepted);
      void refreshRuntime();
    },
    [
      activeIdRef,
      generation,
      refreshRuntime,
      setError,
      setMessages,
    ],
  );
/** Send after preparing attachments or voice input; discard stale results after service changes without clearing new drafts. */
  const send = useCallback(
    async (text: string, accepted: () => void) => {
      if (!active || !conversation.conversationReady || busy || recoveryNeeded) return;
      const revision = connectionRevision;
      const isCurrent = () => activeIdRef.current === active.id && revisionRef.current === revision;
      setBusy(true);
      setError(null);
      const controller = voiceRef.current;
      const resumeCapture = Boolean(controller?.capturing);
      const currentInference = inferenceRef.current;
      try {
        if (
          active.mode !== 'text' &&
          currentInference.realtimeAvailable &&
          controller &&
          !attachments.getAttachments().length
        ) {
          if (!text) return;
          await controller.submitText(text, realtimeSessionConfig(active.id));
          if (!isCurrent()) return;
          accepted();
          setVoiceMessages((current) => [
            ...current,
            {
              id: crypto.randomUUID(),
              sessionId: active.id,
              role: 'user',
              text,
              created_at: new Date().toISOString(),
            },
          ]);
          return;
        }
        if (active.mode !== 'text' && controller?.active) await controller.stop();
        const parts = await attachments.uploadAttachments();
        if (!isCurrent()) return;
        if (text) parts.push({ type: 'text', text });
        if (!parts.length) return;
        setBusy(false);
        await generate(active, parts, true, 'user', () => {
          if (!isCurrent()) return;
          accepted();
          attachments.clearAttachments();
        });
      } catch (cause) {
        if (isCurrent()) setError(errorMessage(cause));
      } finally {
        if (isCurrent()) {
          setBusy(false);
          if (resumeCapture && controller && currentInference.realtimeAvailable) {
            await controller
              .start(realtimeSessionConfig(active.id))
              .catch((cause) => setError(errorMessage(cause)));
          }
        }
      }
    },
    [
      active,
      conversation.conversationReady,
      busy,
      recoveryNeeded,
      connectionRevision,
      attachments,
      activeIdRef,
      voiceRef,
      setVoiceMessages,
      realtimeSessionConfig,
      generate,
      setError,
    ],
  );

  const messageActions = useMessageActions({
    conversation,
    blocked: busy || recoveryNeeded || !conversation.conversationReady,
    generate,
    setBusy,
  });
  const conversationView = useMemo(
    () => ({
      transitioning: conversation.transitioning,
      error: conversation.error,
      setError: conversation.setError,
      selectSession: conversation.selectSession,
      changeSessionModel: conversation.changeSessionModel,
      createSession: conversation.createSession,
      deleteSession: conversation.deleteSession,
      conversationReady: conversation.conversationReady,
      modelAvailable: conversation.modelAvailable,
    }),
    [
      conversation.transitioning,
      conversation.error,
      conversation.setError,
      conversation.selectSession,
      conversation.changeSessionModel,
      conversation.createSession,
      conversation.deleteSession,
      conversation.conversationReady,
      conversation.modelAvailable,
    ],
  );
/** Confirm sidebar-session deletion and remove local voice history after server-side success. */
  const deleteConversation = useCallback(async (id: string) => {
    const session = conversation.sessions.find((item) => item.id === id);
    if (!session || busy || recoveryNeeded) return;
    const title = session.title || trRef.current('未命名会话', 'Untitled chat');
    if (!(await studioConfirm(trRef.current(
      `删除对话“${title}”？此操作无法撤销。`,
      `Delete "${title}"? This cannot be undone.`,
    )))) return;
    if (await conversation.deleteSession(id)) {
      await removeSessionVoiceHistory(id).catch((cause) => setError(errorMessage(cause)));
    }
  }, [conversation.sessions, conversation.deleteSession, busy, recoveryNeeded, removeSessionVoiceHistory, setError]);
/** Replace the old session with a new one after confirmation, also removing the associated voice history. */
  const clearActiveConversation = useCallback(async () => {
    if (
      !active ||
      busy ||
      recoveryNeeded ||
      !(await studioConfirm(trRef.current('清空当前对话？', 'Clear this conversation?')))
    )
      return;
    setBusy(true);
    try {
      const replacement = await sessionsApi.createSession(active.model, active.mode);
      await sessionsApi.deleteSession(active.id);
      if (activeIdRef.current !== active.id) return;
      setSessions((current) => [
        replacement,
        ...current.filter((session) => session.id !== active.id),
      ]);
      const clipCleanup = removeSessionVoiceHistory(active.id);
      setActiveId(replacement.id);
      await clipCleanup;
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }, [
    active,
    busy,
    recoveryNeeded,
    activeIdRef,
    setSessions,
    removeSessionVoiceHistory,
    setActiveId,
    setError,
  ]);
/** Stop audio before changing a session’s interaction mode, then update the session revision returned by the server. */
  const selectInteractionMode = useCallback(
    async (mode: SessionMode) => {
      if (!active || busy || recoveryNeeded || active.mode === mode) return;
      setBusy(true);
      try {
        await voiceRef.current?.stop();
        const updated = await sessionsApi.updateSession(active.id, { mode });
        setSessions((current) =>
          current.map((session) => (session.id === updated.id ? updated : session)),
        );
      } catch (cause) {
        setError(errorMessage(cause));
      } finally {
        setBusy(false);
      }
    },
    [active, busy, recoveryNeeded, voiceRef, setSessions, setError],
  );
/** Toggle microphone capture for the current session, leaving errors to the chat error area. */
  const toggleVoice = useCallback(async () => {
    if (!active || active.mode === 'text' || !inferenceRef.current.realtimeAvailable || busy || recoveryNeeded) return;
    await voiceRef.current
      ?.toggleCapture(realtimeSessionConfig(active.id))
      .catch((cause) => setError(errorMessage(cause)));
  }, [active, busy, recoveryNeeded, voiceRef, realtimeSessionConfig, setError]);
/** Explicitly download or enable the voice component and refresh shared job state after submission. */
  const installOrEnableVoiceOutput = useCallback(async () => {
    if (voiceComponentBusy) return;
    setVoiceComponentBusy(true);
    try {
      if (voiceComponent?.ready) {
        const result = await runtimeApi.activateVoiceOutputComponent();
        if (!result.active)
          throw new Error(result.error || result.reason || 'Voice output activation failed');
      } else await runtimeApi.installVoiceOutputComponent();
      await refreshRuntime(false);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setVoiceComponentBusy(false);
    }
  }, [voiceComponentBusy, voiceComponent?.ready, refreshRuntime, setError]);

  return useMemo(
    () => ({
      conversation: conversationView,
      inference,
      voice,
      messageActions,
      generation,
      generationPhase,
      busy,
      recoveryNeeded,
      send,
      clearActiveConversation,
      deleteConversation,
      selectInteractionMode,
      toggleVoice,
      voiceComponentBusy,
      installOrEnableVoiceOutput,
    }),
    [
      conversationView,
      inference,
      voice,
      messageActions,
      generation,
      generationPhase,
      busy,
      recoveryNeeded,
      send,
      clearActiveConversation,
      deleteConversation,
      selectInteractionMode,
      toggleVoice,
      voiceComponentBusy,
      installOrEnableVoiceOutput,
    ],
  );
}

const ChatContext = createContext<ReturnType<typeof useChatDomain> | null>(null);

/**
* Keep the chat-domain instance alive so page unmount does not cancel ongoing text generation.
 *
* @param props Component properties, including children
 */
export function ChatProvider({ children }: { children: ReactNode }) {
  return (
    <ChatAttachmentsProvider>
      <ChatToolsProvider>
        <ChatLifecycleProvider>{children}</ChatLifecycleProvider>
      </ChatToolsProvider>
    </ChatAttachmentsProvider>
  );
}
/** Keep generation and voice controllers mounted so chat-route changes do not cancel ongoing requests. */
function ChatLifecycleProvider({ children }: { children: ReactNode }) {
  const value = useChatDomain();
  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

/**
* Read the chat-domain interface from the chat page or toolbar.
 *
* @returns Chat-domain context containing sessions, inference, voice, generation, and tool actions
 */
export function useChat() {
  const value = useContext(ChatContext);
  if (!value) throw new Error('ChatProvider is missing');
  return value;
}
