/** 组合聊天领域的会话、消息操作、附件和语音生命周期，跨页面保留进行中的生成。 */
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

/** 按聊天访问惰性加载数据，生成和语音控制器不会因为切换其他页面而丢失。 */
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
  const conversation = useConversationSessions(
    visited || location.pathname === '/chat',
    isGenerationBusy(generationPhase),
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
  const busy = operationBusy || isGenerationBusy(generationPhase);
  const recoveryNeeded = generation.getSnapshot().recoveryNeeded;
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
  const trRef = useRef(tr);
  trRef.current = tr;

  /** 为当前语音连接生成实时配置，使用解析后的模型默认设置。 */
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

  /** 发起文本或工具结果生成，UI 快照与请求身份由独立控制器管理。 */
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

  /** 准备附件或语音输入后发送；服务切换后丢弃旧操作的返回，不清除新草稿。 */
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
      conversation.createSession,
      conversation.deleteSession,
      conversation.conversationReady,
      conversation.modelAvailable,
    ],
  );

  /** 确认删除侧栏会话，并在服务端成功后移除本地语音历史。 */
  const deleteConversation = useCallback(async (id: string) => {
    const session = conversation.sessions.find((item) => item.id === id);
    if (!session || busy || conversation.transitioning) return;
    const title = session.title || trRef.current('未命名会话', 'Untitled chat');
    if (!(await studioConfirm(trRef.current(
      `删除对话“${title}”？此操作无法撤销。`,
      `Delete "${title}"? This cannot be undone.`,
    )))) return;
    if (await conversation.deleteSession(id)) {
      setVoiceMessages((current) => current.filter((message) => message.sessionId !== id));
    }
  }, [conversation.sessions, conversation.transitioning, conversation.deleteSession, busy, setVoiceMessages]);

  /** 用户确认后用新会话替换旧会话，同时移除对应语音历史。 */
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
      setVoiceMessages((current) =>
        current.filter((message) => message.sessionId !== active.id),
      );
      setActiveId(replacement.id);
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
    setVoiceMessages,
    setActiveId,
    setError,
  ]);

  /** 切换会话交互模式前停止音频，成功后更新服务返回的会话版本。 */
  const selectInteractionMode = useCallback(
    async (mode: SessionMode) => {
      if (!active || busy || active.mode === mode) return;
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
    [active, busy, voiceRef, setSessions, setError],
  );

  /** 根据当前会话切换麦克风采集，异常交给聊天错误区域展示。 */
  const toggleVoice = useCallback(async () => {
    if (!active || active.mode === 'text' || !inferenceRef.current.realtimeAvailable || busy) return;
    await voiceRef.current
      ?.toggleCapture(realtimeSessionConfig(active.id))
      .catch((cause) => setError(errorMessage(cause)));
  }, [active, busy, voiceRef, realtimeSessionConfig, setError]);

  /** 显式下载或启用语音组件，提交后刷新共享任务状态。 */
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
 * 维持聊天领域实例，页面卸载不会取消正在进行的文本生成。
 *
 * @param props 组件属性，包含子节点
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

/** 保持生成与语音控制器挂载，不让聊天路由切换取消正在进行的请求。 */
function ChatLifecycleProvider({ children }: { children: ReactNode }) {
  const value = useChatDomain();
  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

/**
 * 从聊天页面或工具栏读取聊天领域接口。
 *
 * @returns 聊天领域上下文，包含会话、推理、语音、生成与工具操作
 */
export function useChat() {
  const value = useContext(ChatContext);
  if (!value) throw new Error('ChatProvider is missing');
  return value;
}
