/** 管理会话列表、选中会话与消息历史，所有异步回写绑定当前连接和会话。 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { sessionsApi } from '../../../shared/api/resources/sessions';
import type { SessionMode } from '../../../shared/api/types';
import { useRuntime } from '../../../app/RuntimeProvider';
import { errorMessage } from '../../../app/formatters';
import { useConversationStore } from '../state/conversationStore';

/** 首次打开聊天才加载会话，切换时取消旧历史请求，跨页面保留已加载状态。 */
export function useConversationSessions(enabled: boolean, generationBusy: boolean) {
  const { ready, connectionRevision, selectedModel, setSelectedModel, models, instances } =
    useRuntime();
  const sessions = useConversationStore((state) => state.sessions);
  const activeId = useConversationStore((state) => state.activeId);
  const messages = useConversationStore((state) => state.messages);
  const responses = useConversationStore((state) => state.responses);
  const historyLoadedId = useConversationStore((state) => state.historyLoadedId);
  const setSessions = useConversationStore((state) => state.setSessions);
  const setActiveId = useConversationStore((state) => state.setActiveId);
  const setMessages = useConversationStore((state) => state.setMessages);
  const setResponses = useConversationStore((state) => state.setResponses);
  const [transitioning, setTransitioning] = useState(false);
  const [importRevision, setImportRevision] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const version = useRef(0);
  const activeIdRef = useRef(activeId);
  activeIdRef.current = activeId;
  const active = sessions.find((session) => session.id === activeId) ?? null;
  const modelAvailable =
    models.some((model) => model.id === selectedModel) ||
    instances.some(
      (instance) => instance.model === selectedModel && ['ready', 'busy'].includes(instance.state),
    );
  useEffect(() => {
    const refresh = () => setImportRevision((current) => current + 1);
    window.addEventListener('mfq:sessions-imported', refresh);
    return () => window.removeEventListener('mfq:sessions-imported', refresh);
  }, []);

  useEffect(() => {
    const request = ++version.current;
    const store = useConversationStore.getState();
    store.reset();
    const epoch = useConversationStore.getState().epoch;
    setError(null);
    if (!ready || !enabled) return;
  void sessionsApi
      .listSessions()
      .then((next) => {
        if (request !== version.current || !useConversationStore.getState().loadSessions(epoch, next)) return;
        const selected = next[0];
        if (selected) setSelectedModel(selected.model);
      })
      .catch((cause) => {
        if (request === version.current) setError(errorMessage(cause));
      });
    return () => {
      ++version.current;
    };
  }, [ready, enabled, connectionRevision, importRevision, setSelectedModel]);

  useEffect(() => {
    if (!activeId) return;
    const store = useConversationStore.getState();
    if (store.activeId !== activeId) return;
    const epoch = store.epoch;
    store.beginHistory(activeId);
    const controller = new AbortController();
    void Promise.all([
      sessionsApi.listMessages(activeId, controller.signal),
      sessionsApi.listResponses(activeId, controller.signal),
    ])
      .then(([nextMessages, nextResponses]) => {
        if (controller.signal.aborted) return;
        useConversationStore.getState().applyHistory(epoch, activeId, nextMessages, nextResponses);
      })
      .catch((cause) => {
        if (
          !controller.signal.aborted &&
          epoch === useConversationStore.getState().epoch &&
          activeId === useConversationStore.getState().activeId
        ) setError(errorMessage(cause));
      });
    return () => controller.abort();
  }, [activeId, connectionRevision]);

  useEffect(() => {
    if (!active || generationBusy || !modelAvailable || active.model === selectedModel) return;
    let current = true;
    setTransitioning(true);
    const epoch = useConversationStore.getState().epoch;
  void sessionsApi
      .forkSession(active.id, null, true, active.title, selectedModel)
      .then((replacement) => {
        if (!current || epoch !== useConversationStore.getState().epoch || useConversationStore.getState().activeId !== active.id) return;
        setSessions((existing) => [replacement, ...existing]);
        setActiveId(replacement.id);
      })
      .catch((cause) => {
        if (current && epoch === useConversationStore.getState().epoch) setError(errorMessage(cause));
      })
      .finally(() => {
        if (current) setTransitioning(false);
      });
    return () => {
      current = false;
      setTransitioning(false);
    };
  }, [active?.id, active?.model, active?.title, selectedModel, modelAvailable, generationBusy]);

  /** 切换会话及其绑定模型，历史请求在 effect 中按会话重建。 */
  const selectSession = useCallback(
    (id: string) => {
      const session = useConversationStore.getState().sessions.find((candidate) => candidate.id === id);
      if (!session || transitioning) return;
      setSelectedModel(session.model);
      setActiveId(id);
    },
    [transitioning, setSelectedModel, setActiveId],
  );

  /** 创建新的空会话，版本戳防止服务切换后的返回污染当前列表。 */
  const createSession = useCallback(
    async (mode: SessionMode = 'text') => {
      if (!selectedModel || transitioning) return;
      const request = version.current;
      const epoch = useConversationStore.getState().epoch;
      setTransitioning(true);
      try {
        const created = await sessionsApi.createSession(selectedModel, mode);
        if (request !== version.current || epoch !== useConversationStore.getState().epoch) return;
        setSessions((current) => [created, ...current]);
        setActiveId(created.id);
      } catch (cause) {
        if (request === version.current && epoch === useConversationStore.getState().epoch) setError(errorMessage(cause));
      } finally {
        if (request === version.current) setTransitioning(false);
      }
    },
    [selectedModel, transitioning, setSessions, setActiveId],
  );

  /** 删除指定会话；仅在当前连接仍有效时更新列表和当前历史。 */
  const deleteSession = useCallback(
    async (id: string): Promise<boolean> => {
      if (generationBusy || transitioning || !useConversationStore.getState().sessions.some((session) => session.id === id)) return false;
      const request = version.current;
      const epoch = useConversationStore.getState().epoch;
      setTransitioning(true);
      setError(null);
      try {
        await sessionsApi.deleteSession(id);
        if (request !== version.current || epoch !== useConversationStore.getState().epoch) return false;
        const state = useConversationStore.getState();
        const remaining = state.sessions.filter((session) => session.id !== id);
        setSessions(remaining);
        if (state.activeId === id) {
          const next = remaining[0];
          if (next) setSelectedModel(next.model);
          setActiveId(next?.id ?? null);
        }
        return true;
      } catch (cause) {
        if (request === version.current && epoch === useConversationStore.getState().epoch) setError(errorMessage(cause));
        return false;
      } finally {
        if (request === version.current) setTransitioning(false);
      }
    },
    [generationBusy, transitioning, setSessions, setActiveId, setSelectedModel],
  );

  return {
    sessions,
    setSessions,
    active,
    activeId,
    setActiveId,
    activeIdRef,
    messages,
    setMessages,
    responses,
    setResponses,
    transitioning,
    error,
    setError,
    selectSession,
    createSession,
    deleteSession,
    conversationReady: Boolean(
      active && modelAvailable && active.model === selectedModel && historyLoadedId === activeId,
    ),
    modelAvailable,
  };
}
