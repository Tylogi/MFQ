/** 保存跨路由会话与历史数据，并以版本戳隔离已失效的异步回写。 */
import { create } from 'zustand';
import type { Dispatch, SetStateAction } from 'react';
import type { Message, ResponseResource, Session } from '../../../shared/api/types';

type Responses = Record<string, ResponseResource>;

/** 会话业务状态及供聊天领域调用的语义动作。 */
export interface ConversationState {
  sessions: Session[];
  activeId: string | null;
  messages: Message[];
  responses: Responses;
  historyLoadedId: string | null;
  epoch: number;
  /** 清除连接相关数据，并使之前发出的异步请求失效。 */
  reset: () => void;
  /** 兼容现有调用方的函数式会话列表更新。 */
  setSessions: Dispatch<SetStateAction<Session[]>>;
  /** 切换会话时同步清除上一会话的历史。 */
  setActiveId: Dispatch<SetStateAction<string | null>>;
  /** 兼容现有调用方的函数式消息更新。 */
  setMessages: Dispatch<SetStateAction<Message[]>>;
  /** 兼容现有调用方的函数式响应更新。 */
  setResponses: Dispatch<SetStateAction<Responses>>;
  /** 在列表请求仍属于当前连接时应用首个会话。 */
  loadSessions: (epoch: number, sessions: Session[]) => boolean;
  /** 清除当前会话的旧历史，等待新请求完成。 */
  beginHistory: (id: string) => void;
  /** 仅在连接及会话均未变化时写入历史。 */
  applyHistory: (
    epoch: number,
    id: string,
    messages: Message[],
    responses: ResponseResource[],
  ) => void;
  /** 同步生成返回的会话版本，仅在目标会话当前激活时覆盖消息历史。 */
  applySynchronized: (session: Session, messages: Message[], responses: ResponseResource[]) => void;
}

/** 将服务端响应按其输出消息标识索引。 */
function indexResponses(responses: ResponseResource[]): Responses {
  return Object.fromEntries(
    responses
      .filter((response) => response.output_message_id)
      .map((response) => [response.output_message_id!, response]),
  );
}

/** 聊天会话的共享 store，组件可按单一字段订阅以避免无关更新。 */
export const useConversationStore = create<ConversationState>()((set) => ({
  sessions: [],
  activeId: null,
  messages: [],
  responses: {},
  historyLoadedId: null,
  epoch: 0,
  reset: () =>
    set((state) => ({
      sessions: [],
      activeId: null,
      messages: [],
      responses: {},
      historyLoadedId: null,
      epoch: state.epoch + 1,
    })),
  setSessions: (value) =>
    set((state) => ({
      sessions: typeof value === 'function' ? value(state.sessions) : value,
    })),
  setActiveId: (value) =>
    set((state) => {
      const activeId = typeof value === 'function' ? value(state.activeId) : value;
      return activeId === state.activeId
        ? state
        : {
            activeId,
            messages: [],
            responses: {},
            historyLoadedId: null,
          };
    }),
  setMessages: (value) =>
    set((state) => ({
      messages: typeof value === 'function' ? value(state.messages) : value,
    })),
  setResponses: (value) =>
    set((state) => ({
      responses: typeof value === 'function' ? value(state.responses) : value,
    })),
  loadSessions: (epoch, sessions) => {
    if (useConversationStore.getState().epoch !== epoch) return false;
    set({
      sessions,
      activeId: sessions[0]?.id ?? null,
      messages: [],
      responses: {},
      historyLoadedId: null,
    });
    return true;
  },
  beginHistory: (id) =>
    set((state) =>
      state.activeId === id ? { messages: [], responses: {}, historyLoadedId: null } : state,
    ),
  applyHistory: (epoch, id, messages, responses) =>
    set((state) =>
      state.epoch === epoch && state.activeId === id
        ? { messages, responses: indexResponses(responses), historyLoadedId: id }
        : state,
    ),
  applySynchronized: (session, messages, responses) =>
    set((state) => ({
      sessions: state.sessions.map((item) => (item.id === session.id ? session : item)),
      ...(state.activeId === session.id ? { messages, responses: indexResponses(responses) } : {}),
    })),
}));

/** 按 selector 订阅单个会话字段，供逐步移除 Context 响应式传播的页面使用。 */
export function useConversationSelector<T>(selector: (state: ConversationState) => T): T {
  return useConversationStore(selector);
}

/** 供父级业务回调直接调用的稳定动作，无需订阅会话实体。 */
export const conversationActions = {
  reset: useConversationStore.getState().reset,
  applySynchronized: useConversationStore.getState().applySynchronized,
  applySync: useConversationStore.getState().applySynchronized,
  setSessions: useConversationStore.getState().setSessions,
  setActiveId: useConversationStore.getState().setActiveId,
  setMessages: useConversationStore.getState().setMessages,
  setResponses: useConversationStore.getState().setResponses,
};
