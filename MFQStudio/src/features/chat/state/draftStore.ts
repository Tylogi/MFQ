/** 按会话隔离输入草稿，保存在内存中并避免键入触发应用外壳渲染。 */
import { create } from 'zustand';

interface DraftState {
  drafts: Record<string, string>;
  /** 更新指定会话草稿；空值移除记录，不持久化可能包含隐私的输入。 */
  setDraft: (sessionId: string, text: string) => void;
}

/** 只在输入组件订阅对应会话的字符串，页面切换时保留未发送内容。 */
export const useDraftStore = create<DraftState>()((set) => ({
  drafts: {},
  setDraft: (sessionId, text) =>
    set((state) => {
      if ((state.drafts[sessionId] ?? '') === text) return state;
      const drafts = { ...state.drafts };
      if (text) drafts[sessionId] = text;
      else delete drafts[sessionId];
      return { drafts };
    }),
}));
