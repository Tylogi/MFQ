/** 向其他领域公开当前聊天模式，隐藏会话 store 的内部结构。 */
import type { SessionMode } from '../../../shared/api/types';
import { useConversationSelector } from '../state/conversationStore';

/**
 * 读取当前活动会话模式；未进入聊天或尚无活动会话时，以文本模式计算设置默认值。
 *
 * @returns 已加载活动会话的真实模式，或默认文本模式
 */
export function useActiveSessionMode(): SessionMode {
  return useConversationSelector(
    (state) => state.sessions.find((session) => session.id === state.activeId)?.mode ?? 'text',
  );
}
