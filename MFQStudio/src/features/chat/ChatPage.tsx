/** 聊天路由入口仅组合页面局部状态与各展示区域。 */
import { useChatPageState } from './hooks/useChatPageState';
import { ChatSessionSidebar } from './components/ChatSessionSidebar';
import { ChatPageHeader } from './components/ChatPageHeader';
import { ChatMessageList } from './components/ChatMessageList';
import { ChatInputArea } from './components/ChatInputArea';

/** 渲染聊天页；跨路由生成生命周期由外层 ChatProvider 保留。 */
export function ChatPage() {
  const page = useChatPageState();
  return (
    <section className={'chat-view ' + (page.chatSessionsOpen ? 'sessions-open' : 'sessions-collapsed')}>
      <ChatSessionSidebar page={page} />
      <ChatPageHeader page={page} />
      <ChatMessageList page={page} />
      <ChatInputArea page={page} />
    </section>
  );
}
