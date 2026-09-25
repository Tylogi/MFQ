/** 按连接隔离聊天工具目录与选择，切路由时保留当前聊天工具偏好。 */
import {
  createContext,
  useEffect,
  useMemo,
  useContext,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from 'react';
import { useLocation } from 'react-router';
import { connectionsApi } from '../../shared/api/resources/connections';
import type { McpToolResource } from '../../shared/api/types';
import { useRuntime } from '../../app/RuntimeProvider';
import { errorMessage } from '../../app/formatters';

interface ChatToolsContextValue {
  mcpTools: McpToolResource[];
  selectedTools: string[];
  setSelectedTools: Dispatch<SetStateAction<string[]>>;
  error: string | null;
}

const ChatToolsContext = createContext<ChatToolsContextValue | null>(null);

/** 在首次访问聊天后加载工具，并在服务切换时丢弃旧服务的工具与选择。 */
export function ChatToolsProvider({ children }: { children: ReactNode }) {
  const location = useLocation();
  const { ready, connectionRevision } = useRuntime();
  const [visited, setVisited] = useState(location.pathname === '/chat');
  const [mcpTools, setMcpTools] = useState<McpToolResource[]>([]);
  const [selectedTools, setSelectedTools] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (location.pathname === '/chat') setVisited(true);
  }, [location.pathname]);

  useEffect(() => {
    setMcpTools([]);
    setSelectedTools([]);
    setError(null);
    if (!visited || !ready) return;
    let current = true;
    const refresh = () => {
      void connectionsApi
        .mcpTools()
        .then((result) => {
          if (current) setMcpTools(result.data);
        })
        .catch((cause) => {
          if (current) setError(errorMessage(cause));
        });
    };
    refresh();
    window.addEventListener('mfq:tools-changed', refresh);
    return () => {
      current = false;
      window.removeEventListener('mfq:tools-changed', refresh);
    };
  }, [visited, ready, connectionRevision]);

  const value = useMemo(
    () => ({ mcpTools, selectedTools, setSelectedTools, error }),
    [mcpTools, selectedTools, error],
  );
  return <ChatToolsContext.Provider value={value}>{children}</ChatToolsContext.Provider>;
}

/** 读取跨路由保留的聊天工具目录与选择。 */
export function useChatTools(): ChatToolsContextValue {
  const value = useContext(ChatToolsContext);
  if (!value) throw new Error('ChatToolsProvider is missing');
  return value;
}
