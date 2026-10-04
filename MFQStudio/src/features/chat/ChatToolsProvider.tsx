/** Isolate chat-tool catalogs and selections by connection, preserving chat-tool preferences across route changes. */
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
/** Load tools on first chat access and discard tools and selections from the old service after a switch. */
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
/** Read the chat-tool catalog and selection preserved across routes. */
export function useChatTools(): ChatToolsContextValue {
  const value = useContext(ChatToolsContext);
  if (!value) throw new Error('ChatToolsProvider is missing');
  return value;
}
